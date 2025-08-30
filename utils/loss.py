import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma, gammaln  # pytorch1.8+支持
import math
from config import Config

"""
变分高斯混合模型损失函数实现

变量定义（与loss.md保持一致）：
- π_{ik}: 变分后验概率 q(z_{ik}=1|x)，表示像素i属于类别k的变分后验概率
- α_{ik}/d_{ik}: Dirichlet浓度参数
- μ_{ik}, σ²_{ik}: 高斯分量的均值和方差
- w_{ik}: Dirichlet权重参数（局部混合权重）

注意：在不同函数中π_{ik}的使用：
1. 损失计算中：π_{ik}作为变分后验概率（符合loss.md定义）
2. compute_gmm_responsibilities中：π_{ik}被用作混合权重来计算GMM责任度
"""


def reconstruction_loss(x, mu, var, pi):
    """
    计算重建损失项L₁ = -E_q[log P(x|z,Ω)]
    
    Args:
        x: [B, 1, C, H, W] - 观测特征
        mu: [B, K, C, H, W] - 高斯分量均值
        var: [B, K, C, H, W] - 高斯分量方差  
        pi: [B, K, H, W] - 变分后验概率π_{ik} (按loss.md定义)
        
    Returns:
        scalar - 重建损失
        
    公式对应loss.md中的L₁，利用E[z_{ik}] = π_{ik}
    """
    log_exp = -0.5 * torch.sum((x - mu) ** 2 / var, dim=2)  # [B, K, H, W]
    log_norm = -0.5 * torch.sum(torch.log(2 * math.pi * var), dim=2)  # [B, K, H, W]
    log_gauss = log_exp + log_norm
    recon_loss = pi * log_gauss  # π_{ik} 作为权重
    return -torch.mean(torch.sum(recon_loss, dim=1))


def kl_categorical_loss(pi, alpha):
    """
    计算类别分布的KL散度损失项L₂ = KL(q(z|x) || p(z|Ω))

    Args:
        pi: [B, K, H, W] - 变分后验概率π_{ik} = q(z_{ik}=1|x) (按loss.md定义)
        alpha: [B, K, H, W] - Dirichlet浓度参数，用于计算先验期望E[log w_{ik}]
    
    Returns:
        scalar - KL散度损失
    
    公式对应loss.md中的L₂：
    L₂ = Σᵢₖ π_{ik} [log π_{ik} - (ψ(α_{ik}) - ψ(Σₖ'α_{ik'}))]
    其中 E[log w_{ik}] = ψ(α_{ik}) - ψ(Σₖ'α_{ik'})
    """
    
    log_pi = torch.log(pi)
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True) # [B, 1, H, W]
    digamma_diff = digamma(alpha) - digamma(alpha_sum)  # E[log w_{ik}]
    kl = torch.sum(pi * (log_pi - digamma_diff), dim=1)  # [B, H, W]

    return kl.mean()


def kl_dirichlet_loss(q_alpha, p_alpha):
    """KL(Dir(q_alpha) || Dir(p_alpha))
    Args:
        q_alpha: [B,K,H,W] 预测 Dirichlet 浓度
        p_alpha: [B,K,H,W] 先验 Dirichlet 浓度
    Returns:
        标量: 所有 batch 与空间位置的平均 KL
    Note:
        若需返回逐像素 KL, 可扩展添加 reduction 参数。
    """
    q_alpha_sum = torch.sum(q_alpha, dim=1, keepdim=True)  # [B, 1, H, W]
    p_alpha_sum = torch.sum(p_alpha, dim=1, keepdim=True)  # [B, 1, H, W]

    logB_q = torch.sum(gammaln(q_alpha), dim=1) - gammaln(q_alpha_sum.squeeze(1))  # [B, H, W]
    logB_p = torch.sum(gammaln(p_alpha), dim=1) - gammaln(p_alpha_sum.squeeze(1))  # [B, H, W]

    digamma_q_alpha = digamma(q_alpha)
    digamma_q_alpha_sum = digamma(q_alpha_sum)

    term = (q_alpha - p_alpha) * (digamma_q_alpha - digamma_q_alpha_sum)
    term_sum = torch.sum(term, dim=1)  # [B, H, W]

    kl = logB_p - logB_q + term_sum  # [B, H, W]

    return kl.mean()


class GmmLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.k = config.GMM_NUM
        self.config = config

    def forward(self, input, mu, var, pi, alpha, prior, epoch, total_epochs):

        B, C, H, W = input.shape
        x = input.unsqueeze(1)  # [B, 1, C, H, W]

        # Reshape
        mu  = mu.reshape(B, self.k, C, H, W)
        var = var.reshape(B, self.k, C, H, W)
        pi  = pi.reshape(B, self.k, H, W)
        alpha   = alpha.reshape(B, self.k, H, W)
        prior  = prior.reshape(B, self.k, H, W)

        # ----------------------
        # 1) Reconstruction term: -E_q(z|x) [ log N(x|mu,var) ]
        recon_loss = reconstruction_loss(x, mu, var, pi)

        # ----------------------
        # 2) KL(q(z|x) || p(z|Ω))
        kl_z_loss = kl_categorical_loss(pi, alpha)
        
        # ----------------------
        # 3) KL(Dir(alpha) || Dir(prior)), exact closed form
        kl_o_loss = kl_dirichlet_loss(alpha, prior)


        weight_1 = 1.0
        weight_2 = 1.0
        weight_3 = 1.0
        
        if epoch > 0.1 * total_epochs:
            weight_3 = 1.0 - 0.99 * ((epoch + 1) / total_epochs)


        loss_1 = recon_loss * weight_1
        loss_2 = kl_z_loss  * weight_2
        loss_3 = kl_o_loss  * weight_3

        loss_mse = nn.MSELoss()(alpha, prior) * 0.0

        total_loss = loss_1 + loss_2 + loss_3 + loss_mse


        # 字典返回形式
        return {
            'total': total_loss,
            'recon': loss_1,
            'kl_pi': loss_2,          # KL(q(z|x) || p(z|Ω))
            'kl_dir': loss_3,         # KL(Dir(alpha) || Dir(prior))，已乘以动态权重
            'loss_mse': loss_mse,
            'weight_2': weight_2,
            'weight_3': weight_3,
        }













