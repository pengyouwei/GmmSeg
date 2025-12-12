import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma, gammaln  # pytorch1.8+支持
import math
from config import Config


def reconstruction_loss(x, mu, var, r, eps: float = 1e-6):
    var = var.clamp_min(eps)

    mu_part = -0.5 * torch.sum((x - mu) ** 2 / var, dim=2)  # [B, K, H, W]
    var_part = -0.5 * torch.sum(torch.log(2 * math.pi * var), dim=2)  # [B, K, H, W]
    
    # 完整的高斯对数似然
    log_gauss = mu_part + var_part

    # 用混合权重r加权
    r = torch.clamp(r, min=eps)
    recon_loss = r * log_gauss
    recon_mu_part = r * mu_part
    recon_var_part = r * var_part
    
    # 返回总损失和分解部分
    total_recon = -torch.mean(torch.sum(recon_loss, dim=1))
    mu_component = -torch.mean(torch.sum(recon_mu_part, dim=1))
    var_component = -torch.mean(torch.sum(recon_var_part, dim=1))
    return total_recon, mu_component, var_component


def kl_categorical_loss(r, alpha, eps: float = 1e-6):
    r = torch.clamp(r, min=eps)
    alpha = torch.clamp(alpha, min=eps)
    log_r = torch.log(r)
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True) # [B, 1, H, W]
    digamma_diff = digamma(alpha) - digamma(alpha_sum)  # E[log w_{ik}]
    kl = torch.sum(r * (log_r - digamma_diff), dim=1)  # [B, H, W]
    return kl.mean()


def kl_dirichlet_loss(q_alpha, p_alpha, eps: float = 1e-6):
    # 保障参数为正
    q_alpha = torch.clamp(q_alpha, min=eps)
    p_alpha = torch.clamp(p_alpha, min=eps)
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


def kl_gaussian_loss(mu, var, eps: float = 1e-6):
    """
    KL(N(mu,var) || N(0,I))，假设 var 是对角协方差
    mu: [B, K, C, H, W]
    var: [B, K, C, H, W]
    """
    var = torch.clamp(var, min=eps)
    kl = 0.5 * (mu.pow(2) + var - torch.log(var) - 1)
    kl = kl.sum(dim=2)
    return kl.mean()



import math
def cosine_warmup(epoch: int, warmup_epochs: int) -> float:
    # 0 -> 1 的余弦热身
    e = max(0, min(epoch, warmup_epochs))
    if warmup_epochs <= 0:
        return 1.0
    return 0.5 * (1 - math.cos(math.pi * e / warmup_epochs))

def cosine_decay(epoch, max_epoch, start=1.0, end=0.0):
    """余弦衰减：前后平缓，中期下降快（对 max_epoch=0 做保护）"""
    # 当总轮数为1时（max_epoch=0），直接返回start，避免除零
    if max_epoch <= 0:
        return float(start)
    e = max(0, min(epoch, max_epoch))
    cos_inner = math.pi * e / max_epoch
    return end + (start - end) * (1 + math.cos(cos_inner)) / 2



class GmmLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.k = config.GMM_NUM
        self.config = config

    def forward(self, input, mu, var, r, alpha, prior, epoch, total_epochs):

        # r = compute_responsibilities(input, mu, var, alpha)  # [B,K,H,W]

        B, C, H, W = input.shape
        x = input.unsqueeze(1)  # [B, 1, C, H, W]

        # Reshape
        mu  = mu.reshape(B, self.k, C, H, W)
        var = var.reshape(B, self.k, C, H, W)
        r = r.reshape(B, self.k, H, W)
        alpha = alpha.reshape(B, self.k, H, W)
        prior = prior.reshape(B, self.k, H, W)

        # ----------------------
        # 1 Reconstruction term: -E_q(z|x) [ log N(x|mu,var) ]
        recon_loss, mu_component, var_component = reconstruction_loss(x, mu, var, r)

        # ----------------------
        # 2 KL(q(z|x) || p(z|Ω))
        kl_z_loss = kl_categorical_loss(r, alpha)
        
        # ----------------------
        # 3 KL(Dir(alpha) || Dir(prior)), exact closed form
        kl_o_loss = kl_dirichlet_loss(alpha, prior)
        
        weight_1 = self.config.LOSS1_WEIGHT
        weight_2 = self.config.LOSS2_WEIGHT
        weight_3 = cosine_decay(epoch + 1, total_epochs, start=self.config.LOSS3_WEIGHT, end=self.config.LOSS3_WEIGHT_END)

        loss_1 = recon_loss * weight_1
        loss_2 = kl_z_loss * weight_2
        loss_3 = kl_o_loss * weight_3

        regularization_term = kl_gaussian_loss(mu, var) * 0.001
        # regularization_term = (mu_component ** 2) * 10

        total_loss = loss_1 + loss_2 + loss_3 + regularization_term 


        # 字典返回形式
        return {
            'total': total_loss,
            'recon': loss_1,
            'recon_mu': mu_component,
            'recon_var': var_component,
            'kl_cat': loss_2, 
            'kl_dir': loss_3,  
            'weight_1': weight_1,
            'weight_2': weight_2,
            'weight_3': weight_3,
        }













