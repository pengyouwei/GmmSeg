import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma, gammaln  # pytorch1.8+支持
import math


def reconstruction_loss(x, mu, var, pi):
    """
    计算重建损失
    """
    log_exp = -0.5 * torch.sum((x - mu) ** 2 / var, dim=2)  # [B, K, H, W]
    log_norm = -0.5 * torch.sum(torch.log(2 * math.pi * var), dim=2)  # [B, K, H, W]
    log_gauss = log_exp + log_norm
    recon_loss = pi * log_gauss
    return -torch.mean(torch.sum(recon_loss, dim=1))


def kl_categorical_loss(post_prob, alpha):
    """
    计算类别分布的KL散度损失

    post_prob: Tensor, shape [B, K, H, W] - 后验类别概率 q(z|x)
    alpha: Tensor, shape [B, K, H, W] - Dirichlet参数，计算先验类别概率 p(z|Ω)
    
    返回标量loss
    """
    
    log_post_prob = torch.log(post_prob)
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True) # [B, 1, H, W]
    digamma_diff = digamma(alpha) - digamma(alpha_sum)
    kl = torch.sum(post_prob * (log_post_prob - digamma_diff), dim=1)  # [B, H, W]

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
    def __init__(self, GMM_NUM: int = 4):
        super().__init__()
        self.k = GMM_NUM

    def forward(self, input, mu, var, pi, d, d0, epoch=None, total_epochs=None):

        B, C, H, W = input.shape
        x = input.unsqueeze(1)  # [B, 1, C, H, W]

        # Reshape
        mu  = mu.reshape(B, self.k, C, H, W)
        var = var.reshape(B, self.k, C, H, W)
        pi  = pi.reshape(B, self.k, H, W)
        d   = d.reshape(B, self.k, H, W)
        d0  = d0.reshape(B, self.k, H, W)

        # ----------------------
        # 1) Reconstruction term: -E_q(z|x) [ log N(x|mu,var) ]
        recon_loss = reconstruction_loss(x, mu, var, pi)

        # ----------------------
        # 2) KL(q(z|x) || p(z|Ω))
        kl_z_loss = kl_categorical_loss(pi, d)
        
        # ----------------------
        # 3) KL(Dir(d) || Dir(d0)), exact closed form
        kl_o_loss = kl_dirichlet_loss(d, d0)

        # dynamic weights
        weight_1 = 1.0
        weight_2 = 1.0
        if epoch is not None and total_epochs is not None:
            progress = max(0.0, min(1.0, epoch / max(1, total_epochs)))
            if progress < 0.1:
                weight_3 = 1.0
            elif progress < 0.3:
                weight_3 = 1.0 - 0.7 * ((progress - 0.1) / 0.2)
            elif progress < 0.6:
                weight_3 = 0.3 - 0.2 * ((progress - 0.3) / 0.3)
            elif progress < 0.8:
                weight_3 = 0.1 - 0.09 * ((progress - 0.6) / 0.2)
            else:
                weight_3 = 0.01
        else:
            weight_3 = 0.5

        loss_1 = recon_loss * weight_1
        loss_2 = kl_z_loss * weight_2
        loss_3 = kl_o_loss * weight_3

        loss_mse = nn.MSELoss()(d, d0) * 0.001

        total_loss = loss_1 + loss_2 + loss_3 + loss_mse

        if not torch.isfinite(total_loss):
            print("Warning: total_loss has NaN/Inf.")

    # 字典返回形式
        return {
            'total': total_loss,
            'recon': loss_1,
            'kl_pi': loss_2,          # 这里把原 kl_z_loss 视为 gating vs prior
            'kl_dir': loss_3,         # 已乘以动态权重
            'loss_mse': loss_mse,
            'w_dir': weight_3,
            'post_prob': pi,
        }


def posterior_from_params(x, mu, var, pi, eps: float = 1e-8):
    """根据 (x, mu, var, pi) 计算 posterior 责任 r。
    输入形状:
        x:  [B,1,C,H,W]
        mu: [B,K,C,H,W]
        var:[B,K,C,H,W]
        pi: [B,K,H,W]
    返回:
        r:  [B,K,H,W]
    """
    # log N
    log_exp = -0.5 * torch.sum((x - mu) ** 2 / var.clamp_min(eps), dim=2)  # [B,K,H,W]
    log_norm = -0.5 * torch.sum(torch.log(2 * math.pi * var.clamp_min(eps)), dim=2)
    log_gauss = log_exp + log_norm
    log_weighted = torch.log(pi.clamp_min(eps)) + log_gauss
    m = torch.max(log_weighted, dim=1, keepdim=True).values
    exp_shift = torch.exp(log_weighted - m)
    r = exp_shift / (exp_shift.sum(dim=1, keepdim=True) + eps)
    return r


class DirichletGmmLoss(nn.Module):
    """Simplified loss: remove explicit pi network; derive pi = d / sum(d).
    Supports warmup using expectation form then switches to true mixture NLL.
    """
    def __init__(self, GMM_NUM: int = 4, nll_warmup_epochs: int = 0):
        super().__init__()
        self.k = GMM_NUM
        self.nll_warmup_epochs = nll_warmup_epochs

    @staticmethod
    def _log_gaussians(x, mu, var):
        # x: [B,1,C,H,W]; mu,var: [B,K,C,H,W]
        log_exp = -0.5 * torch.sum((x - mu) ** 2 / var.clamp_min(1e-8), dim=2)  # [B,K,H,W]
        log_norm = -0.5 * torch.sum(torch.log(2 * math.pi * var.clamp_min(1e-8)), dim=2)  # [B,K,H,W]
        return log_exp + log_norm

    def forward(self, input, mu, var, d, d0, epoch=None, total_epochs=None):
        B, C, H, W = input.shape
        x = input.unsqueeze(1)
        mu  = mu.reshape(B, self.k, C, H, W)
        var = var.reshape(B, self.k, C, H, W)
        d   = d.reshape(B, self.k, H, W).clamp_min(1e-6)
        d0  = d0.reshape(B, self.k, H, W).clamp_min(1e-6)

        alpha_sum = d.sum(dim=1, keepdim=True)
        pi = d / (alpha_sum + 1e-8)

        log_gauss = self._log_gaussians(x, mu, var)
        if epoch is not None and epoch < self.nll_warmup_epochs:
            recon = -(pi * log_gauss).sum(dim=1).mean()
            used_nll = False
        else:
            log_mix = torch.log(pi.clamp_min(1e-8)) + log_gauss
            m = torch.max(log_mix, dim=1, keepdim=True).values
            log_sum = m + torch.log(torch.sum(torch.exp(log_mix - m), dim=1, keepdim=True))
            recon = -log_sum.mean()
            used_nll = True

        # posterior responsibilities
        r = posterior_from_params(x=x, mu=mu, var=var, pi=pi)

        kl_dir = kl_dirichlet_loss(d, d0)
        entropy = -(pi * (pi.clamp_min(1e-8).log())).sum(dim=1).mean()
        entropy_reg = 0.001 * entropy
        total = recon + kl_dir + entropy_reg

        return {
            'total': total,
            'recon': recon,
            'kl_dir': kl_dir,
            'entropy_reg': entropy_reg,
            'w_dir': 1.0,
            'post_prob': r,
            'pi': pi,
            'used_nll': used_nll,
        }



