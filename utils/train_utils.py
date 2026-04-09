import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from data.transform import apply_affine_transform, apply_rotate_transform
import numpy as np
import cv2
import math
from torch.special import digamma


def standardize_features(features, config: Config = None, eps: float = 1e-6):
    """
    特征标准化：对每个样本每通道做空间均值/方差归一化
    标准化后的特征，范围软限制在 [-6, 6]
    """
    # Instance normalization: 对每个样本每通道做空间标准化
    mean = features.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    var = features.var(dim=(2, 3), keepdim=True, unbiased=False)  # [B, C, 1, 1]
    
    std = torch.sqrt(var.clamp(min=eps))
    x = (features - mean) / std
    
    # 软限制：避免极端值导致的梯度爆炸
    x = torch.clamp(x, -6.0, 6.0)

    # 处理异常值（NaN/Inf）
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.zeros_like(features)
    
    return x


def process_gmm_parameters(output_x, output_z, output_o, config: Config, epsilon=1e-6):
    K = config.GMM_NUM
    C = config.FEATURE_NUM
    mu_raw = output_x[:, :C * K, :, :]
    log_var_raw = output_x[:, C * K:, :, :]

    # ---- μ ----
    mu_scale = max(1.0, config.MU_RANGE)
    mu = config.MU_RANGE * torch.tanh(mu_raw / mu_scale)

    # ---- σ² ----
    log_var_min, log_var_max = config.LOG_VAR_MIN, config.LOG_VAR_MAX
    log_var = log_var_min + 0.5 * (log_var_max - log_var_min) * (torch.tanh(log_var_raw) + 1)
    var = torch.exp(log_var)

    # ---- π ----
    r = torch.softmax(output_z, dim=1)
    r = r + epsilon
    r = r / r.sum(dim=1, keepdim=True)

    # ---- α 浓度参数 ----
    alpha = F.softplus(output_o) + epsilon  # 保证 alpha > 0
    # alpha = epsilon + (config.PRIOR_INTENSITY - epsilon) * torch.sigmoid(output_o) 

    return mu, var, r, alpha


def compute_dirichlet_priors(alpha, prior, reg_net, config: Config, epoch):
    alpha = alpha / alpha.sum(dim=1, keepdim=True)
    alpha = torch.clamp(alpha, min=1e-6, max=1.0)

    # 验证输入数值范围
    assert alpha.min() >= 0 and alpha.max() <= 1, f"alpha 范围异常: [{alpha.min():.4f}, {alpha.max():.4f}]"
    assert prior.min() >= 0 and prior.max() <= 1, f"prior 范围异常: [{prior.min():.4f}, {prior.max():.4f}]"

    # 使用配准网络
    reg_net.eval()  # 设置为评估模式
    reg_input = torch.cat((prior[:, 1:2, :, :], alpha[:, 1:2, :, :]), dim=1) # [B,2,H,W]
    with torch.no_grad():  # 保持 reg_net 冻结
        affine_output = reg_net(reg_input)
        scale_pred = torch.clamp(affine_output[:, 0], config.PRIOR_SCALE_RANGE[0], config.PRIOR_SCALE_RANGE[1])
        tx_pred = torch.clamp(affine_output[:, 1], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])
        ty_pred = torch.clamp(affine_output[:, 2], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])

    if epoch >= config.START_REG:
        affined_prior = apply_affine_transform(prior, scale_pred, tx_pred, ty_pred)
    else:
        affined_prior = prior
    return affined_prior, scale_pred, tx_pred, ty_pred


def forward_pass(image, label, 
                 prior, 
                 unet, x_net, z_net, o_net, reg_net, align_net, scale_net,
                 ds,
                 config: Config, 
                 epoch=None, 
                 epsilon=1e-6):
    
    angle = None
    if ds == "SCD" or ds == "YORK":
        # 旋转对齐
        with torch.no_grad():
            angle = align_net(image)  # 预测旋转角度，单位：弧度
            # Config.ROTATE_RANGE 是以“度”为单位，需转换为弧度后再进行裁剪
            low_rad = math.radians(config.ROTATE_RANGE[0])
            high_rad = math.radians(config.ROTATE_RANGE[1])
            angle = torch.clamp(angle, low_rad, high_rad)
            image = apply_rotate_transform(image, -angle[:, 0])  # 逆向旋转对齐
            label = apply_rotate_transform(label.float(), -angle[:, 0], mode='nearest').long()


    # 特征提取和标准化
    with torch.no_grad():
        feature_4chs = unet(image)  # 提取特征
        # output_o = feature_4chs.clone()
    
    feature_4chs = standardize_features(features=feature_4chs)  # 标准化特征

    # GMM网络前向传播
    output_x = x_net(feature_4chs)  # mu, var
    output_z = z_net(feature_4chs)  # r
    output_o = o_net(feature_4chs)  # d
    # with torch.no_grad():

    # 处理GMM参数
    mu, var, r, alpha = process_gmm_parameters(output_x=output_x, 
                                               output_z=output_z, 
                                               output_o=output_o, 
                                               config=config, 
                                               epsilon=epsilon)
    
    d1 = alpha
    d0, prior_scale, prior_tx, prior_ty = None, None, None, None
    # 计算Dirichlet先验分布(配准)
    if reg_net != None:
        affined_prior, prior_scale, prior_tx, prior_ty = compute_dirichlet_priors(alpha=alpha, 
                                                                                  prior=prior, 
                                                                                  reg_net=reg_net, 
                                                                                  config=config,
                                                                                  epoch=epoch)
        prior_mapped = config.PRIOR_INTENSITY * affined_prior   # 先验浓度参数
        d0 = prior_mapped       # 先验浓度参数
    # d0 = prior * config.PRIOR_INTENSITY  # 如果不配准



    return {
        "feature_4chs": feature_4chs,
        "image": image,
        "label": label,
        "mu": mu,
        "var": var,
        "r": r,
        "d1": d1,
        "d0": d0,
        "prior_scale": prior_scale,
        "prior_tx": prior_tx,
        "prior_ty": prior_ty,
        "angle": angle,
    }


def gmm_posterior(x, mu, var, pi, eps: float = 1e-6):
    B, C, H, W = x.shape
    x = x.unsqueeze(1)  # [B, 1, C, H, W]

    # Reshape
    mu  = mu.reshape(B, 4, C, H, W)
    var = var.reshape(B, 4, C, H, W)


    var = var.clamp_min(eps)

    mu_part = -0.5 * torch.sum((x - mu) ** 2 / var, dim=2)  # [B, K, H, W]
    var_part = -0.5 * torch.sum(torch.log(2 * math.pi * var), dim=2)  # [B, K, H, W]
    
    # 完整的高斯对数似然
    log_gauss = mu_part + var_part

    # 用混合权重 pi 加权（先归一化到概率单纯形上）
    pi = torch.clamp(pi, min=eps)
    pi = pi / pi.sum(dim=1, keepdim=True)
    log_pi = torch.log(pi)
    logits = log_pi + log_gauss  # [B, K, H, W]

    # posterior = softmax(logits)
    log_norm = torch.logsumexp(logits, dim=1, keepdim=True)  # [B,1,H,W]
    posterior = torch.exp(logits - log_norm)
    posterior = posterior.clamp_min(eps)
    posterior = posterior / posterior.sum(dim=1, keepdim=True)

    return posterior