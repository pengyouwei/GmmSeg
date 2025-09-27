import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from data.transform import apply_affine_transform, apply_rotate_transform
from utils.postprocess import comprehensive_postprocess
import numpy as np
import cv2

import math
from torch.special import digamma

def compute_responsibilities(x, mu, var, alpha, eps=1e-6):
    B, C, H, W = x.shape
    K = mu.shape[1] // C  # GMM分量数
    x = x.unsqueeze(1)  # [B, 1, C, H, W]

    # Reshape
    mu  = mu.reshape(B, K, C, H, W)
    var = var.reshape(B, K, C, H, W)
    alpha  = alpha.reshape(B, K, H, W)

    # x: [B,1,C,H,W], mu/var: [B,K,C,H,W], alpha: [B,K,H,W]
    var = var.clamp_min(eps)

    # log N(x|mu,var)，对通道求和 -> [B,K,H,W]
    log_gauss = -0.5 * torch.sum((x - mu) ** 2 / var, dim=2) \
                -0.5 * torch.sum(torch.log(2.0 * math.pi * var), dim=2)

    # 先验的 E[log w_k] -> [B,K,H,W]
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True).clamp_min(eps)
    log_prior = digamma(alpha) - digamma(alpha_sum)

    # 后验 log r_k ∝ log_prior + log_likelihood
    log_r = log_prior + log_gauss
    r = torch.softmax(log_r, dim=1)  # [B,K,H,W]
    return r  # 概率分割，argmax 即分割标签


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
    pi = torch.softmax(output_z, dim=1)
    pi = pi + epsilon
    pi = pi / pi.sum(dim=1, keepdim=True)

    # ---- α 浓度参数 ----
    alpha = F.softplus(output_o) + epsilon  # 保证 alpha > 0

    return mu, var, pi, alpha


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



def zoom_around_center(
    tensor: torch.Tensor,         # [B,C,H,W]
    scale: torch.Tensor,          # [B]
    out_size: int,
    is_label: bool = False,
    padding_mode: str = 'reflection'
):
    """
    基于图像几何中心的“中心裁剪×scale，再resize回 out_size”。
    不依赖标签，适用于左心室已在图像中心的情况。
    """
    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    ys = torch.linspace(-1.0, 1.0, out_size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, out_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,out,out,2]

    # 中心即 (0,0) 的归一化坐标，不需要偏移
    s = scale.view(B, 1, 1, 1).clamp_min(1e-6)
    alpha_x = (float(out_size) / max(W, 1)) * s
    alpha_y = (float(out_size) / max(H, 1)) * s
    alpha_xy = torch.stack([alpha_x.squeeze(-1), alpha_y.squeeze(-1)], dim=-1)

    src_grid = base_grid * alpha_xy
    mode = 'bilinear'
    if is_label: mode = 'nearest'
    out = F.grid_sample(tensor, src_grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    if is_label:
        out = out.long()
    return out



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
        feature_4chs = standardize_features(features=feature_4chs)  # 标准化特征

    scaled_image, scaled_label, image_scale = None, None, None
    if config.DO_IMAGE_SCALE:
        feature_lv_myo = feature_4chs[:, 1:3, ...]
        scale_output = scale_net(feature_lv_myo)
        image_scale = torch.clamp(scale_output[:, 0], config.IMAGE_SCALE_RANGE[0], config.IMAGE_SCALE_RANGE[1])

        # 按“裁剪尺寸 = IMG_SIZE * scale → resize 回 IMG_SIZE”的语义进行缩放（围绕 LV 质心）
        scaled_image = zoom_around_center(
            tensor=image, scale=image_scale, out_size=config.IMG_SIZE,
            is_label=False, padding_mode='reflection'
        )
        scaled_label = zoom_around_center(
            tensor=label.float(), scale=image_scale, out_size=config.IMG_SIZE,
            is_label=True, padding_mode='zeros'
        )

        if ds in {"MM", "SCD"}:
            feature_4chs = unet(scaled_image)  # 提取特征（保留计算图，允许梯度回传到 scale）
            feature_4chs = standardize_features(features=feature_4chs)  # 标准化特征


    # GMM网络前向传播
    output_x = x_net(feature_4chs)  # mu, var
    output_z = z_net(feature_4chs)  # pi
    output_o = o_net(feature_4chs)  # d

    # 处理GMM参数
    mu, var, pi, alpha = process_gmm_parameters(output_x=output_x, 
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



    return {
        "feature_4chs": feature_4chs,
        "scaled_image": scaled_image,
        "scaled_label": scaled_label,
        "image": image,
        "label": label,
        "mu": mu,
        "var": var,
        "pi": pi,
        "d1": d1,
        "d0": d0,
        "image_scale": image_scale,
        "prior_scale": prior_scale,
        "prior_tx": prior_tx,
        "prior_ty": prior_ty,
        "angle": angle,
    }