import torch
import torch.nn as nn
from data.transform import apply_affine_transform
from config import Config


def standardize_features(features, config: Config = None, eps: float = 1e-6):
    """
    特征标准化：对每个样本每通道做空间均值/方差归一化
    
    Args:
        features: [B, C, H, W] 输入特征
        config: 配置对象（保留接口兼容性，未使用）
        eps: 数值稳定性参数
        
    Returns:
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
    """
    处理 GMM 参数：μ, σ², π, α (使用平滑映射替代硬截断)
    
    Args:
        output_x: [B, 2*K*C, H, W] 网络输出 (前半部分为 μ_raw, 后半部分为 log_var_raw)
        output_z: [B, K, H, W] π 的原始 logits
        output_o: [B, K, H, W] α (浓度参数) 的原始输出
        config: 配置对象
        eps: 数值稳定性参数
        
    Returns:
        mu: [B, K*C, H, W] 均值参数，范围 [-MU_RANGE, MU_RANGE]
        var: [B, K*C, H, W] 方差参数，范围 [VAR_MIN, VAR_MAX]  
        pi: [B, K, H, W] 概率参数，和为1
        alpha: [B, K, H, W] 浓度参数，范围 [0, 1]
    """
    K = config.GMM_NUM
    C = config.FEATURE_NUM
    mu_raw = output_x[:, :C * K, :, :]
    log_var_raw = output_x[:, C * K:, :, :]

    # ---- μ 平滑有界 ----
    mu_scale = max(1.0, config.MU_RANGE)  # 可配置放缩线性区
    mu = config.MU_RANGE * torch.tanh(mu_raw / mu_scale)

    # ---- σ² 温度缩放 + Sigmoid 映射 ----
    T_var = config.VAR_TEMP
    raw_scaled = log_var_raw / max(T_var, epsilon)
    sp_norm = torch.sigmoid(raw_scaled)  # 稳定的 [0,1] 映射
    var = config.VAR_MIN + (config.VAR_MAX - config.VAR_MIN) * sp_norm

    # ---- π 概率 ----
    T = max(1e-4, config.PI_TEMPERATURE)  # 温度参数，防止除0
    pi = torch.softmax(output_z / T, dim=1)
    pi = pi + epsilon
    pi = pi / pi.sum(dim=1, keepdim=True)

    # ---- α 浓度参数归一化到 (0,1) 范围 (供配准网络使用) ----
    alpha = torch.sigmoid(output_o)

    return mu, var, pi, alpha


def compute_dirichlet_priors(alpha, prior, reg_net, config: Config, epoch):
    """
    计算Dirichlet先验分布(配准)
    
    Args:
        alpha: [B,K,H,W] 当前浓度参数，应在 [0,1] 范围
        prior: [B,K,H,W] 先验概率，应在 [0,1] 范围
        reg_net: 配准网络
        config: 配置对象
        epoch: 当前训练轮数
    """
    # 验证输入数值范围
    assert alpha.min() >= 0 and alpha.max() <= 1, f"alpha 范围异常: [{alpha.min():.4f}, {alpha.max():.4f}]"
    assert prior.min() >= 0 and prior.max() <= 1, f"prior 范围异常: [{prior.min():.4f}, {prior.max():.4f}]"
    
    # 使用配准网络
    # 对整个 batch 一次性处理以提升效率
    reg_input = torch.cat((prior, alpha), dim=1)  # [B,2K,H,W]
    with torch.no_grad():  # 保持 reg_net 冻结
        affine_output = reg_net(reg_input)
        scale_pred = torch.clamp(affine_output[:, 0], config.SCALE_RANGE[0], config.SCALE_RANGE[1])
        # tx_pred = torch.clamp(affine_output[:, 1], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])
        tx_pred = torch.clamp(affine_output[:, 1], 0.0, 0.0)
        ty_pred = torch.clamp(affine_output[:, 2], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])

        # 仅在指定 epoch 之后应用配准形变
    reg_start = getattr(config, 'REG_START_EPOCH', 10)
    if epoch >= reg_start:
        affined_prior = apply_affine_transform(prior, scale_pred, tx_pred, ty_pred)
    else:
        affined_prior = prior

    return affined_prior


def forward_pass(image, label, prior, 
                 unet, x_net, z_net, o_net, reg_net, 
                 config: Config, 
                 epoch, 
                 epsilon):
    
    # 特征提取和标准化
    with torch.no_grad():
        image_4_features = unet(image)  # 提取特征
        image_4_features = standardize_features(features=image_4_features)  # 标准化特征

    # GMM网络前向传播
    output_x = x_net(image_4_features)  # mu, var
    output_z = z_net(image_4_features)  # pi
    output_o = o_net(image_4_features)  # d
    
    # 处理GMM参数
    mu, var, pi, alpha = process_gmm_parameters(output_x=output_x, 
                                                output_z=output_z, 
                                                output_o=output_o, 
                                                config=config, 
                                                epsilon=epsilon)
    # 计算Dirichlet先验分布(配准)
    affined_prior = compute_dirichlet_priors(alpha=alpha, 
                                             prior=prior, 
                                             reg_net=reg_net, 
                                             config=config,
                                             epoch=epoch)

    # 将浓度参数从 [0,1] 映射到配置指定范围 [base_conc, max_conc]
    base_conc = float(getattr(config, 'PRIOR_BASE_CONC', 2.0))
    max_conc = float(getattr(config, 'PRIOR_MAX_CONC', 8.0))
    
    alpha_mapped = base_conc + (max_conc - base_conc) * alpha
    prior_mapped = base_conc + (max_conc - base_conc) * affined_prior

    d1 = alpha_mapped       # 当前浓度参数
    d0 = prior_mapped       # 先验浓度参数

    return image_4_features, mu, var, pi, d1, d0