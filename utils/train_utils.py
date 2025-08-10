import torch
import torch.nn as nn
from data.transform import apply_affine_transform
from config import Config


def standardize_features(features, config: Config, eps: float = 1e-8):
    """多策略特征标准化。
    模式:
      instance: (默认) 对每个样本每通道做空间均值/方差归一化，然后可学习范围内截断
      channel:  对 batch 维聚合 (B,H,W) 求每通道统计量
      none:     不做归一化，直接返回
    归一化后不再用硬 tanh 压缩，保留分布动态；仅对极端数值进行轻度截断。
    """
    mode = getattr(config, 'FEATURE_NORM', 'instance') or 'instance'
    if mode == 'none':
        return features

    if mode == 'instance':
        mean = features.mean(dim=(2, 3), keepdim=True)
        var = features.var(dim=(2, 3), keepdim=True, unbiased=False)
    elif mode == 'channel':
        mean = features.mean(dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
        var = features.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    else:
        mean = features.mean(dim=(2, 3), keepdim=True)
        var = features.var(dim=(2, 3), keepdim=True, unbiased=False)

    std = torch.sqrt(var.clamp(min=eps))
    x = (features - mean) / std
    # 软限制：仅在极端时截断，避免梯度爆炸（不使用 tanh 以保留线性信息）
    x = torch.clamp(x, -6.0, 6.0)

    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.zeros_like(features)
    return x


def process_gmm_parameters(output_x, output_z, output_o, config: Config, epsilon):
        """处理 GMM 参数：μ, σ², π, d
        新策略：
            - μ 直接 clamp 到 [-MU_RANGE, MU_RANGE]
            - 方差通过 log_var 参数化：log_var_raw -> clamp -> var = exp(log_var)
            - π 使用 softmax( logits / T )，更标准的概率输出
            - d 保持当前浓度设定
        """
        K = config.GMM_NUM
        C = config.FEATURE_NUM
        mu_raw = output_x[:, :C * K, :, :]
        log_var_raw = output_x[:, C * K:, :, :]

        # μ clamp
        mu = torch.clamp(mu_raw, -config.MU_RANGE, config.MU_RANGE)

        # log σ² -> clamp in log space -> σ²
        # 允许 var ∈ [VAR_MIN, VAR_MAX]
        log_var_min = torch.log(torch.tensor(config.VAR_MIN, device=mu.device))
        log_var_max = torch.log(torch.tensor(config.VAR_MAX, device=mu.device))
        log_var = torch.clamp(log_var_raw, log_var_min, log_var_max)
        var = torch.exp(log_var)

        # π softmax with temperature
        T = max(1e-3, float(getattr(config, 'PI_TEMPERATURE', 1.0)))
        pi = torch.softmax(output_z / T, dim=1)
        pi = torch.clamp(pi, min=1e-6)
        pi = pi / pi.sum(dim=1, keepdim=True)

        # d: Dirichlet concentration parameters
        d_raw = nn.Softplus()(output_o) + 0.5
        d = torch.clamp(d_raw, 0.5, 10.0)
        return mu, var, pi, d


def compute_dirichlet_priors(d, slice_info, num_of_slice_info, dirichlet_priors, reg_net, config: Config, epoch):
    """
    计算Dirichlet先验分布 (改进版本)
    Args:
        d: d参数 [B, K, H, W]
        slice_info: slice信息
        num_of_slice_info: slice数量信息
        dirichlet_priors: 预加载的先验分布列表
        reg_net: 配准网络
        config: 配置对象
        epoch: 当前epoch
    Returns:
        dirichlet: 计算得到的Dirichlet分布 [B, K, H, W]
    """
    batch_size = d.shape[0]
    dirichlet = torch.zeros(batch_size, config.GMM_NUM, config.IMG_SIZE, config.IMG_SIZE, 
                          dtype=torch.float32, device=config.DEVICE)
    
    for i in range(batch_size):
        try:
            slice_id = slice_info[i].item()
            slice_num = num_of_slice_info[i].item()
            
            # 改进的slice_id映射，避免索引越界
            if slice_num > 0:
                normalized_slice_id = min(int(slice_id / slice_num * len(dirichlet_priors)), len(dirichlet_priors) - 1)
            else:
                normalized_slice_id = 0
            
            normalized_slice_id = max(0, normalized_slice_id)  # 确保非负
            
            prior_prob = dirichlet_priors[normalized_slice_id].unsqueeze(0)  # [1,K,H,W] 原始概率（或近似）
            # 归一化到概率 (数值稳定)
            prior_prob = prior_prob.clamp_min(1e-8)
            prior_prob = prior_prob / prior_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # 使用概率参与配准：构造配准输入 (概率 + 当前模型的类别期望)
            # 当前 d 是浓度参数, 先得到其期望概率 d_expect
            d_expect = d[i].unsqueeze(0)
            d_expect = d_expect / d_expect.sum(dim=1, keepdim=True).clamp_min(1e-8)
            reg_input = torch.cat((prior_prob, d_expect), dim=1)  # [1,2K,H,W]

            with torch.no_grad():  # 保持 reg_net 冻结
                affine_output = reg_net(reg_input)
                scale_pred = torch.clamp(affine_output[0, 0], config.SCALE_RANGE[0], config.SCALE_RANGE[1])
                tx_pred = torch.clamp(affine_output[0, 1], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])
                ty_pred = torch.clamp(affine_output[0, 2], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])

            # 仅在指定 epoch 之后应用配准形变
            reg_start = getattr(config, 'REG_START_EPOCH', 25)
            if epoch >= reg_start:
                warped_prob = apply_affine_transform(prior_prob, scale_pred, tx_pred, ty_pred)
            else:
                warped_prob = prior_prob

            # 再次归一化防插值偏差
            warped_prob = warped_prob.clamp_min(1e-8)
            warped_prob = warped_prob / warped_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # 映射到浓度范围: base + p * (max - base)
            base_c = getattr(config, 'PRIOR_BASE_CONC', 2.0)
            max_c  = getattr(config, 'PRIOR_MAX_CONC', 8.0)
            conc = base_c + warped_prob * (max_c - base_c)
            conc = torch.clamp(conc, 0.5, 10.0)
            dirichlet[i] = conc.squeeze(0)
            
        except (IndexError, ValueError) as e:
            print(f"Error occurred while processing slice {i}: {e}")
            # 如果出现错误，使用默认的适中先验
            dirichlet[i] = torch.full((config.GMM_NUM, config.IMG_SIZE, config.IMG_SIZE), 
                                    2.0, dtype=torch.float32, device=config.DEVICE)
    
    return dirichlet


def forward_pass(image, unet, x_net, z_net, o_net, reg_net, dirichlet_priors, 
                slice_info, num_of_slice_info, config: Config, epoch, epsilon):
    """
    完整的前向传播过程
    Args:
        image: 输入图像
        各种网络和参数
    Returns:
        所有需要的输出参数
    """
    # 特征提取和标准化
    with torch.no_grad():
        image_4_features = unet(image)  # 提取特征
        image_4_features = standardize_features(features=image_4_features, config=config)  # 标准化特征

    # GMM网络前向传播
    output_x = x_net(image_4_features)  # mu, var
    output_z = z_net(image_4_features) if z_net is not None else None  # pi (可能为 None)
    output_o = o_net(image_4_features)  # d
    
    # 处理GMM参数
    if z_net is not None:
        mu, var, pi, d = process_gmm_parameters(output_x=output_x, 
                                                output_z=output_z, 
                                                output_o=output_o, 
                                                config=config, 
                                                epsilon=epsilon)
    else:
        # 仍需 μ, var, d；π 置为 None，后续 DirichletGmmLoss 内部由 d 归一得到
        K = config.GMM_NUM
        C = config.FEATURE_NUM
        mu_raw = output_x[:, :C * K, :, :]
        log_var_raw = output_x[:, C * K:, :, :]
        mu = torch.clamp(mu_raw, -config.MU_RANGE, config.MU_RANGE)
        log_var_min = torch.log(torch.tensor(config.VAR_MIN, device=mu.device))
        log_var_max = torch.log(torch.tensor(config.VAR_MAX, device=mu.device))
        log_var = torch.clamp(log_var_raw, log_var_min, log_var_max)
        var = torch.exp(log_var)
        d_raw = nn.Softplus()(output_o) + 0.5
        d = torch.clamp(d_raw, 0.5, 10.0)
        pi = None

    # 计算Dirichlet先验
    dirichlet = compute_dirichlet_priors(d=d, 
                                         slice_info=slice_info, 
                                         num_of_slice_info=num_of_slice_info, 
                                         dirichlet_priors=dirichlet_priors, 
                                         reg_net=reg_net, 
                                         config=config, 
                                         epoch=epoch)

    d1 = d              # 当前d
    d0 = dirichlet      # 先验d0
    
    return image_4_features, mu, var, pi, d1, d0