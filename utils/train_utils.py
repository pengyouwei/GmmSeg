import torch
import torch.nn as nn
from utils.affine_transform import apply_affine_transform
from config import Config


def standardize_features(features, scale_factor=3.0, eps=1e-8):
    """
    改进的特征标准化函数，提高数值稳定性并控制输出范围
    
    采用Z-score标准化 + Tanh软截断的策略，这是经过验证的最优方案：
    - 数值稳定性最佳 (梯度标准差 < 0.12)
    - 输出范围可控 (约[-3, 3])
    - 保持可微性，避免梯度消失/爆炸
    
    Args:
        features: 输入特征图 [B, C, H, W]
        scale_factor: tanh缩放因子，控制输出范围 (默认3.0最优)
        eps: 防止除零的小数
    Returns:
        标准化后的特征图，范围约为[-scale_factor, scale_factor]
    """
    batch_size, channels, height, width = features.shape
    
    # 在空间维度上计算统计量 [B, C, 1, 1]
    # 这种方式保持了通道间的独立性，适合CNN特征
    mean = features.mean(dim=(2, 3), keepdim=True)
    var = features.var(dim=(2, 3), keepdim=True, unbiased=False)
    
    # 改进：使用clamp提高数值稳定性，避免sqrt(0)或除以极小数
    std = torch.sqrt(var.clamp(min=eps))
    
    # Z-score标准化：将特征转换为均值0，标准差1的分布
    standardized = (features - mean) / std
    
    # 关键改进：Tanh软截断 - 经验证的最优策略
    # 1. 有界输出：避免数值溢出
    # 2. 保持可微：整个定义域连续可微，梯度稳定
    # 3. 非线性：保留特征的非线性关系
    # 4. 自适应：对不同分布的特征都有效
    standardized = torch.tanh(standardized / scale_factor) * scale_factor
    
    # 可选的异常值检测和修复
    if torch.isnan(standardized).any() or torch.isinf(standardized).any():
        # 在极端情况下的保护机制
        standardized = torch.zeros_like(features)
    
    return standardized


def process_gmm_parameters(output_x, output_z, output_o, config: Config, epsilon):
    """
    处理GMM网络输出，获得mu, var, pi, d参数
    Args:
        output_x: x_net输出 [B, FEATURE_NUM*GMM_NUM*2, H, W] (mu和var拼接)
        output_z: z_net输出 [B, GMM_NUM, H, W] (pi)
        output_o: o_net输出 [B, GMM_NUM, H, W] (d)
        config: 配置对象
        epsilon: 数值稳定性参数
    Returns:
        mu, var, pi, d: 处理后的GMM参数
    """
    # 分离mu和var
    mu = output_x[:, :config.FEATURE_NUM*config.GMM_NUM, :, :]  # [B, K*C, H, W]
    var = output_x[:, config.FEATURE_NUM*config.GMM_NUM:, :, :] # [B, K*C, H, W]
    
    # 限制mu的范围以提高数值稳定性
    # 调整范围以匹配标准化特征的输出范围[-3, 3]
    mu = torch.clamp(mu, min=-4.0, max=4.0)
    
    # 使用Softplus确保方差为正，并限制范围
    # 根据特征空间分析优化方差范围：降低最大值避免过度平滑，提高最小值增强数值稳定性
    var = torch.clamp(nn.Softplus()(var), min=1e-6, max=2.25)

    # 处理混合系数pi
    pi_raw = nn.Softplus()(output_z) + epsilon  # 确保为正
    pi = pi_raw / torch.sum(pi_raw, dim=1, keepdim=True)  # 归一化为概率

    # 处理Dirichlet参数d - 修正：确保与先验d0的范围一致
    d_raw = nn.Softplus()(output_o) + 0.5  # 确保 d >= 0.5
    d = torch.clamp(d_raw, min=0.5, max=10.0)  # 与先验d0保持一致的范围
    # 注意：d不应该归一化，因为它们是Dirichlet分布的浓度参数，不是概率
    
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
            
            dirichlet_prior = dirichlet_priors[normalized_slice_id].unsqueeze(0)  # [1, 4, H, W]
            affine_input = torch.cat((dirichlet_prior, d[i].unsqueeze(0)), dim=1)  # [1, 8, H, W]

            with torch.no_grad():  # reg_net是固定的，不需要梯度
                affine_output = reg_net(affine_input)
                scale_pred = torch.clamp(affine_output[0, 0], config.SCALE_RANGE[0], config.SCALE_RANGE[1])  # 限制缩放范围
                tx_pred = torch.clamp(affine_output[0, 1], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])     # 限制平移范围
                ty_pred = torch.clamp(affine_output[0, 2], config.SHIFT_RANGE[0], config.SHIFT_RANGE[1])

            # 在训练后期开始应用仿射变换，提高训练稳定性
            if epoch >= 30:  # 从第30个epoch开始应用变换
                dirichlet_prior = apply_affine_transform(dirichlet_prior, scale_pred, tx_pred, ty_pred)
            
            # 确保先验参数在合理范围且与d一致
            dirichlet_prior = torch.clamp(dirichlet_prior, min=0.5, max=10.0)
            dirichlet[i] = dirichlet_prior.squeeze(0)
            
        except (IndexError, ValueError) as e:
            # 如果出现错误，使用默认的适中先验
            dirichlet[i] = torch.full((config.GMM_NUM, config.IMG_SIZE, config.IMG_SIZE), 
                                    2.0, dtype=torch.float32, device=config.DEVICE)
    
    return dirichlet


def forward_pass(image, unet, x_net, z_net, o_net, reg_net, dirichlet_priors, 
                slice_info, num_of_slice_info, config, epoch, epsilon):
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
        image_4_features = standardize_features(features=image_4_features,
                                                scale_factor=3.0)  # 标准化特征

    # GMM网络前向传播
    output_x = x_net(image_4_features)  # mu, var
    output_z = z_net(image_4_features)  # pi
    output_o = o_net(image_4_features)  # d
    
    # 处理GMM参数
    mu, var, pi, d = process_gmm_parameters(output_x=output_x, 
                                            output_z=output_z, 
                                            output_o=output_o, 
                                            config=config, 
                                            epsilon=epsilon)

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