import torch
import torch.nn as nn
from utils.affine_transform import apply_affine_transform
from config import Config


def standardize_features(features, eps=1e-8):
    """
    将特征图标准化到均值0方差1
    Args:
        features: 输入特征图 [B, C, H, W]
        eps: 防止除零的小数
    Returns:
        标准化后的特征图
    """
    # 计算每个样本每个通道的均值和标准差
    batch_size, channels, height, width = features.shape
    
    # 在空间维度上计算统计量 [B, C, 1, 1]
    mean = features.mean(dim=(2, 3), keepdim=True)
    var = features.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    
    # 标准化
    standardized = (features - mean) / std
    
    return standardized


def process_gmm_parameters(output_x, output_z, output_o, config: Config, epsilon):
    """
    处理GMM网络输出，获得mu, var, pi, d参数
    Args:
        output_x: x_net输出 [mu, var]
        output_z: z_net输出 [pi]
        output_o: o_net输出 [d]
        config: 配置对象
        epsilon: 数值稳定性参数
    Returns:
        mu, var, pi, d: 处理后的GMM参数
    """
    # 分离mu和var
    mu, var = output_x[:, :config.FEATURE_NUM*config.GMM_NUM, :, :], output_x[:, config.FEATURE_NUM*config.GMM_NUM:, :, :]
    mu = torch.clamp(mu, min=-1.2, max=1.2)  # 限制 mu 的范围
    var = torch.clamp(nn.Softplus()(var), min=epsilon, max=1.0)  # 限制方差范围

    # 处理pi
    pi = nn.Softplus()(output_z) + epsilon
    pi = pi / torch.sum(pi, dim=1, keepdim=True)  # 归一化

    # 处理d
    d = nn.Softplus()(output_o) + epsilon  # 使用 Softplus 保证 d > 0
    d = d / torch.sum(d, dim=1, keepdim=True)  # 归一化
    
    return mu, var, pi, d


def compute_dirichlet_priors(d, slice_info, num_of_slice_info, dirichlet_priors, reg_net, config: Config, epoch):
    """
    计算Dirichlet先验分布
    Args:
        d: d参数 [B, K, H, W]
        slice_info: slice信息
        num_of_slice_info: slice数量信息
        dirichlet_priors: 预加载的先验分布列表
        reg_net: 回归网络
        config: 配置对象
        epoch: 当前epoch
    Returns:
        dirichlet: 计算得到的Dirichlet分布 [B, K, H, W]
    """
    batch_size = d.shape[0]
    dirichlet = torch.zeros(batch_size, config.GMM_NUM, config.IMG_SIZE, config.IMG_SIZE, 
                          dtype=torch.float32, device=config.DEVICE)
    
    for i in range(batch_size):
        slice_id = slice_info[i].item()
        slice_num = num_of_slice_info[i].item()
        slice_id = min(int(slice_id / slice_num * 10), 9)  # 将slice_id归一化到 0 到 9，防止索引越界

        dirichlet_prior = dirichlet_priors[slice_id].unsqueeze(0)  # [1, 4, H, W]
        affine_input = torch.cat((dirichlet_prior, d[i].unsqueeze(0)), dim=1)  # [1, 8, H, W]

        affine_output = reg_net(affine_input)
        scale_pred = affine_output[0, 0]
        tx_pred = affine_output[0, 1]
        ty_pred = affine_output[0, 2]

        # 在第50个epoch后开始应用仿射变换
        if epoch >= 50:
            dirichlet_prior = apply_affine_transform(dirichlet_prior, scale_pred, tx_pred, ty_pred)  # [1, 4, H, W]
        dirichlet[i] = dirichlet_prior.squeeze(0)
    
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
        image_4_features = standardize_features(image_4_features)  # 标准化特征
    
    # GMM网络前向传播
    output_x = x_net(image_4_features)  # mu, var
    output_z = z_net(image_4_features)  # pi
    output_o = o_net(image_4_features)  # d
    
    # 处理GMM参数
    mu, var, pi, d = process_gmm_parameters(output_x, output_z, output_o, config, epsilon)
    
    # 计算Dirichlet先验
    dirichlet = compute_dirichlet_priors(d, slice_info, num_of_slice_info, 
                                       dirichlet_priors, reg_net, config, epoch)
    
    d1 = d * 1.0
    d0 = dirichlet * 1.0
    
    return image_4_features, mu, var, pi, d1, d0