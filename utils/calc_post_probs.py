import torch
import torch.nn.functional as F


def calculate_posterior_probs(input, mu, var, pi, gmm_num=4, epsilon=1e-6):
    """
    计算每个像素点在每个高斯分量下的后验概率
    input: [B, C, H, W]        每个像素是C维特征向量
    mu:    [B, K, C, H, W]     第k个分量的均值向量
    var:   [B, K, C, H, W]     第k个分量的协方差对角线（只需4个数）
    pi:    [B, K, H, W]        第k个分量的混合系数
    输出: posterior_probs: [B, K, H, W]
    """
    B, C, H, W = input.shape
    K = gmm_num

    # [B, 1, C, H, W] - 为了和mu做broadcast
    input = input.unsqueeze(1)  # [B, 1, C, H, W]

    # 保证形状正确
    mu = mu.reshape(B, K, C, H, W)
    var = var.reshape(B, K, C, H, W)
    pi = pi.reshape(B, K, H, W)

    # ---------- log 高斯密度 ----------
    diff = input - mu  # [B, K, C, H, W]
    # 对角协方差时，直接除以对角线元素
    log_exponent = -0.5 * torch.sum((diff ** 2) / (var + epsilon), dim=2)               # [B, K, H, W]
    log_normalizer = -0.5 * torch.sum(torch.log(2 * torch.pi * var + epsilon), dim=2)   # [B, K, H, W]
    log_pdf = log_exponent + log_normalizer  # [B, K, H, W]

    # ---------- 加上 log π ----------
    log_pi = torch.log(pi + epsilon)  # [B, K, H, W]
    log_joint = log_pi + log_pdf      # [B, K, H, W]

    # ---------- 使用 log-sum-exp ----------
    log_sum = torch.logsumexp(log_joint, dim=1, keepdim=True)  # [B, 1, H, W]

    # ---------- 得到后验概率 ----------
    log_posterior = log_joint - log_sum             # [B, K, H, W]
    posterior_probs = torch.exp(log_posterior)      # [B, K, H, W]
    
    # 数值稳定性检查
    if torch.any(torch.isnan(posterior_probs)) or torch.any(torch.isinf(posterior_probs)):
        print("Warning: NaN or Inf detected in posterior_probs")
        posterior_probs = torch.nan_to_num(posterior_probs, nan=1.0/K, posinf=1.0, neginf=0.0)
    
    # 确保概率和为1（数值稳定性）
    prob_sum = torch.sum(posterior_probs, dim=1, keepdim=True)
    posterior_probs = posterior_probs / (prob_sum + epsilon)

    return posterior_probs


if __name__ == "__main__":
    # 测试代码
    B, C, H, W = 4, 4, 128, 128  # 批大小、通道数、高度、宽度
    K = 4  # 高斯分量数

    input = torch.randn(B, C, H, W)
    mu = torch.randn(B, K, C, H, W)
    var = torch.abs(torch.randn(B, K, C, H, W)) + 1e-8  # 确保方差为正
    pi = F.softmax(torch.randn(B, K, H, W), dim=1)  # 混合系数

    posterior_probs = calculate_posterior_probs(input, mu, var, pi)
    print(posterior_probs.shape)  # 应该是 [B, K, H, W]