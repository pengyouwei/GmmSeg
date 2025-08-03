import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GmmSegLoss(nn.Module):
    def __init__(self, GMM_NUM=4):
        super(GmmSegLoss, self).__init__()
        self.k = GMM_NUM
    
    def forward(self, input, mu, var, pi, d, d0, epoch=None, total_epochs=None):
        epsilon = 1e-6
    
        B, C, H, W = input.shape
        input = input.unsqueeze(1)  # [B, 1, C, H, W]

        # 狄利克雷分布参数先验
        d0 = d0.reshape(B, self.k, -1, H, W)
        
        # 重塑参数
        mu  = mu.reshape(B, self.k, -1, H, W)
        var = var.reshape(B, self.k, -1, H, W)
        pi  = pi.reshape(B, self.k, -1, H, W)
        d   = d.reshape(B, self.k, -1, H, W)

        log_pi = torch.log(pi + epsilon)  # 防止 log(0)
        d_sum = d.sum(dim=1, keepdim=True)  # [B, 1, C, H, W]
        d0_sum = d0.sum(dim=1, keepdim=True)  # [B, 1, C, H, W]

        # 计算 loss_1
        log_gaussian = -0.5 * ((input - mu) ** 2 / var + torch.log(2 * math.pi * var))
        log_gaussian = log_gaussian.sum(dim=2, keepdim=True)
        pi_i_k = pi.sum(dim=2, keepdim=True)  # [B, 1, C, H, W]
        loss_1 = -torch.sum(pi_i_k * log_gaussian, dim=1).mean()

        # 计算 loss_2, torch.digamma 双伽马函数是伽马函数的对数的导数
        digamma_diff = torch.digamma(d + epsilon) - torch.digamma(d_sum + epsilon)
        loss_2 = torch.sum(pi * (log_pi - digamma_diff), dim=1).mean()


        # 计算 loss_3, torch.lgamma 伽马函数的对数
        norm_posterior = torch.lgamma(d_sum + epsilon) - torch.lgamma(d + epsilon).sum(dim=1, keepdim=True)
        norm_prior = torch.lgamma(d0_sum + epsilon) - torch.lgamma(d0 + epsilon).sum(dim=1, keepdim=True)
        loss_3 = torch.mean(norm_posterior - norm_prior) + torch.mean(((d - d0) * digamma_diff).sum(dim=1))


        # 调整损失权重
        weight_1 = 1
        weight_2 = 1
        if epoch < 50:
            weight_3 = 0.1
        elif epoch < 100:
            weight_3 = 0.05
        elif epoch < 150:
            weight_3 = 0.01
        else:
            weight_3 = 0.001

        # 应用权重
        loss_1 = loss_1 * weight_1
        loss_2 = loss_2 * weight_2
        loss_3 = loss_3 * weight_3
        loss_mse = nn.MSELoss()(d, d0) * 0  # 使用均方误差损失来计算 d 和 d0 之间的差异

        # 计算总损失
        total_loss = loss_1 + loss_2 + loss_3 + loss_mse

        return total_loss, loss_1, loss_2, loss_3, loss_mse




def test_gmmseg_loss_accurate_mu_var():
    # 初始化 GmmSegLoss
    k = 4
    loss_fn = GmmSegLoss(k=k)

    # 构造测试输入
    B, C, H, W = 2, 4, 8, 8  # Batch size, Channels, Height, Width
    input = torch.randn(B, C, H, W)  # 模拟输入数据

    # 设置 mu 非常接近 input
    mu = torch.randn(B, k*C, H, W) * 0.01 + input.repeat(1, k, 1, 1)  # 确保 mu 非常接近 input

    # 设置 var 为一个非常小的正数
    var = torch.abs(torch.randn(B, k*C, H, W)) * 0.01 + 1e-6  # 确保 var 是正数且不为零

    # 构造 pi 和 d
    d = torch.abs(torch.randn(B, k, H, W)) + 1e-6  # 狄利克雷分布的后验参数
    d0 = torch.abs(torch.randn(B, k, H, W)) + 1e-6  # 狄利克雷分布的先验参数
    pi = d / torch.sum(d, dim=1, keepdim=True)  # 归一化为概率

    print("mu shape:", mu.shape)
    print("var shape:", var.shape)
    print("pi shape:", pi.shape)
    print("d shape:", d.shape)
    print("d0 shape:", d0.shape)

    # 计算损失
    total_loss, loss_1, loss_2, loss_3, loss_mse = loss_fn(input, mu, var, pi, d, d0)

    # 打印损失值
    print(f"Total Loss: {total_loss.item()}")
    print(f"Loss 1 (Gaussian NLL): {loss_1.item()}")
    print(f"Loss 2 (Dirichlet KL): {loss_2.item()}")
    print(f"Loss 3 (Dirichlet Regularization): {loss_3.item()}")
    print(f"Loss MSE: {loss_mse.item()}")

    # 验证损失值是否为有限值
    assert torch.isfinite(total_loss).all(), "Total loss contains NaN or Inf!"
    assert torch.isfinite(loss_1).all(), "Loss 1 contains NaN or Inf!"
    assert torch.isfinite(loss_2).all(), "Loss 2 contains NaN or Inf!"
    assert torch.isfinite(loss_3).all(), "Loss 3 contains NaN or Inf!"
    assert torch.isfinite(loss_mse).all(), "Loss MSE contains NaN or Inf!"


if __name__ == "__main__":
    test_gmmseg_loss_accurate_mu_var()
