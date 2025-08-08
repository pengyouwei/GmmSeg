import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GmmSegLoss(nn.Module):
    def __init__(self, GMM_NUM=4):
        super(GmmSegLoss, self).__init__()
        self.k = GMM_NUM
    
    def forward(self, image_4_features, mu, var, pi, d, d0, epoch=None, total_epochs=None):
        epsilon = 1e-6
    
        B, C, H, W = image_4_features.shape
        input = image_4_features.unsqueeze(1)  # [B, 1, C, H, W]

        # 重塑GMM参数 - 修正维度处理
        mu = mu.reshape(B, self.k, C, H, W)  # [B, K, C, H, W]
        var = var.reshape(B, self.k, C, H, W)  # [B, K, C, H, W] 
        
        # pi和d保持原有维度：[B, K, H, W]
        pi = pi.reshape(B, self.k, H, W)   # [B, K, H, W]
        d = d.reshape(B, self.k, H, W)     # [B, K, H, W]
        d0 = d0.reshape(B, self.k, H, W)   # [B, K, H, W]

        # 确保pi归一化且数值稳定
        pi = pi / (torch.sum(pi, dim=1, keepdim=True) + epsilon)  # [B, K, H, W]
        log_pi = torch.log(pi + epsilon)  # [B, K, H, W]
        
        
        # Dirichlet参数求和
        d_sum = d.sum(dim=1, keepdim=True)    # [B, 1, H, W]
        d0_sum = d0.sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # 计算 loss_1: 重建损失 (Reconstruction Loss)
        # 对应 ELBO 第一项: -E_q(z,Ω|x) log p(x | z, Ω)
        
        # 计算每个高斯分量的对数概率密度
        diff = input - mu  # [B, K, C, H, W]
        log_exponent = -0.5 * torch.sum((diff ** 2) / (var + epsilon), dim=2)  # [B, K, H, W]
        log_normalizer = -0.5 * C * math.log(2 * math.pi) - 0.5 * torch.sum(torch.log(var + epsilon), dim=2)  # [B, K, H, W]
        log_gaussian = log_exponent + log_normalizer  # [B, K, H, W]
        
        
        # 期望形式：L1 = -Σ_i Σ_k q(z_ik|x_i) * log N(x_i | μ_ik, σ²_ik)
        # 重要说明：这里的 pi 在期望形式中应该理解为后验概率 q(z_ik|x_i)
        # 在变分推断中，我们用变分后验 q(z_ik|x_i) 来近似真实后验
        posterior_prob = pi  # pi 作为后验概率 q(z_ik|x_i)
        weighted_log_likelihood = posterior_prob * log_gaussian  # [B, K, H, W]
        reconstruction_loss = -torch.mean(torch.sum(weighted_log_likelihood, dim=1))  # 对K维求和，然后平均
        

        # 计算 loss_2: KL散度项1 - 潜在变量正则化
        # 对应 ELBO 第二项: E_q(Ω) KL(q(z | Ω, x) || p(z | Ω))
        # 
        # 重要区分：
        # - q(z_ik|x_i) = pi（后验概率，神经网络输出，依赖于观测x）
        # - p(z_ik) = E_q[π_k]（先验混合权重的期望，来自Dirichlet分布，独立于x）
        #
        # KL散度计算：KL(q(z|x) || p(z)) = Σ_k q(z_k|x) * log(q(z_k|x) / p(z_k))
        
        digamma_d = torch.digamma(d + epsilon)  # [B, K, H, W]
        digamma_d_sum = torch.digamma(d_sum + epsilon)  # [B, 1, H, W]
        expected_log_pi = digamma_d - digamma_d_sum  # [B, K, H, W]
        
        
        # 混合模型形式：使用传统的 π 与 Dirichlet 后验期望的 KL 散度
        kl_divergence = pi * (log_pi - expected_log_pi)  # [B, K, H, W]
        
        kl_z_raw = torch.mean(torch.sum(kl_divergence, dim=1))  # 对K维求和，然后平均
        
        # 数值稳定性：确保KL散度非负（理论上KL散度应该≥0）
        kl_z_loss = torch.clamp(kl_z_raw, min=0.0)


        # 计算 loss_3: KL散度项2 - 参数先验正则化
        # 对应 ELBO 第三项: KL(q(Ω | x) || p(Ω))
        # 约束学习的 Dirichlet 参数 d 接近先验 d0
        # 计算标准化常数的对数
        log_norm_posterior = torch.lgamma(d_sum + epsilon) - torch.sum(torch.lgamma(d + epsilon), dim=1, keepdim=True)  # [B, 1, H, W]
        log_norm_prior = torch.lgamma(d0_sum + epsilon) - torch.sum(torch.lgamma(d0 + epsilon), dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Dirichlet KL散度: KL(Dir(d) || Dir(d0))
        kl_dir_term1 = torch.mean(log_norm_prior - log_norm_posterior)
        kl_dir_term2 = torch.mean(torch.sum((d - d0) * expected_log_pi, dim=1))
        kl_omega_raw = kl_dir_term1 + kl_dir_term2
        
        # 数值稳定性：确保KL散度非负（理论上KL散度应该≥0）
        kl_omega_loss = torch.clamp(kl_omega_raw, min=0.0)

        # 改进的动态权重调整策略
        weight_1 = 1.0
        weight_2 = 1.0
        
        # 先验正则化权重的自适应调整
        if epoch is not None and total_epochs is not None:
            # 使用余弦退火策略，训练初期先验约束较强，后期逐渐减弱
            progress = epoch / total_epochs
            
            if progress < 0.1:  # 前10%训练阶段：强先验引导
                weight_3 = 1.0
            elif progress < 0.3:  # 10%-30%阶段：平滑过渡
                # 从1.0平滑下降到0.3
                weight_3 = 1.0 - 0.7 * ((progress - 0.1) / 0.2)
            elif progress < 0.6:  # 30%-60%阶段：中等约束
                # 从0.3平滑下降到0.1
                weight_3 = 0.3 - 0.2 * ((progress - 0.3) / 0.3)
            elif progress < 0.8:  # 60%-80%阶段：弱约束
                # 从0.1平滑下降到0.01
                weight_3 = 0.1 - 0.09 * ((progress - 0.6) / 0.2)
            else:  # 80%+阶段：最小约束，网络自由优化
                weight_3 = 0.01
        elif epoch is not None:
            # 如果没有total_epochs，使用固定epoch数的策略
            if epoch < 20:  # 强先验引导期
                weight_3 = 1.0
            elif epoch < 50:  # 过渡期
                weight_3 = 1.0 - 0.7 * ((epoch - 20) / 30)
            elif epoch < 100:  # 中等约束期
                weight_3 = 0.3 - 0.2 * ((epoch - 50) / 50)
            elif epoch < 150:  # 弱约束期
                weight_3 = 0.1 - 0.09 * ((epoch - 100) / 50)
            else:  # 最小约束期
                weight_3 = 0.01
        else:
            weight_3 = 0.5  # 默认中等约束

        # 应用权重
        loss_1 = reconstruction_loss * weight_1
        loss_2 = kl_z_loss * weight_2
        loss_3 = kl_omega_loss * weight_3
        
        # 改进的辅助损失：在训练初期更强调d和d0的一致性
        if epoch is not None and epoch < 30:
            # 训练初期：强制d向d0靠拢
            mse_weight = 0.1 * (1.0 - epoch / 30)  # 从0.1线性下降到0
        else:
            # 训练后期：最小约束
            mse_weight = 0.001
        loss_mse = nn.MSELoss()(d, d0) * mse_weight

        # 计算总损失 = -ELBO (我们最小化-ELBO等价于最大化ELBO)
        total_loss = loss_1 + loss_2 + loss_3 + loss_mse
        
        # 数值稳定性检查
        if not torch.isfinite(total_loss).all():
            print("⚠️  Warning: 总损失包含NaN或Inf值!")
            print(f"reconstruction_loss: {loss_1.item():.6f}, kl_z_loss: {loss_2.item():.6f}, kl_omega_loss: {loss_3.item():.6f}")

        return total_loss, loss_1, loss_2, loss_3, loss_mse, weight_3




def test_gmmseg_loss_accurate_mu_var():
    """测试修正后的GmmSegLoss（期望形式）"""
    print("=== 测试修正后的GmmSegLoss（期望形式）===")
    k = 4
    loss_fn = GmmSegLoss(GMM_NUM=k, use_expectation_form=True)  # 使用期望形式

    # 构造测试输入
    B, C, H, W = 2, 4, 8, 8  # Batch size, Channels, Height, Width
    image_4_features = torch.randn(B, C, H, W)  # 模拟特征

    # 设置GMM参数
    mu = torch.randn(B, k*C, H, W)  # [B, K*C, H, W] -> reshape to [B, K, C, H, W]
    var = torch.abs(torch.randn(B, k*C, H, W)) + 1e-6  # 确保方差为正

    # 设置混合系数和Dirichlet参数
    pi_raw = torch.abs(torch.randn(B, k, H, W)) + 1e-6
    pi = pi_raw / torch.sum(pi_raw, dim=1, keepdim=True)  # 归一化为概率

    d = torch.abs(torch.randn(B, k, H, W)) + 1.0  # 后验Dirichlet参数
    d0 = torch.abs(torch.randn(B, k, H, W)) + 1.0  # 先验Dirichlet参数

    print(f"image_4_features shape: {image_4_features.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"var shape: {var.shape}")
    print(f"pi shape: {pi.shape}")
    print(f"d shape: {d.shape}")
    print(f"d0 shape: {d0.shape}")

    # 测试损失计算
    try:
        total_loss, loss_1, loss_2, loss_3, loss_mse, weight_3 = loss_fn(
            image_4_features, mu, var, pi, d, d0, epoch=10, total_epochs=200
        )

        print(f"\n=== ELBO损失值分解（期望形式）===")
        print(f"Total Loss (-ELBO): {total_loss.item():.6f}")
        print(f"Loss 1 (Expectation Reconstruction): {loss_1.item():.6f}")
        print(f"Loss 2 (KL z regularization): {loss_2.item():.6f}")
        print(f"Loss 3 (KL Ω regularization): {loss_3.item():.6f}")
        print(f"Loss MSE: {loss_mse.item():.6f}")
        print(f"Weight 3 (Prior strength): {weight_3:.6f}")
        
        # ELBO值 (应该为负数，越大越好)
        elbo_value = -total_loss.item()
        print(f"ELBO Value: {elbo_value:.6f}")

        # 验证损失值是否为有限值
        assert torch.isfinite(total_loss).all(), "Total loss contains NaN or Inf!"
        assert torch.isfinite(loss_1).all(), "Loss 1 contains NaN or Inf!"
        assert torch.isfinite(loss_2).all(), "Loss 2 contains NaN or Inf!"
        assert torch.isfinite(loss_3).all(), "Loss 3 contains NaN or Inf!"
        assert torch.isfinite(loss_mse).all(), "Loss MSE contains NaN or Inf!"

        print("\n✅ 期望形式损失函数实现成功！所有值都是有限的。")
        
        # 同时测试混合模型形式进行对比
        print(f"\n=== 对比混合模型形式 ===")
        loss_fn_mixture = GmmSegLoss(GMM_NUM=k, use_expectation_form=False)
        total_loss_mix, loss_1_mix, _, _, _, _ = loss_fn_mixture(
            image_4_features, mu, var, pi, d, d0, epoch=10, total_epochs=200
        )
        
        print(f"期望形式重建损失: {loss_1.item():.6f}")
        print(f"混合形式重建损失: {loss_1_mix.item():.6f}")
        print(f"差异: {(loss_1.item() - loss_1_mix.item()):.6f}")
        print(f"根据Jensen不等式：期望形式 ≥ 混合形式 {'✅' if loss_1.item() >= loss_1_mix.item() else '❌'}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise

def test_loss_backward():
    """测试损失函数的反向传播"""
    print("\n=== 测试反向传播 ===")
    k = 4
    loss_fn = GmmSegLoss(GMM_NUM=k, use_expectation_form=True)  # 使用期望形式

    B, C, H, W = 1, 4, 16, 16
    image_4_features = torch.randn(B, C, H, W, requires_grad=False)
    
    mu = torch.randn(B, k*C, H, W, requires_grad=True)
    var = torch.abs(torch.randn(B, k*C, H, W)) + 1e-6
    var.requires_grad_(True)
    
    pi_raw = torch.abs(torch.randn(B, k, H, W)) + 1e-6
    pi = pi_raw / torch.sum(pi_raw, dim=1, keepdim=True)
    pi.requires_grad_(True)
    
    d = torch.abs(torch.randn(B, k, H, W)) + 1.0
    d.requires_grad_(True)
    d0 = torch.abs(torch.randn(B, k, H, W)) + 1.0

    total_loss, _, _, _, _, _ = loss_fn(image_4_features, mu, var, pi, d, d0, epoch=10, total_epochs=200)
    
    # 测试反向传播
    total_loss.backward()
    
    print(f"mu.grad 是否存在: {mu.grad is not None}")
    print(f"var.grad 是否存在: {var.grad is not None}")
    print(f"pi.grad 是否存在: {pi.grad is not None}")
    print(f"d.grad 是否存在: {d.grad is not None}")
    
    if mu.grad is not None:
        print(f"mu梯度范围: [{mu.grad.min().item():.6f}, {mu.grad.max().item():.6f}]")
    
    print("✅ 期望形式反向传播测试通过！")


if __name__ == "__main__":
    test_gmmseg_loss_accurate_mu_var()
    test_loss_backward()
