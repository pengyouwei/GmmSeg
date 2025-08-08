import os
import numpy as np
import matplotlib.pyplot as plt



def create_visualization(image_show, feature_show, mu_show, var_show, pi_show, 
                        d0_show, d1_show, label_show, pred_show, slice_id, 
                        output_dir, epoch, logger):
    """
    创建可视化图像
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印统计信息
    if logger:
        # 打印mu, var, pi的统计信息
        logger.info(f"=== Slice {slice_id} 参数统计 ===")
        logger.info(f"μ 范围: [{mu_show.min():.3f}, {mu_show.max():.3f}] (期望: [-4, 4])")
        logger.info(f"σ² 范围: [{var_show.min():.6f}, {var_show.max():.3f}] (期望: [1e-6, 2.25])")
        logger.info(f"π 范围: [{pi_show.min():.3f}, {pi_show.max():.3f}] (期望: [0, 1])")
        logger.info(f"特征 范围: [{feature_show.min():.3f}, {feature_show.max():.3f}] (期望: [-3, 3])")
        
        # 检查参数是否在预期范围内
        mu_in_range = (mu_show.min() >= -4.1) and (mu_show.max() <= 4.1)
        var_in_range = (var_show.min() >= 0) and (var_show.max() <= 2.3)
        pi_in_range = (pi_show.min() >= -0.01) and (pi_show.max() <= 1.01)
        
        if not mu_in_range:
            logger.warning(f"⚠️  μ 超出预期范围 [-4, 4]!")
        if not var_in_range:
            logger.warning(f"⚠️  σ² 超出预期范围 [1e-6, 2.25]!")
        if not pi_in_range:
            logger.warning(f"⚠️  π 超出预期范围 [0, 1]!")
    
    # 第一个可视化图形 - mu, var, pi
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 0.1, 0.3], 
                        height_ratios=[1, 1, 1], 
                        hspace=0.4, wspace=0.3, 
                        left=0.05, right=0.85, top=0.95, bottom=0.05)
    
    # 显示 image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_show, cmap='gray')
    ax.set_title(f'Image_{slice_id}\n[{image_show.min():.2f}, {image_show.max():.2f}]')
    ax.axis('off')
    
    # 显示 label
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(label_show, cmap='gray')
    ax.set_title(f'Label_{slice_id}')
    ax.axis('off')
    
    # 显示 mu 的四个通道
    mu_axes = []
    for j in range(4):
        ax = fig.add_subplot(gs[0, j + 1])
        im = ax.imshow(mu_show[j], cmap='gray', vmin=-4.0, vmax=4.0)
        ax.set_title(f'μ_{j}\n[{mu_show[j].min():.2f}, {mu_show[j].max():.2f}]')
        ax.axis('off')
        mu_axes.append((ax, im))
    
    cbar_ax_mu = fig.add_subplot(gs[0, 5])
    cbar_mu = plt.colorbar(mu_axes[-1][1], cax=cbar_ax_mu)
    cbar_mu.set_label('μ value', rotation=270, labelpad=15)
    
    # 显示 var 的四个通道
    var_axes = []
    for j in range(4):
        ax = fig.add_subplot(gs[1, j + 1])
        im = ax.imshow(var_show[j], cmap='gray', vmin=0, vmax=2.25)
        ax.set_title(f'σ²_{j}\n[{var_show[j].min():.3f}, {var_show[j].max():.3f}]')
        ax.axis('off')
        var_axes.append((ax, im))
    
    cbar_ax_var = fig.add_subplot(gs[1, 5])
    cbar_var = plt.colorbar(var_axes[-1][1], cax=cbar_ax_var)
    cbar_var.set_label('σ² value', rotation=270, labelpad=15)
    
    # 显示 pi 的四个通道
    pi_axes = []
    for j in range(4):
        ax = fig.add_subplot(gs[2, j + 1])
        im = ax.imshow(pi_show[j], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'π_{j}\n[{pi_show[j].min():.3f}, {pi_show[j].max():.3f}]')
        ax.axis('off')
        pi_axes.append((ax, im))
    
    cbar_ax_pi = fig.add_subplot(gs[2, 5])
    cbar_pi = plt.colorbar(pi_axes[-1][1], cax=cbar_ax_pi)
    cbar_pi.set_label('π value', rotation=270, labelpad=15)
    
    if epoch % 5 == 0:
        plt.savefig(f"{output_dir}/mu_var_pi_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 第二个可视化图形 - d0/d1 对比
    fig2 = plt.figure(figsize=(14, 6))
    gs2 = fig2.add_gridspec(2, 6, 
                          hspace=0.15, wspace=0.25, 
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # 第一行：image, label, d0的四个通道
    ax = fig2.add_subplot(gs2[0, 0])
    ax.imshow(image_show, cmap='gray')
    ax.set_title(f'Image_{slice_id}', fontsize=10)
    ax.axis('off')

    ax = fig2.add_subplot(gs2[0, 1])
    ax.imshow(label_show, cmap='gray')
    ax.set_title(f'Label_{slice_id}', fontsize=10)
    ax.axis('off')

    for j in range(4):
        ax = fig2.add_subplot(gs2[0, j + 2])
        ax.imshow(d0_show[j], cmap='gray', vmin=0.5, vmax=10)
        ax.set_title(f'd0_{j}_{slice_id}', fontsize=10)
        ax.axis('off')
    
    # 第二行：空位，pred，d1的四个通道
    ax = fig2.add_subplot(gs2[1, 1])
    ax.imshow(pred_show, cmap='gray')
    ax.set_title('Prediction', fontsize=10)
    ax.axis('off')

    for j in range(4):
        ax = fig2.add_subplot(gs2[1, j + 2])
        ax.imshow(d1_show[j], cmap='gray', vmin=0.5, vmax=10)
        ax.set_title(f'd1_{j}', fontsize=10)
        ax.axis('off')

    if epoch % 5 == 0:
        plt.savefig(f"{output_dir}/output_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()