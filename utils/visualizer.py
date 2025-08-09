import os
import numpy as np
import matplotlib.pyplot as plt


def create_visualization(image_show, feature_show, mu_show, var_show, pi_show,
                        d0_show, d1_show, label_show, pred_show, slice_id,
                        output_dir, epoch, logger):
    """
    保存两张 3x5 网格图片：
    图1(3x5)：
      R1: [Image] + mu 的 4 个通道
      R2: [Label] + var 的 4 个通道
      R3: [Pred ] + pi 的 4 个通道
    图2(3x5)：
      R1: [Image] + feature 的 4 个通道
      R2: [Label] + d0 的 4 个通道
      R3: [Pred ] + d1 的 4 个通道
    """
    os.makedirs(output_dir, exist_ok=True)

    def _norm01(x, min_val=None, max_val=None):
        """Normalize numpy array to [0,1]. If range given, use fixed range; else use data min/max."""
        x = np.asarray(x, dtype=np.float32)
        if min_val is not None and max_val is not None:
            denom = float(max_val) - float(min_val)
            if denom <= 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            y = (x - float(min_val)) / (denom + 1e-8)
        else:
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            if not np.isfinite(xmin) or not np.isfinite(xmax) or (xmax - xmin) <= 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            y = (x - xmin) / (xmax - xmin + 1e-8)
        return np.clip(y, 0.0, 1.0)

    # 基础范围检查与告警
    if logger is not None:
        try:
            if not (-4.1 <= float(np.nanmin(mu_show)) <= 4.1 and -4.1 <= float(np.nanmax(mu_show)) <= 4.1):
                logger.warning("μ 超出预期范围 [-4, 4]")
        except Exception:
            pass
        try:
            if not (0.0 <= float(np.nanmin(var_show)) <= 2.3 and 0.0 <= float(np.nanmax(var_show)) <= 2.3):
                logger.warning("σ² 超出预期范围 [1e-6, 2.25]")
        except Exception:
            pass
        try:
            if not (-0.01 <= float(np.nanmin(pi_show)) <= 1.01 and -0.01 <= float(np.nanmax(pi_show)) <= 1.01):
                logger.warning("π 超出预期范围 [0, 1]")
        except Exception:
            pass

    # ---------- 图1：Image + (mu, var, pi) ----------
    fig1, axes1 = plt.subplots(3, 5, figsize=(15, 9))
    fig1.suptitle(f"Slice {slice_id} - mu/var/pi", fontsize=12)

    # R1C1: Image
    axes1[0, 0].imshow(_norm01(image_show), cmap='gray', vmin=0, vmax=1)
    axes1[0, 0].set_title("Image")
    axes1[0, 0].axis('off')

    # R1C2-5: mu 4通道
    for j in range(4):
        axes1[0, j + 1].imshow(_norm01(mu_show[j], -3.0, 3.0), cmap='gray', vmin=0, vmax=1)
        axes1[0, j + 1].set_title(f"mu[{j}]")
        axes1[0, j + 1].axis('off')

    # R2C1: Label
    axes1[1, 0].imshow(_norm01(label_show), cmap='gray', vmin=0, vmax=1)
    axes1[1, 0].set_title("Label")
    axes1[1, 0].axis('off')

    # R2C2-5: var 4通道
    for j in range(4):
        axes1[1, j + 1].imshow(_norm01(var_show[j], 0, 2.25), cmap='gray', vmin=0, vmax=1)
        axes1[1, j + 1].set_title(f"var[{j}]")
        axes1[1, j + 1].axis('off')

    # R3C1: Pred
    axes1[2, 0].imshow(_norm01(pred_show), cmap='gray', vmin=0, vmax=1)
    axes1[2, 0].set_title("Pred")
    axes1[2, 0].axis('off')

    # R3C2-5: pi 4通道
    for j in range(4):
        axes1[2, j + 1].imshow(_norm01(pi_show[j], 0.0, 1.0), cmap='gray', vmin=0, vmax=1)
        axes1[2, j + 1].set_title(f"pi[{j}]")
        axes1[2, j + 1].axis('off')

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if epoch % 5 == 0:
        fig1.savefig(f"{output_dir}/grid1_mu_var_pi_{epoch}.png", dpi=150)
    plt.close(fig1)

    # ---------- 图2：Image/feature + (d0, d1) ----------
    fig2, axes2 = plt.subplots(3, 5, figsize=(15, 9))
    fig2.suptitle(f"Slice {slice_id} - feature/d0/d1", fontsize=12)

    # R1C1: Image
    axes2[0, 0].imshow(_norm01(image_show), cmap='gray', vmin=0, vmax=1)
    axes2[0, 0].set_title("Image")
    axes2[0, 0].axis('off')

    # R1C2-5: feature 4通道（取前4个）
    # feature_show 形状应为 [C, H, W] 或 [H, W]（已 squeeze）
    if feature_show.ndim == 2:
        # 若只有一个通道，则复制填充
        feat_stack = [feature_show for _ in range(4)]
    else:
        c = feature_show.shape[0]
        idx = min(4, c)
        feat_stack = [feature_show[j] for j in range(idx)]
        if idx < 4:
            feat_stack += [feature_show[0]] * (4 - idx)
    for j in range(4):
        axes2[0, j + 1].imshow(_norm01(feat_stack[j], -3.0, 3.0), cmap='gray', vmin=0, vmax=1)
        axes2[0, j + 1].set_title(f"feat[{j}]")
        axes2[0, j + 1].axis('off')

    # R2C1: Label
    axes2[1, 0].imshow(_norm01(label_show), cmap='gray', vmin=0, vmax=1)
    axes2[1, 0].set_title("Label")
    axes2[1, 0].axis('off')

    # R2C2-5: d0 4通道
    for j in range(4):
        axes2[1, j + 1].imshow(_norm01(d0_show[j], 0.5, 10.0), cmap='gray', vmin=0, vmax=1)
        axes2[1, j + 1].set_title(f"d0[{j}]")
        axes2[1, j + 1].axis('off')

    # R3C1: Pred
    axes2[2, 0].imshow(_norm01(pred_show), cmap='gray', vmin=0, vmax=1)
    axes2[2, 0].set_title("Pred")
    axes2[2, 0].axis('off')

    # R3C2-5: d1 4通道
    for j in range(4):
        axes2[2, j + 1].imshow(_norm01(d1_show[j], 0.5, 10.0), cmap='gray', vmin=0, vmax=1)
        axes2[2, j + 1].set_title(f"d1[{j}]")
        axes2[2, j + 1].axis('off')

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    if epoch % 5 == 0:
        fig2.savefig(f"{output_dir}/grid2_feat_d0_d1_{epoch}.png", dpi=150)
    plt.close(fig2)