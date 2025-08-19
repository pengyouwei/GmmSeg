import os
import numpy as np
import matplotlib.pyplot as plt
from config import Config


def create_visualization(image_show, feature_show, mu_show, var_show, pi_show,
                         d0_show, d1_show, label_show, pred_show, slice_id,
                         output_dir, epoch, logger=None):
    """根据 train.py 传入的张量补全可视化：
    图1(3x5 灰度):
      R1: Image + 4 个组件 μ L2 范数
      R2: Label + 4 个组件 Var 迹
      R3: Pred  + 4 个组件 Var log|Σ| (对角协方差 → log(prod σ²))
    图2(3x5 灰度):
      R1: Image + d0 4 通道 (Dirichlet 先验浓度)
      R2: Label + d1 4 通道 (预测 Dirichlet 浓度)
      R3: Pred  + π   4 通道 (后验/变分概率)

    约束范围 (固定归一化，便于跨 epoch 比较)：
      μ:   每组件 L2，范围估计 [0, MU_RANGE * sqrt(C)]
      迹:  [C * VAR_MIN, C * VAR_MAX]
      log|Σ|: [log(VAR_MIN^C), log(VAR_MAX^C)]
      d0/d1: [0.5, 10.0]
      π: [0,1]

    仅每 5 个 epoch (含 0) 保存一次；缺失数据则跳过。
    全部使用灰度 colormap，确保风格统一。
    """
    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)

    # 基础校验
    required = [image_show, mu_show, var_show, pi_show,
                d0_show, d1_show, label_show, pred_show]
    if any(v is None for v in required):
        if logger:
            logger.warning("Skip visualization: some required data is None")
        return

    if (epoch % 1) != 0 and epoch != 0:
        return  # 只在每1个 epoch 或第 0 个保存

    try:
        # -------- 预处理形状 --------
        # mu_show, var_show: [K*C, H, W] 需 reshape 为 [K,C,H,W]
        K = cfg.GMM_NUM
        C = cfg.FEATURE_NUM
        H, W = mu_show.shape[-2], mu_show.shape[-1]

        def _safe_reshape(x):
            if x.ndim == 3 and x.shape[0] == K * C:
                return x.reshape(K, C, H, W)
            if x.ndim == 4 and x.shape[0] == K and x.shape[1] == C:
                return x
            # 尝试自动推断 C
            if x.ndim == 3 and (x.shape[0] % K == 0):
                c2 = x.shape[0] // K
                return x.reshape(K, c2, H, W)
            return None

        mu_kc = _safe_reshape(mu_show)
        var_kc = _safe_reshape(var_show)
        if mu_kc is None or var_kc is None:
            if logger:
                logger.warning(
                    "Skip visualization: cannot reshape mu/var to [K,C,H,W]")
            return

        # 后验 / 浓度检查
        if pi_show.ndim != 3 or pi_show.shape[0] != K:
            if logger:
                logger.warning("Skip visualization: pi_show shape mismatch")
            return
        if d0_show.ndim != 3 or d0_show.shape[0] != K:
            if logger:
                logger.warning("Skip visualization: d0_show shape mismatch")
            return
        if d1_show.ndim != 3 or d1_show.shape[0] != K:
            if logger:
                logger.warning("Skip visualization: d1_show shape mismatch")
            return

        # -------- 统计量计算 --------
        mu_l2 = np.sqrt(np.sum(mu_kc ** 2, axis=1))  # [K,H,W]
        var_tr = np.sum(var_kc, axis=1)              # [K,H,W]
        var_det = np.prod(np.clip(var_kc, cfg.VAR_MIN,
                          cfg.VAR_MAX), axis=1)  # [K,H,W]
        var_logdet = np.log(
            np.clip(var_det, cfg.VAR_MIN ** C, cfg.VAR_MAX ** C))

        # -------- 归一化工具 --------
        def norm_fixed(x, lo, hi):
            x = np.asarray(x, dtype=np.float32)
            return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        # 设定固定范围
        mu_max = cfg.MU_RANGE * np.sqrt(C)
        trace_lo, trace_hi = C * cfg.VAR_MIN, C * cfg.VAR_MAX
        logdet_lo = C * np.log(cfg.VAR_MIN)
        logdet_hi = C * np.log(cfg.VAR_MAX)

        # -------- 图1 --------
        fig1, axes1 = plt.subplots(3, 5, figsize=(15, 9))
        fig1.suptitle(
            f"Slice {slice_id} - GMM Stats (Epoch {epoch})", fontsize=12)

        # Row1 Col1: Image
        axes1[0, 0].imshow(norm_fixed(image_show, image_show.min(
        ), image_show.max()), cmap='gray', vmin=0, vmax=1)
        axes1[0, 0].set_title("Image")
        axes1[0, 0].axis('off')
        # Row1 Col2-5: μ L2
        for j in range(4):
            ax = axes1[0, j + 1]
            if j < K:
                ax.imshow(norm_fixed(
                    mu_l2[j], 0.0, mu_max), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"muL2[{j}]")
            ax.axis('off')

        # Row2 Col1: Label
        axes1[1, 0].imshow(norm_fixed(label_show, label_show.min(
        ), label_show.max()), cmap='gray', vmin=0, vmax=1)
        axes1[1, 0].set_title("Label")
        axes1[1, 0].axis('off')
        # Row2 Col2-5: Var trace
        for j in range(4):
            ax = axes1[1, j + 1]
            if j < K:
                ax.imshow(norm_fixed(
                    var_tr[j], trace_lo, trace_hi), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"trΣ[{j}]")
            ax.axis('off')

        # Row3 Col1: Pred
        axes1[2, 0].imshow(norm_fixed(pred_show, pred_show.min(
        ), pred_show.max()), cmap='gray', vmin=0, vmax=1)
        axes1[2, 0].set_title("Pred")
        axes1[2, 0].axis('off')
        # Row3 Col2-5: log|Σ|
        for j in range(4):
            ax = axes1[2, j + 1]
            if j < K:
                ax.imshow(norm_fixed(
                    var_logdet[j], logdet_lo, logdet_hi), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"log|Σ|[{j}]")
            ax.axis('off')

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        # 新命名: eXXXX_sliceS_stats.png (包含 μ L2 / trace / log|Σ|)
        safe_slice = slice_id if isinstance(slice_id, (int, np.integer)) and slice_id >= 0 else 'u'
        fig1.savefig(os.path.join(
            output_dir, f"e{epoch:04d}_slice{safe_slice}_stats.png"), dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # -------- 图2 --------
        fig2, axes2 = plt.subplots(3, 5, figsize=(15, 9))
        fig2.suptitle(
            f"Slice {slice_id} - d0/d1/pi (Epoch {epoch})", fontsize=12)

        # Row1: Image + d0
        axes2[0, 0].imshow(norm_fixed(image_show, image_show.min(
        ), image_show.max()), cmap='gray', vmin=0, vmax=1)
        axes2[0, 0].set_title("Image")
        axes2[0, 0].axis('off')
        for j in range(4):
            ax = axes2[0, j + 1]
            if j < K:
                ax.imshow(norm_fixed(
                    d0_show[j], 0.5, 10.0), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"d0[{j}]")
            ax.axis('off')

        # Row2: Label + d1
        axes2[1, 0].imshow(norm_fixed(label_show, label_show.min(
        ), label_show.max()), cmap='gray', vmin=0, vmax=1)
        axes2[1, 0].set_title("Label")
        axes2[1, 0].axis('off')
        for j in range(4):
            ax = axes2[1, j + 1]
            if j < K:
                ax.imshow(norm_fixed(
                    d1_show[j], 0.5, 10.0), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"d[{j}]")
            ax.axis('off')

        # Row3: Pred + pi
        axes2[2, 0].imshow(norm_fixed(pred_show, pred_show.min(
        ), pred_show.max()), cmap='gray', vmin=0, vmax=1)
        axes2[2, 0].set_title("Pred")
        axes2[2, 0].axis('off')
        for j in range(4):
            ax = axes2[2, j + 1]
            if j < K:
                ax.imshow(norm_fixed(
                    pi_show[j], 0.0, 1.0), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"pi[{j}]")
            ax.axis('off')

        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        # 新命名: eXXXX_sliceS_dirichlet.png (包含 d0 / d1 / pi)
        fig2.savefig(os.path.join(
            output_dir, f"e{epoch:04d}_slice{safe_slice}_dirichlet.png"), dpi=150, bbox_inches='tight')
        plt.close(fig2)

    except Exception as e:
        if logger:
            logger.warning(f"Visualization failed: {e}")
        # 避免因可视化失败中断训练
        return
