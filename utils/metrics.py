import numpy as np
from medpy import metric


def preprocess_input(y_pred, y_true, background):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise ValueError("Both y_pred and y_true must be numpy arrays.")
    if y_pred.shape != y_true.shape:
        raise ValueError("Shape mismatch between y_pred and y_true.")
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")
    if y_pred.ndim not in (2, 3):
        raise ValueError("Inputs must be 2D (H,W) or 3D (B,H,W) arrays.")
    start_class = 0 if background else 1
    return y_pred, y_true, start_class

def evaluate_segmentation(y_pred, y_true, num_classes, background=False, test=False):
    """
    统一评价指标：
      - Dice
      - Jaccard
      - HD95
      - ASD
    """
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, background)
    results = {}

    # Per-class metrics（仅保留最基础的计算，不再统计 std / per-sample 分布）
    dice: dict = {}
    jaccard: dict = {}
    hd95: dict = {}
    asd: dict = {}

    for cls in range(start_class, num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)

        # mean（保持原有行为）：当输入为(B,H,W)时，medpy 会把它当作 3D 体积整体计算
        try:
            dice[cls] = float(metric.binary.dc(pred_cls, true_cls))
        except Exception:
            dice[cls] = float("nan")

        if not test:
            continue  # 仅计算 Dice，节省时间

        try:
            jaccard[cls] = float(metric.binary.jc(pred_cls, true_cls))
        except Exception:
            jaccard[cls] = float("nan")
        try:
            hd95[cls] = float(metric.binary.hd95(pred_cls, true_cls))
        except Exception:
            hd95[cls] = float("nan")
        try:
            asd[cls] = float(metric.binary.asd(pred_cls, true_cls))
        except Exception:
            asd[cls] = float("nan")

    # 汇总结果：均值 + 每类单独值
    results["Dice"] = {
        "mean": float(np.nanmean(list(dice.values()))),
        "per_class": dice,
    }

    if not test:
        return results

    results["Jaccard"] = {
        "mean": float(np.nanmean(list(jaccard.values()))),
        "per_class": jaccard,
    }
    results["HD95"] = {
        "mean": float(np.nanmean(list(hd95.values()))),
        "per_class": hd95,
    }
    results["ASD"] = {
        "mean": float(np.nanmean(list(asd.values()))),
        "per_class": asd,
    }

    return results
