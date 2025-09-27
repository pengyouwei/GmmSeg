import numpy as np
from scipy.spatial.distance import cdist


def preprocess_input(y_pred, y_true, background):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise ValueError("Both y_pred and y_true must be numpy arrays.")
    if y_pred.shape != y_true.shape:
        raise ValueError("Shape mismatch between y_pred and y_true.")
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")
    start_class = 0 if background else 1
    return y_pred, y_true, start_class


def hausdorff_distance(y_pred, y_true, cls=1):
    pred_pts = np.argwhere(y_pred == cls)
    true_pts = np.argwhere(y_true == cls)
    if len(pred_pts) == 0 or len(true_pts) == 0:
        return np.nan
    dists = cdist(pred_pts, true_pts)
    return max(np.max(np.min(dists, axis=1)), np.max(np.min(dists, axis=0)))


def hausdorff_distance_95(y_pred, y_true, cls=1):
    pred_pts = np.argwhere(y_pred == cls)
    true_pts = np.argwhere(y_true == cls)
    if len(pred_pts) == 0 or len(true_pts) == 0:
        return np.nan
    dists = cdist(pred_pts, true_pts)
    d1 = np.min(dists, axis=1)  # 每个预测点到真实的最近距离
    d2 = np.min(dists, axis=0)  # 每个真实点到预测的最近距离
    return max(np.percentile(d1, 95), np.percentile(d2, 95))


def average_surface_distance(y_pred, y_true, cls=1):
    pred_pts = np.argwhere(y_pred == cls)
    true_pts = np.argwhere(y_true == cls)
    if len(pred_pts) == 0 or len(true_pts) == 0:
        return np.nan
    dists = cdist(pred_pts, true_pts)
    return (np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))) / 2.0


def evaluate_segmentation(y_pred, y_true, num_classes, background=False):
    """
    统一评价指标：
      - Dice
      - IoU
      - Pixel Accuracy
      - PRE / REC / F1
      - HD Distance (HD)
      - Average Surface Distance (ASD)
    """
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, background)
    results = {}

    # Pixel Accuracy（全局）
    results["Pixel_Accuracy"] = np.mean(y_pred == y_true)

    # Per-class metrics
    dice_scores = {}
    iou_scores = {}
    hd_scores, asd_scores = {}, {}

    for cls in range(start_class, num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)

        TP = np.sum(pred_cls * true_cls)
        FP = np.sum(pred_cls * (1 - true_cls))
        FN = np.sum((1 - pred_cls) * true_cls)

        # Dice
        denom_dice = (np.sum(pred_cls) + np.sum(true_cls))
        dice_scores[cls] = (2 * TP / denom_dice) if denom_dice > 0 else 1.0

        # IoU
        denom_iou = (TP + FP + FN)
        iou_scores[cls] = (TP / denom_iou) if denom_iou > 0 else 1.0

        # HD Distance & ASD
        # hd_scores[cls] = hausdorff_distance_95(y_pred, y_true, cls)
        # asd_scores[cls] = average_surface_distance(y_pred, y_true, cls)

    # 汇总结果
    results["Dice"] = {
        "mean": np.nanmean(list(dice_scores.values())),
        "per_class": dice_scores
    }
    results["IoU"] = {
        "mean": np.nanmean(list(iou_scores.values())),
        "per_class": iou_scores
    }
    # results["HD"] = {
    #     "mean": np.nanmean(list(hd_scores.values())),
    #     "per_class": hd_scores
    # }
    # results["ASD"] = {
    #     "mean": np.nanmean(list(asd_scores.values())),
    #     "per_class": asd_scores
    # }

    return results


if __name__ == "__main__":
    import pprint

    num_classes = 4  # 0=background, 1,2,3=organs

    # ===== Case1: 完美匹配 =====
    y_true1 = np.zeros((6, 6), dtype=np.int32)
    y_true1[2:4, 2:4] = 1
    y_pred1 = y_true1.copy()

    # ===== Case2: Over-segmentation =====
    y_true2 = np.zeros((6, 6), dtype=np.int32)
    y_true2[2:4, 2:4] = 1
    y_pred2 = np.zeros((6, 6), dtype=np.int32)
    y_pred2[1:5, 1:5] = 1  # 预测区域比GT大

    # ===== Case3: Under-segmentation =====
    y_true3 = np.zeros((6, 6), dtype=np.int32)
    y_true3[1:5, 1:5] = 1
    y_pred3 = np.zeros((6, 6), dtype=np.int32)
    y_pred3[2:4, 2:4] = 1  # 预测区域比GT小

    # ===== Case4: 类别混淆 =====
    y_true4 = np.zeros((6, 6), dtype=np.int32)
    y_true4[2:4, 2:4] = 1
    y_pred4 = np.zeros((6, 6), dtype=np.int32)
    y_pred4[2:4, 2:4] = 2  # 预测成了类别2

    # 创建案例列表
    cases = [
        ("Case1: Perfect match", y_pred1, y_true1),
        ("Case2: Over-segmentation", y_pred2, y_true2),
        ("Case3: Under-segmentation", y_pred3, y_true3),
        ("Case4: Class confusion", y_pred4, y_true4),
    ]

    # 循环评估
    for name, y_pred, y_true in cases:
        print(f"\n=== {name} ===")
        results = evaluate_segmentation(
            y_pred, y_true, num_classes, background=False)
        pprint.pprint(results)
