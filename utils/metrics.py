import numpy as np


def preprocess_input(y_pred, y_true, background):
    """
    校验输入并确定起始类别。
    注意：不再基于 background 对像素进行裁剪，避免忽略背景区域的假阳性（FP）。
    当 background=False 时，仅排除背景类(0)参与类别平均；像素层面仍在整幅图上统计。
    """
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise ValueError("Both y_pred and y_true must be numpy arrays.")
    if y_pred.shape != y_true.shape:
        print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        raise ValueError("Shape mismatch between y_pred and y_true.")
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    # 当不计背景时：从类别1开始；否则从0开始
    start_class = 0 if background else 1
    return y_pred, y_true, start_class


def dice_coefficient(y_pred, y_true, num_classes, background=False):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
        
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, background)
    
    if y_pred.size == 0:
        return 0.0
        
    dice = 0.0
    valid_classes = 0
    
    for cls in range(start_class, num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)
        intersection = np.sum(pred_cls * true_cls)
        union = np.sum(pred_cls) + np.sum(true_cls)
        
        if union == 0:
            # 如果该类在预测和真实标签中都不存在，认为预测完全正确
            dice_cls = 1.0
        else:
            dice_cls = (2.0 * intersection) / union
            
        dice += dice_cls
        valid_classes += 1
    
    return dice / valid_classes if valid_classes > 0 else 0.0


def iou_score(y_pred, y_true, num_classes, background=False):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
        
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, background)
    
    if y_pred.size == 0:
        return 0.0
        
    iou = 0.0
    valid_classes = 0
    
    for cls in range(start_class, num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)
        intersection = np.sum(pred_cls * true_cls)
        union = np.sum(pred_cls) + np.sum(true_cls) - intersection
        
        if union == 0:
            # 如果该类在预测和真实标签中都不存在，认为预测完全正确
            iou_cls = 1.0
        else:
            iou_cls = intersection / union
            
        iou += iou_cls
        valid_classes += 1
    
    return iou / valid_classes if valid_classes > 0 else 0.0


def pixel_error(y_pred, y_true, num_classes, background=False):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    # 对像素级精度：
    # - 当 background=False 时，优先只在前景像素(y_true!=0)上计算；
    #   若样本中不存在前景，则回退为全图计算（避免空数组导致不稳定）。
    # - 当 background=True 时，直接在全图上计算。
    y_pred, y_true, _ = preprocess_input(y_pred, y_true, background)

    if not background:
        mask = (y_true != 0)
        if np.any(mask):
            y_pred = y_pred[mask]
            y_true = y_true[mask]

    if y_pred.size == 0:
        return 1.0  # 空样本时返回最大错误率（保守处理）

    return 1.0 - np.mean(y_pred == y_true)



if __name__ == "__main__":
    # Minimal test cases to verify metrics behavior
    # Classes: 0(background), 1, 2, 3
    num_classes = 4

    def show(title, y_pred, y_true):
        print(f"\n=== {title} ===")
        print("y_true:\n", y_true)
        print("y_pred:\n", y_pred)
        for bg in (True, False):
            d = dice_coefficient(y_pred, y_true, num_classes, background=bg)
            i = iou_score(y_pred, y_true, num_classes, background=bg)
            pe = pixel_error(y_pred, y_true, num_classes, background=bg)
            print(f"background={'True' if bg else 'False'} -> Dice: {d:.4f}, IoU: {i:.4f}, PixelErr: {pe:.4f}")

    # Case 1: Perfect match (1-class foreground)
    y_true_1 = np.zeros((6, 6), dtype=np.int32)
    y_true_1[2:4, 2:4] = 1
    y_pred_1 = y_true_1.copy()
    show("Case1: perfect match", y_pred_1, y_true_1)

    # Case 2: Over-segmentation (predict larger region than GT)
    y_true_2 = np.zeros((6, 6), dtype=np.int32)
    y_true_2[2:4, 2:4] = 1
    y_pred_2 = np.zeros((6, 6), dtype=np.int32)
    y_pred_2[1:5, 1:5] = 1  # larger area -> should lower Dice/IoU when background=False
    show("Case2: over-segmentation", y_pred_2, y_true_2)

    # Case 3: Under-segmentation (predict smaller region than GT)
    y_true_3 = np.zeros((6, 6), dtype=np.int32)
    y_true_3[1:5, 1:5] = 1
    y_pred_3 = np.zeros((6, 6), dtype=np.int32)
    y_pred_3[2:4, 2:4] = 1
    show("Case3: under-segmentation", y_pred_3, y_true_3)

    # Case 4: Class confusion (predict class 2 where GT is class 1)
    y_true_4 = np.zeros((6, 6), dtype=np.int32)
    y_true_4[2:4, 2:4] = 1
    y_pred_4 = np.zeros((6, 6), dtype=np.int32)
    y_pred_4[2:4, 2:4] = 2
    show("Case4: class confusion", y_pred_4, y_true_4)
