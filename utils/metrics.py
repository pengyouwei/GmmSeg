import numpy as np

def convert_labels(labels, dataset_name):
    if not isinstance(labels, np.ndarray):
        raise ValueError("Input labels must be a numpy array.")
    if dataset_name == "ACDC_aligned":
        new_labels = np.zeros_like(labels)
        new_labels[labels == 1] = 1  # RV
        new_labels[labels == 2] = 2  # MYO
        new_labels[labels == 3] = 3  # LV
    return new_labels

def preprocess_input(y_pred, y_true, dataset_name, background):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise ValueError("Both y_pred and y_true must be numpy arrays.")
    if y_pred.shape != y_true.shape:
        print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        raise ValueError("Shape mismatch between y_pred and y_true.")
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")
        
    y_true = convert_labels(y_true, dataset_name)
    start_class = 0
    if not background:
        mask = y_true != 0
        if not np.any(mask):
            # 如果没有前景像素，返回原始数据但设置适当的start_class
            print("Warning: No foreground pixels found in y_true")
            return y_pred, y_true, 1
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        start_class = 1
    return y_pred, y_true, start_class


def dice_coefficient(y_pred, y_true, num_classes, dataset_name, background=False):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
        
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, dataset_name, background)
    
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


def iou_score(y_pred, y_true, num_classes, dataset_name, background=False):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
        
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, dataset_name, background)
    
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


def pixel_error(y_pred, y_true, num_classes, dataset_name, background=False):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
        
    y_pred, y_true, _ = preprocess_input(y_pred, y_true, dataset_name, background)
    
    if y_pred.size == 0:
        return 1.0  # 如果没有像素，返回最大错误率
        
    return 1.0 - np.mean(y_pred == y_true)


def detailed_metrics(y_pred, y_true, num_classes, dataset_name, background=False):
    """
    计算每个类别的详细指标
    返回字典格式的结果，包含每个类别的Dice、IoU等指标
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
        
    y_pred, y_true, start_class = preprocess_input(y_pred, y_true, dataset_name, background)
    
    if y_pred.size == 0:
        return {"dice": {}, "iou": {}, "pixel_accuracy": 0.0}
    
    class_names = {0: "Background", 1: "RV", 2: "MYO", 3: "LV"} if dataset_name == "ACDC" else {}
    
    results = {
        "dice": {},
        "iou": {},
        "pixel_accuracy": 1.0 - np.mean(y_pred == y_true)
    }
    
    for cls in range(start_class, num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)
        intersection = np.sum(pred_cls * true_cls)
        
        # Dice coefficient
        union_dice = np.sum(pred_cls) + np.sum(true_cls)
        if union_dice == 0:
            dice_cls = 1.0
        else:
            dice_cls = (2.0 * intersection) / union_dice
            
        # IoU
        union_iou = np.sum(pred_cls) + np.sum(true_cls) - intersection
        if union_iou == 0:
            iou_cls = 1.0
        else:
            iou_cls = intersection / union_iou
        
        class_name = class_names.get(cls, f"Class_{cls}")
        results["dice"][class_name] = dice_cls
        results["iou"][class_name] = iou_cls
    
    return results


if __name__ == "__main__":
    # 测试用例1: 基本功能测试
    print("=== 基本功能测试 ===")
    y_true = np.array([0, 85, 170, 255, 0])  # ACDC标签格式
    y_pred = np.array([0, 1, 2, 3, 0])       # 预测结果（类别索引）
    dataset_name = "ACDC"
    num_classes = 4

    print("y_true (原始):", y_true)
    print("y_pred:", y_pred)
    print("y_true (转换后):", convert_labels(y_true, dataset_name))

    dice = dice_coefficient(y_pred, y_true, num_classes, dataset_name, background=False)
    iou = iou_score(y_pred, y_true, num_classes, dataset_name, background=False)
    pixel_err = pixel_error(y_pred, y_true, num_classes, dataset_name, background=False)
    
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Pixel Error: {pixel_err:.4f}")

    # 测试用例2: 详细指标
    print("\n=== 详细指标测试 ===")
    detailed = detailed_metrics(y_pred, y_true, num_classes, dataset_name, background=False)
    print("详细指标:")
    for metric, values in detailed.items():
        if isinstance(values, dict):
            print(f"  {metric.upper()}:")
            for class_name, value in values.items():
                print(f"    {class_name}: {value:.4f}")
        else:
            print(f"  {metric}: {values:.4f}")

    # 测试用例3: 边界条件测试
    print("\n=== 边界条件测试 ===")
    try:
        # 空数组测试
        empty_pred = np.array([])
        empty_true = np.array([])
        print("空数组测试: 预期抛出异常")
        dice_coefficient(empty_pred, empty_true, num_classes, dataset_name)
    except ValueError as e:
        print(f"✓ 正确捕获异常: {e}")

    # 测试用例4: 完美预测
    print("\n=== 完美预测测试 ===")
    perfect_true = np.array([0, 85, 170, 255])
    perfect_pred = np.array([0, 1, 2, 3])
    
    perfect_dice = dice_coefficient(perfect_pred, perfect_true, num_classes, dataset_name, background=True)
    perfect_iou = iou_score(perfect_pred, perfect_true, num_classes, dataset_name, background=True)
    perfect_pixel_err = pixel_error(perfect_pred, perfect_true, num_classes, dataset_name, background=True)
    
    print(f"完美预测 - Dice: {perfect_dice:.4f}, IoU: {perfect_iou:.4f}, Pixel Error: {perfect_pixel_err:.4f}")

