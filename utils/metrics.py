import numpy as np


def preprocess_input(y_pred, y_true, background):
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise ValueError("Both y_pred and y_true must be numpy arrays.")
    if y_pred.shape != y_true.shape:
        print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        raise ValueError("Shape mismatch between y_pred and y_true.")
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")
        
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
        
    y_pred, y_true, _ = preprocess_input(y_pred, y_true, background)
    
    if y_pred.size == 0:
        return 1.0  # 如果没有像素，返回最大错误率
        
    return 1.0 - np.mean(y_pred == y_true)



if __name__ == "__main__":
    pass
