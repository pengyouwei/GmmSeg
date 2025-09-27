import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure

def remove_small_components(pred_mask, min_size=50):
    """移除小连通域"""
    pred_clean = pred_mask.copy()
    
    for class_id in np.unique(pred_mask):
        if class_id == 0:  # 跳过背景
            continue
            
        # 获取当前类别的mask
        class_mask = (pred_mask == class_id).astype(np.uint8)
        
        # 连通域分析
        labeled = measure.label(class_mask)
        props = measure.regionprops(labeled)
        
        # 保留最大的连通域或面积超过阈值的连通域
        for prop in props:
            if prop.area < min_size:
                # 移除小连通域
                coords = prop.coords
                pred_clean[coords[:, 0], coords[:, 1]] = 0
    
    return pred_clean

def morphological_cleanup(pred_mask, kernel_size=3):
    """形态学操作清理"""
    pred_clean = pred_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    for class_id in np.unique(pred_mask):
        if class_id == 0:
            continue
            
        class_mask = (pred_mask == class_id).astype(np.uint8)
        
        # 开运算：先腐蚀后膨胀，移除小噪声
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        
        # 闭运算：先膨胀后腐蚀，填充小孔洞
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        
        pred_clean[pred_mask == class_id] = 0
        pred_clean[class_mask == 1] = class_id
    
    return pred_clean

def anatomical_constraint_cleanup(pred_mask):
    """基于解剖学约束的清理"""
    pred_clean = pred_mask.copy()
    
    # 对于心脏分割，可以基于以下约束：
    # 1. LV应该在心肌内部
    # 2. 心肌应该是环形结构
    # 3. 各结构应该相邻
    
    lv_mask = (pred_mask == 1).astype(np.uint8)
    myo_mask = (pred_mask == 2).astype(np.uint8)
    rv_mask = (pred_mask == 3).astype(np.uint8)
    
    if np.any(lv_mask):
        # 找到LV的质心
        lv_props = measure.regionprops(measure.label(lv_mask))
        if lv_props:
            lv_centroid = lv_props[0].centroid
            
            # 移除距离LV质心过远的心肌区域
            if np.any(myo_mask):
                myo_labeled = measure.label(myo_mask)
                myo_props = measure.regionprops(myo_labeled)
                
                for prop in myo_props:
                    # 计算到LV质心的距离
                    dist = np.sqrt((prop.centroid[0] - lv_centroid[0])**2 + 
                                 (prop.centroid[1] - lv_centroid[1])**2)
                    
                    # 如果距离过远，移除该连通域
                    if dist > 100:  # 可调整阈值
                        coords = prop.coords
                        pred_clean[coords[:, 0], coords[:, 1]] = 0
    
    return pred_clean

def comprehensive_postprocess(pred_mask, min_size=30, kernel_size=3):
    """综合后处理流程"""
    # 1. 移除小连通域
    pred_clean = remove_small_components(pred_mask, min_size)
    
    # 2. 形态学清理
    pred_clean = morphological_cleanup(pred_clean, kernel_size)
    
    # 3. 解剖学约束清理
    pred_clean = anatomical_constraint_cleanup(pred_clean)
    
    # 4. 再次移除小连通域（经过形态学操作后可能产生新的小块）
    pred_clean = remove_small_components(pred_clean, min_size//2)
    
    return pred_clean