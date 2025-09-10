import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
from PIL import Image


def get_lv_centroid_from_prior(prior_array, lv_channel_idx=1):
    """
    从先验数组中计算左心室的质心坐标
    Args:
        prior_array: numpy array of prior, shape [4, H, W]
        lv_channel_idx: 左心室对应的通道索引，默认为1
    Returns:
        center_y, center_x: 左心室质心的坐标
    """
    # 获取左心室通道
    lv_prior = prior_array[lv_channel_idx]
    
    # 找到左心室的主要区域（使用阈值来确定主要区域）
    threshold = np.max(lv_prior) * 0.5
    lv_mask = (lv_prior >= threshold)
    # lv_mask = (lv_prior == 1.0)
    
    if not np.any(lv_mask):
        # 如果没有找到左心室，返回图像中心
        h, w = lv_prior.shape
        return h // 2, w // 2
    
    # 计算质心
    y_coords, x_coords = np.where(lv_mask)
    center_y = round(np.mean(y_coords))
    center_x = round(np.mean(x_coords))

    return center_y, center_x


def center_crop_prior_4chs(prior_array, target_size, lv_channel_idx=1):
    """
    基于左心室中心对先验进行裁剪
    Args:
        prior_array: numpy array, shape [4, H, W]
        target_size: int, 目标裁剪尺寸
        lv_channel_idx: 左心室对应的通道索引，默认为1
    Returns:
        cropped_prior: 裁剪后的先验，shape [4, target_size, target_size]
    """
    # 获取左心室质心
    center_y, center_x = get_lv_centroid_from_prior(prior_array, lv_channel_idx)
    
    C, H, W = prior_array.shape
    half_size = target_size // 2
    
    # 计算裁剪区域，确保不超出图像边界
    start_y = max(0, center_y - half_size)
    end_y = min(H, center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(W, center_x + half_size)
    
    # 如果裁剪区域不够大，需要调整
    if end_y - start_y < target_size:
        if start_y == 0:
            end_y = min(H, start_y + target_size)
        else:
            start_y = max(0, end_y - target_size)
    
    if end_x - start_x < target_size:
        if start_x == 0:
            end_x = min(W, start_x + target_size)
        else:
            start_x = max(0, end_x - target_size)
    
    # 裁剪
    cropped_prior = prior_array[:, start_y:end_y, start_x:end_x]
    
    # 如果裁剪后的尺寸仍然不够，进行填充
    if cropped_prior.shape[1] < target_size or cropped_prior.shape[2] < target_size:
        pad_h = max(0, target_size - cropped_prior.shape[1])
        pad_w = max(0, target_size - cropped_prior.shape[2])
        
        cropped_prior = np.pad(cropped_prior, 
                              ((0, 0),  # 不对通道维度进行填充
                               (pad_h//2, pad_h - pad_h//2), 
                               (pad_w//2, pad_w - pad_w//2)), 
                              mode='constant', constant_values=0)
    
    return cropped_prior


def get_lv_centroid(label_array):
    """
    计算左心室(像素值为1)的质心坐标
    Args:
        label_array: numpy array of label
    Returns:
        center_y, center_x: 左心室质心的坐标
    """
    # 找到左心室的像素位置
    lv_mask = (label_array == 1)
    
    if not np.any(lv_mask):
        # 如果没有找到左心室，返回图像中心
        h, w = label_array.shape
        return h // 2, w // 2
    
    # 计算质心
    y_coords, x_coords = np.where(lv_mask)
    center_y = round(np.mean(y_coords))
    center_x = round(np.mean(x_coords))

    return center_y, center_x


def center_crop_image_label(image, label, target_size):
    """
    基于左心室中心进行裁剪
    Args:
        image: PIL Image
        label: PIL Image 
        target_size: int, 目标裁剪尺寸
    Returns:
        cropped_image, cropped_label: 裁剪后的图像和标签
    """
    # 将PIL图像转换为numpy数组以计算质心
    label_array = np.array(label)
    image_array = np.array(image)
    
    # 获取左心室质心
    center_y, center_x = get_lv_centroid(label_array)
    
    h, w = label_array.shape
    half_size = target_size // 2
    
    # 计算裁剪区域，确保不超出图像边界
    start_y = max(0, center_y - half_size)
    end_y = min(h, center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(w, center_x + half_size)
    
    # 如果裁剪区域不够大，需要调整
    if end_y - start_y < target_size:
        if start_y == 0:
            end_y = min(h, start_y + target_size)
        else:
            start_y = max(0, end_y - target_size)
    
    if end_x - start_x < target_size:
        if start_x == 0:
            end_x = min(w, start_x + target_size)
        else:
            start_x = max(0, end_x - target_size)
    
    # 裁剪
    cropped_image_array = image_array[start_y:end_y, start_x:end_x]
    cropped_label_array = label_array[start_y:end_y, start_x:end_x]
    
    # 如果裁剪后的尺寸仍然不够，进行填充
    if cropped_image_array.shape[0] < target_size or cropped_image_array.shape[1] < target_size:
        pad_h = max(0, target_size - cropped_image_array.shape[0])
        pad_w = max(0, target_size - cropped_image_array.shape[1])
        
        cropped_image_array = np.pad(cropped_image_array, 
                                   ((pad_h//2, pad_h - pad_h//2), 
                                    (pad_w//2, pad_w - pad_w//2)), 
                                   mode='constant', constant_values=0)
        cropped_label_array = np.pad(cropped_label_array, 
                                   ((pad_h//2, pad_h - pad_h//2), 
                                    (pad_w//2, pad_w - pad_w//2)), 
                                   mode='constant', constant_values=0)
    
    # 转换回PIL图像
    cropped_image = Image.fromarray(cropped_image_array.astype(np.uint8))
    cropped_label = Image.fromarray(cropped_label_array.astype(np.uint8))
    
    return cropped_image, cropped_label


class ConditionalResize(transforms.Resize):
    def __init__(self, size, img_size):
        super().__init__(size)
        self.img_size = img_size
        
    def __call__(self, img):
        if img.size == (self.img_size, self.img_size):
            return img
        return super().__call__(img)

def get_image_transform(img_size):
    return transforms.Compose([
        ConditionalResize((img_size, img_size), img_size),
        transforms.CenterCrop((img_size, img_size)), 
        transforms.ToTensor(),  # 转为张量，范围 [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到 [-1, 1]
    ])

def get_label_transform(img_size):
    return transforms.Compose([
        ConditionalResize((img_size, img_size), img_size),
        transforms.CenterCrop((img_size, img_size)),
        transforms.PILToTensor()
    ])

# 随机生成仿射变换参数
def random_affine_params(scale_range=(0.4, 2.5), shift_range=(-20, 20)):
    scale = random.uniform(*scale_range)
    tx = random.uniform(*shift_range)
    ty = random.uniform(*shift_range)
    return scale, tx, ty

# 可微分刚性变换
def apply_affine_transform(img, scale, tx, ty, mode='bilinear', padding_mode='border'):

    img = img.float()  # 确保输入是浮点数类型

    B, C, H, W = img.shape  # batch_size, channels, height, width
    theta = torch.zeros(B, 2, 3).to(img.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx / (W / 2)  # 归一化平移
    theta[:, 1, 2] = ty / (H / 2)

    grid = nn.functional.affine_grid(theta, img.size(), align_corners=False)
    transformed_img = nn.functional.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    
    return transformed_img


# 随机生成旋转角度（单位：弧度）
def random_rotate_params(angle_range=(-30, 30)):
    angle = random.uniform(*angle_range)  # 单位：度
    rad = math.radians(angle)             # 转成弧度
    return rad


def apply_rotate_transform(img, angle, mode='bilinear', padding_mode='border'):
    """对 batch 图像施加旋转 (可微)。
    支持:
        angle: float(标量弧度) | torch.Tensor 形状 [B] 或 [B,1]
    要求: angle 为弧度。
    """
    img = img.float()  # 确保浮点数类型
    B, C, H, W = img.shape

    # 处理角度到 [B]
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, dtype=img.dtype, device=img.device).repeat(B)
    else:
        angle = angle.to(img.device, dtype=img.dtype)
        if angle.dim() == 2 and angle.size(1) == 1:
            angle = angle.squeeze(1)
        if angle.dim() == 0:
            angle = angle.repeat(B)
        assert angle.shape[0] == B, f"angle batch size {angle.shape[0]} != image batch size {B}"

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    theta = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    grid = F.affine_grid(theta, img.size(), align_corners=False)
    rotated_img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    return rotated_img