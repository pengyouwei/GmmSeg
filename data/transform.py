import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
from PIL import Image


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
    基于左心室中心进行裁剪；若裁剪区域越界则先填充，再围绕中心裁剪，
    确保左心室中心出现在输出的几何中心。
    Args:
        image: PIL Image
        label: PIL Image 
        target_size: int, 目标裁剪尺寸(正方形)
    Returns:
        cropped_image, cropped_label: 裁剪后的图像和标签
    """
    # 转为 numpy
    label_array = np.array(label)
    image_array = np.array(image)

    # 左心室质心（若不存在返回图像中心）
    center_y, center_x = get_lv_centroid(label_array)

    H, W = label_array.shape
    half = target_size // 2

    # 以质心为几何中心的窗口
    y0 = center_y - half
    x0 = center_x - half
    y1 = y0 + target_size
    x1 = x0 + target_size

    # 计算需要的四向填充量（不足即为正的 pad）
    pad_top    = max(0, -y0)
    pad_left   = max(0, -x0)
    pad_bottom = max(0,  y1 - H)
    pad_right  = max(0,  x1 - W)

    if pad_top or pad_left or pad_bottom or pad_right:
        # 图像采用反射填充；若尺寸过小或 pad 过大导致 reflect 不可行，则降级为 edge(=replicate)
        eff_mode = 'reflect'
        if (H < 2 or W < 2 or
            pad_top >= H or pad_bottom >= H or
            pad_left >= W or pad_right >= W):
            eff_mode = 'edge'

        if eff_mode == 'reflect':
            image_array = np.pad(image_array,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='reflect')
        elif eff_mode == 'edge':
            image_array = np.pad(image_array,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='edge')
        else:  # 兜底：常数0
            image_array = np.pad(image_array,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='constant', constant_values=0)

        # 标签恒为背景常数填充
        label_array = np.pad(label_array,
                             ((pad_top, pad_bottom), (pad_left, pad_right)),
                             mode='constant', constant_values=0)

        # 更新中心坐标到新坐标系
        center_y += pad_top
        center_x += pad_left
        H += pad_top + pad_bottom
        W += pad_left + pad_right

    # 现在可以安全裁剪，且确保中心在中点
    start_y = center_y - half
    start_x = center_x - half
    end_y = start_y + target_size
    end_x = start_x + target_size

    cropped_image_array = image_array[start_y:end_y, start_x:end_x]
    cropped_label_array = label_array[start_y:end_y, start_x:end_x]

    # 转回 PIL
    cropped_image = Image.fromarray(cropped_image_array.astype(np.uint8))
    cropped_label = Image.fromarray(cropped_label_array.astype(np.uint8))
    return cropped_image, cropped_label


from torchvision.transforms import InterpolationMode
class ConditionalResize(transforms.Resize):
    def __init__(self, size, img_size, interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, interpolation=interpolation)
        self.img_size = img_size
        
    def __call__(self, img):
        if img.size == (self.img_size, self.img_size):
            return img
        return super().__call__(img)


def get_image_transform(img_size):
    return transforms.Compose([
        ConditionalResize((img_size, img_size), img_size, interpolation=InterpolationMode.BILINEAR),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_label_transform(img_size):
    return transforms.Compose([
        ConditionalResize((img_size, img_size), img_size, interpolation=InterpolationMode.NEAREST),  
        transforms.PILToTensor()
    ])

# 随机生成仿射变换参数
def random_affine_params(scale_range=(0.5, 2.0), shift_range=(-20, 20)):
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
def random_rotate_params(angle_range=(-60, 60)):
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