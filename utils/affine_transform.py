import random
import torch
import torch.nn as nn
from config import Config



# 随机生成仿射变换参数
def random_affine_params(scale_range=(0.2, 5.0), shift_range=(-20, 20)):
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