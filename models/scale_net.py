import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from config import Config


class Scale_ResNet(nn.Module):
    def __init__(self, input_channels=4, scale_range: tuple = None):
        super(Scale_ResNet, self).__init__()
        # 正确的缩放范围配置（默认为全局 Config 的范围）
        if scale_range is None:
            scale_range = Config().IMAGE_SCALE_RANGE
        self.scale_range = scale_range

        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 1)  # 让最终输出变成 1 个刚性变换参数

        
    def forward(self, x):
        x = self.resnet(x)
        # 将输出限制到指定尺度范围
        scale = torch.sigmoid(x[:, 0]) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        return scale.unsqueeze(1)  # [B, 1]