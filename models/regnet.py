import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config


class RR_ResNet(nn.Module):
    def __init__(self, input_channels=8, scale_range=Config().SCALE_RANGE, shift_range=Config().SHIFT_RANGE):
        super(RR_ResNet, self).__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 3)  # 让最终输出变成 3 个刚性变换参数

    def forward(self, x):
        x = self.resnet(x)
        # 这里的 sigmoid 函数将输出限制在 [0, 1] 范围内，然后通过线性变换映射到指定范围
        scale = torch.sigmoid(x[:, 0]) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        tx = torch.sigmoid(x[:, 1]) * (self.shift_range[1] - self.shift_range[0]) + self.shift_range[0]
        ty = torch.sigmoid(x[:, 2]) * (self.shift_range[1] - self.shift_range[0]) + self.shift_range[0]
        return torch.stack([scale, tx, ty], dim=1)  # 返回 [B, 3] 的张量