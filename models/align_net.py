import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from config import Config


class Align_ResNet(nn.Module):
    def __init__(self, input_channels=1, rotate_range=Config().ROTATE_RANGE):
        super(Align_ResNet, self).__init__()
        self.rotate_range = rotate_range
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 1)  # 让最终输出变成 1 个刚性变换参数

    def forward(self, x):
        x = self.resnet(x)
        # 预测角度(单位: 弧度)。 rotate_range 为度，需要转换。
        rotate_deg = self.rotate_range[0] + (self.rotate_range[1] - self.rotate_range[0]) * torch.sigmoid(x[:, 0])
        rotate_rad = rotate_deg * math.pi / 180.0
        return rotate_rad.unsqueeze(1)  # [B, 1]
