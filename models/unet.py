import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    '''双卷积'''
 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)


class InConv(nn.Module):
    '''输入层'''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(InConv, self).__init__()
        self.in_conv = DoubleConv(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        return self.in_conv(x)


class DownSampling(nn.Module):
    '''下采样'''
 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DownSampling, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, kernel_size, padding)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
 
 
class UpSampling(nn.Module):
    '''上采样'''
 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size, padding)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 调整x1的尺寸，使其与x2的尺寸相同
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=True)
 
        x = torch.cat((x2, x1), dim=1)
        return self.double_conv(x)
 
 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
    

class UNetEncoder(nn.Module):
    '''Encoder part of U-Net'''

    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(UNetEncoder, self).__init__()
        self.in_conv = InConv(in_channels, 64, kernel_size, padding)
        self.down1 = DownSampling(64, 128, kernel_size, padding)
        self.down2 = DownSampling(128, 256, kernel_size, padding)
        self.down3 = DownSampling(256, 512, kernel_size, padding)
        self.down4 = DownSampling(512, 1024, kernel_size, padding)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]



class UNetDecoder(nn.Module):
    '''Decoder part of U-Net'''

    def __init__(self, out_channels, kernel_size=3, padding=1):
        super(UNetDecoder, self).__init__()
        self.up1 = UpSampling(1024, 512, kernel_size, padding)
        self.up2 = UpSampling(512, 256, kernel_size, padding)
        self.up3 = UpSampling(256, 128, kernel_size, padding)
        self.up4 = UpSampling(128, 64, kernel_size, padding)
        self.out_conv = OutConv(64, out_channels)

    def forward(self, encoder_outputs):
        x1, x2, x3, x4, x5 = encoder_outputs
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.out_conv(x9)
        return [x6, x7, x8, x9, output]



class UNet(nn.Module):
    '''U-Net'''
    
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        output = self.decoder(encoder_outputs)
        return output[-1]

#####################################################################################################################


class UNetEncoder_ResNet(nn.Module):
    '''Encoder part of U-Net using pre-trained ResNet'''
    def __init__(self, in_channels):
        super(UNetEncoder_ResNet, self).__init__()
        # 加载预训练的 ResNet 模型
        resnet = models.resnet18(weights='DEFAULT')

        # 调整第一层卷积层
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 提取 ResNet 不同阶段的层
        self.in_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.down1 = resnet.layer1
        self.down2 = resnet.layer2
        self.down3 = resnet.layer3
        self.down4 = resnet.layer4

    def forward(self, x):
        x1 = self.in_conv(x)
        x = self.maxpool(x1)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class UpSampling_ResNet(nn.Module):
    '''上采样'''

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, padding=1):
        super(UpSampling_ResNet, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 调整 DoubleConv 的输入通道数为拼接后的通道数
        self.double_conv = DoubleConv(skip_channels + out_channels, out_channels, kernel_size, padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat((x2, x1), dim=1)
        return self.double_conv(x)


class UNetDecoder_ResNet(nn.Module):
    '''Decoder part of U-Net'''

    def __init__(self, out_channels, kernel_size=3, padding=1):
        super(UNetDecoder_ResNet, self).__init__()
        self.up1 = UpSampling_ResNet(512, 256, 256, kernel_size, padding)
        self.up2 = UpSampling_ResNet(256, 128, 128, kernel_size, padding)
        self.up3 = UpSampling_ResNet(128, 64, 64, kernel_size, padding)
        self.up4 = UpSampling_ResNet(64, 64, 64, kernel_size, padding)
        self.out_conv = OutConv(64, out_channels)

    def forward(self, encoder_outputs):
        x1, x2, x3, x4, x5 = encoder_outputs
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.out_conv(x9)
        return [x6, x7, x8, x9, output]



class UNet_ResNet(nn.Module):
    '''U-Net with ResNet as encoder'''

    def __init__(self, in_channels, out_channels):
        super(UNet_ResNet, self).__init__()
        self.encoder = UNetEncoder_ResNet(in_channels)
        self.decoder = UNetDecoder_ResNet(out_channels)

        
    def forward(self, x):
        encoder_outputs = self.encoder(x)
        output = self.decoder(encoder_outputs)
        return output[-1]



if __name__ == '__main__':
    img_tensor = torch.rand((1, 1, 256, 256))

    encoder = UNetEncoder(1)
    encoder_outputs = encoder(img_tensor)
    for i, out in enumerate(encoder_outputs):
        print(f"x{i+1}", out.shape[1:])
    
    decoder = UNetDecoder(1)
    decoder_outputs = decoder(encoder_outputs)
    for i, out in enumerate(decoder_outputs):
        print(f"x{i+6}", out.shape[1:])

    unet = UNet(1, 1)
    unet_outputs = unet(img_tensor)
    print("unet", unet_outputs.shape[1:])

    print("--------------")


    # encoder_resnet = UNetEncoder_ResNet(1)
    # encoder_outputs_resnet = encoder_resnet(img_tensor)
    # for i, out in enumerate(encoder_outputs_resnet):
    #     print(f"x{i+1}", out.shape[1:])

    # decoder_resnet = UNetDecoder_ResNet(1)
    # decoder_outputs_resnet = decoder_resnet(encoder_outputs_resnet)
    # for i, out in enumerate(decoder_outputs_resnet):
    #     print(f"x{i+6}", out.shape[1:])
    
    # unet_resnet = UNet_ResNet(1, 1)
    # unet_resnet_outputs = unet_resnet(img_tensor)
    # print("unet_resnet", unet_resnet_outputs.shape[1:])


