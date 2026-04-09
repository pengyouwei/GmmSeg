import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShapeVAE(nn.Module):
    """
    Lightweight shape-VAE for SDF reconstruction.

    Input:
        x: [B, C, H, W]
    Output:
        recon: [B, C, H, W]
        mu: [B, latent_dim]
        logvar: [B, latent_dim]
        z: [B, latent_dim]
    """

    def __init__(self, in_channels: int, latent_dim: int = 16, base_channels: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)

        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 4)

        self.enc1 = ConvBlock(self.in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_mu = nn.Linear(c3, self.latent_dim)
        self.fc_logvar = nn.Linear(c3, self.latent_dim)

        self.fc_decode = nn.Linear(self.latent_dim, c3)
        self.dec3 = ConvBlock(c3, c2)
        self.dec2 = ConvBlock(c2, c1)
        self.dec1 = ConvBlock(c1, c1)
        self.out_conv = nn.Conv2d(c1, self.in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.enc1(x)
        h2 = self.enc2(self.pool(h1))
        h3 = self.enc3(self.pool(h2))

        h = self.global_pool(h3).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        h = self.fc_decode(z).unsqueeze(-1).unsqueeze(-1)

        h = F.interpolate(h, size=(max(1, output_hw[0] // 4), max(1, output_hw[1] // 4)), mode="bilinear", align_corners=False)
        h = self.dec3(h)

        h = F.interpolate(h, size=(max(1, output_hw[0] // 2), max(1, output_hw[1] // 2)), mode="bilinear", align_corners=False)
        h = self.dec2(h)

        h = F.interpolate(h, size=output_hw, mode="bilinear", align_corners=False)
        h = self.dec1(h)

        return self.out_conv(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, (h, w))
        return recon, mu, logvar, z
