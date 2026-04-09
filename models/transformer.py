from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Transformer2DConfig:
    patch_size: int = 4
    embed_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0


def _pad_to_multiple(
    x: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims (H,W) so they are divisible by `multiple`."""
    if multiple <= 1:
        return x, (0, 0)

    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)

    x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, (pad_h, pad_w)


def _crop_pad(x: torch.Tensor, pad_hw: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = int(pad_hw[0]), int(pad_hw[1])
    if pad_h == 0 and pad_w == 0:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    return x[..., : h - pad_h, : w - pad_w]


class Transformer2D(nn.Module):
    """Lightweight ViT-style encoder + upsample head for dense prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: Transformer2DConfig | None = None,
    ):
        super().__init__()
        cfg = cfg or Transformer2DConfig()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_size = int(cfg.patch_size)
        self.embed_dim = int(cfg.embed_dim)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

        # ---- FIXED positional embedding (learnable 2D base + interpolate) ----
        # 你可以根据训练/推理中最大分辨率调整这个值
        max_hw = 16  # e.g. 256x256 with patch_size=16 → 16x16
        self.pos_embed_base = nn.Parameter(
            torch.zeros(1, self.embed_dim, max_hw, max_hw)
        )
        nn.init.trunc_normal_(self.pos_embed_base, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(cfg.num_heads),
            dim_feedforward=int(self.embed_dim * float(cfg.mlp_ratio)),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(cfg.depth),
            norm=nn.LayerNorm(self.embed_dim),  # pre-norm架构需要最终LayerNorm
        )

        self.pre_ln = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(cfg.dropout))

        self.head = nn.Conv2d(
            self.embed_dim, self.out_channels, kernel_size=1, bias=True
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        # ViT-style init for transformer encoder layers
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_pos_embed(
        self,
        hp: int,
        wp: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns positional embedding of shape [1, Hp*Wp, E]
        """
        pe = F.interpolate(
            self.pos_embed_base,
            size=(hp, wp),
            mode="bilinear",
            align_corners=False,
        )
        pe = pe.flatten(2).transpose(1, 2)
        return pe.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

        # Pad input
        x_pad, pad_hw = _pad_to_multiple(x, self.patch_size)
        b, _, h_pad, w_pad = x_pad.shape

        # Patch embedding
        feat = self.patch_embed(x_pad)  # [B,E,Hp,Wp]
        hp, wp = int(feat.shape[-2]), int(feat.shape[-1])

        # Tokens
        tokens = feat.flatten(2).transpose(1, 2)  # [B,N,E]
        tokens = self.pre_ln(tokens)

        pos = self._get_pos_embed(hp, wp, tokens.device, tokens.dtype)
        tokens = tokens + pos
        tokens = self.dropout(tokens)

        # Transformer
        tokens = self.encoder(tokens)

        # Back to feature map
        feat2 = tokens.transpose(1, 2).reshape(b, self.embed_dim, hp, wp)
        out_low = self.head(feat2)

        # Upsample + crop
        out = F.interpolate(
            out_low,
            size=(h_pad, w_pad),
            mode="bilinear",
            align_corners=False,
        )
        out = _crop_pad(out, pad_hw)
        return out


# Alias for drop-in replacement
TransformerNet = Transformer2D
