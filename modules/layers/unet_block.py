from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import LayerNorm, MultiheadAttention

from ..utils import weight_init


class UpSample(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, with_conv: bool = True
    ) -> None:
        super(UpSample, self).__init__()
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        if with_conv:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, with_conv: bool = True
    ) -> None:
        super(DownSample, self).__init__()
        self.with_conv = with_conv
        self.down = nn.MaxPool1d(kernel_size=2, stride=2)
        if with_conv:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = self.conv(nn.functional.pad(x, (1, 1), mode="constant", value=0))
        else:
            x = self.down(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        emb_channels: int = 512,
    ) -> None:
        super(ResnetBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.emb_proj = nn.Linear(emb_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = nn.functional.silu(self.norm1(x))
        x = self.conv1(x)
        x += self.emb_proj(nn.functional.silu(emb))[:, :, None]
        x = nn.functional.silu(self.norm2(x))
        x = self.conv2(self.dropout(x))
        if hasattr(self, "shortcut"):
            residual = self.shortcut(residual)
        return x + residual


class AttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        norm_first: bool = False,
    ) -> None:
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm_first = norm_first

        self.attn = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_norm = LayerNorm(hidden_dim)
        self.ff_norm = LayerNorm(hidden_dim)

        self.apply(weight_init)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # do attention on channels
        q = x
        k, v = (x, x) if kv is None else (kv, kv)
        if self.norm_first:
            x = x + self.attn_norm(
                self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
            )
            x = x + self.ff_norm(self.ff_mlp(x))
        else:
            x = self.attn_norm(
                x + self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
            )
            x = self.ff_norm(x + self.ff_mlp(x))
        return x.permute(0, 2, 1)
