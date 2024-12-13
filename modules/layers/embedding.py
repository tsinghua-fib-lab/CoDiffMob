import math
from typing import List, Optional

import torch
import torch.nn as nn

from ..utils import weight_init


class PositionalEncoding(nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            0, self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x * freqs.view(1, -1).to(x.dtype)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels: int, scale: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * (self.freqs * 2 * torch.pi).view(1, 1, -1).to(x.dtype)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x
