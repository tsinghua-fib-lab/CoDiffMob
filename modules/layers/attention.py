from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm

from ..utils import weight_init


class AttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        norm_first: bool = False,
    ) -> None:
        super(AttentionLayer, self).__init__()
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
        return x
