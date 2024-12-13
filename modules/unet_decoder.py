from typing import List, Mapping, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .layers.fourier_embedding import FourierEmbedding as NoiseFourierEmbedding
from .layers.mlp import MLPLayer
from .layers.unet_block import AttentionBlock, DownSample, ResnetBlock, UpSample
from .utils import weight_init


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg["model"]["unet"]
        self.input_dim = cfg["model"]["input_dim"]
        self.output_dim = cfg["model"]["output_dim"]
        self.resolution = cfg["model"]["resolution"]
        self.model_channel = self.cfg["model_channel"]
        self.channel_mult = self.cfg["channel_mult"]
        self.channel_mult_emb = self.cfg["channel_mult_emb"]
        self.channels_per_head = self.cfg["channels_per_head"]
        self.num_blocks = self.cfg["num_blocks"]
        self.dropout = self.cfg["dropout"]

        self.emb_channel = self.model_channel * self.channel_mult_emb
        self.noise_emb = NoiseFourierEmbedding(
            input_dim=1,
            hidden_dim=self.emb_channel,
            num_freq_bands=self.emb_channel // 2,
            noise=True,
        )

        # conditional embedding
        num_regions = cfg["model"]["num_regions"]
        self.home_emb = nn.Embedding(num_regions, self.emb_channel)
        self.end_emb = nn.Embedding(num_regions, self.emb_channel)
        self.cond_proj = MLPLayer(
            input_dim=self.emb_channel * 2,
            hidden_dim=self.emb_channel,
            output_dim=self.emb_channel,
        )

        # downsample
        self.conv_in = nn.Conv1d(
            self.input_dim, self.model_channel, kernel_size=3, padding=1
        )
        self.down_blocks = nn.ModuleDict()
        cin, cout = self.model_channel, self.model_channel
        for level, mult in enumerate(self.channel_mult):
            res = self.resolution >> level
            cin, cout = cout, self.model_channel * mult
            res_block = nn.ModuleList()
            for _ in range(self.num_blocks):
                res_block.append(ResnetBlock(cin, cout, self.dropout, self.emb_channel))
                cin = cout
            self.down_blocks[f"{res}_resnet"] = res_block
            if level < len(self.channel_mult) - 1:
                self.down_blocks[f"{res}_down"] = DownSample(cout, cout)

        # middle
        self.middle_block = nn.ModuleDict(
            {
                "resnet1": ResnetBlock(cout, cout, self.dropout, self.emb_channel),
                "attn": AttentionBlock(
                    hidden_dim=cout,
                    num_heads=cout // self.channels_per_head,
                    dropout=self.dropout,
                ),
                "resnet2": ResnetBlock(cout, cout, self.dropout, self.emb_channel),
            }
        )

        # upsample
        self.up_blocks = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(self.channel_mult[:-1]))):
            res = self.resolution >> level
            cin, cout = cout, self.model_channel * mult
            self.up_blocks[f"{res}_up"] = UpSample(cin, cout)
            res_block = nn.ModuleList()
            for i in range(self.num_blocks):
                if i == 0:
                    res_block.append(ResnetBlock(cout * 2, cout, self.dropout))
                else:
                    res_block.append(ResnetBlock(cout, cout, self.dropout))
            self.up_blocks[f"{res}_resnet"] = res_block

        # output
        self.norm_out = nn.GroupNorm(32, cout)
        self.conv_out = nn.Conv1d(cout, self.output_dim, kernel_size=3, padding=1)
        self.apply(weight_init)

    def forward(
        self,
        data: HeteroData,
        noised_gt: torch.Tensor,
        noise_labels: torch.Tensor,
    ) -> torch.Tensor:
        x = noised_gt.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)

        noise_emb = self.noise_emb(
            continuous_inputs=noise_labels.unsqueeze(-1), categorical_embs=None
        )  # (batch_size, model_channel * channel_mult_emb)
        home, end = data["x_loc"][:, 0], data["x_loc"][:, -1]
        h_emb, e_emb = self.home_emb(home), self.end_emb(end)
        cond_emb = self.cond_proj(
            torch.cat([h_emb, e_emb], dim=1)
        )  # (batch_size, model_channel * channel_mult_emb)
        noise_emb += cond_emb

        # downsample
        x = self.conv_in(x)  # (batch_size, model_channel, seq_len)
        skips = []
        for name, block in self.down_blocks.items():
            if "resnet" in name:
                for res_block in block:
                    x = res_block(x, noise_emb)
            elif "down" in name:
                skips.append(x)
                x = block(x)

        # middle
        for name, block in self.middle_block.items():
            if "resnet" in name:
                x = block(x, noise_emb)
            else:
                x = block(x)

        # upsample
        for name, block in self.up_blocks.items():
            if "up" in name:
                x = block(x)
            elif "resnet" in name:
                x = torch.cat([x, skips.pop()], dim=1)
                for res_block in block:
                    x = res_block(x, noise_emb)

        # output
        x = self.norm_out(x)  # (batch_size, model_channel, seq_len)
        x = self.conv_out(x)  # (batch_size, output_dim, seq_len)
        return x.permute(0, 2, 1)  # (batch_size, seq_len, output_dim)
