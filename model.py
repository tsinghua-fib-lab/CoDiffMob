import inspect
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from metrics.nanchang import (
    INDEX2GRID,
    INDEX2LONLAT,  # REGION_EMB,
    INDEX2XY,
    LONLAT_MEAN,
    LONLAT_STD,
    MAXX,
    MAXY,
    MINX,
    MINY,
    complete_transition_matrix,
    daily_loc,
    duration,
    gyration_radius,
    ks_test,
    travel_distance,
)
from modules.unet_decoder import UNet
from modules.utils import weight_init
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

P_MEAN = -1.2
P_STD = 1.2
TIME_INTERVAL = 1800
LENGTH = 86400


class RegionDiff(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super(RegionDiff, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg["model"]
        self.dataset = cfg["dataset"]["name"]
        self.target = self.cfg["target"]
        self.noise_prior = self.cfg["noise_prior"]
        self.input_dim = self.cfg["input_dim"]
        self.output_dim = self.cfg["output_dim"]
        self.lr = self.cfg["lr"]
        self.lr_scheduler = self.cfg["lr_scheduler"]
        self.weight_decay = self.cfg["weight_decay"]
        self.T_max = self.cfg["T_max"]
        self.metrics = self.cfg["metrics"]
        self.diffusion_steps = self.cfg["diffusion"]["num_steps"]
        self.sample_steps = self.cfg["diffusion"]["num_sample_steps"]
        self.beta_start = self.cfg["diffusion"]["beta_start"]
        self.beta_end = self.cfg["diffusion"]["beta_end"]

        # diffusion params
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.diffusion_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.unet_decoder = UNet(cfg=cfg)

        # validation metrics
        self.valildation_step_outputs = {
            m: {"real": [], "gen": []} for m in self.metrics
        }
        self.valildation_step_outputs["transition"] = {
            "real": torch.zeros((INDEX2GRID.max() + 1, INDEX2GRID.max() + 1)),
            "gen": torch.zeros((INDEX2GRID.max() + 1, INDEX2GRID.max() + 1)),
        }

        self.apply(weight_init)

    def forward(
        self, data: torch.Tensor, noised_gt: torch.Tensor, noise_label: torch.Tensor
    ) -> torch.Tensor:
        noised_gt = noised_gt.to(torch.float32)
        out = self.unet_decoder(data, noised_gt, noise_label)
        return out

    def training_step(self, batch, batch_idx):
        # add noise to gt
        gt = self._get_training_target(batch)
        batch_size = gt.shape[0]
        t = torch.randint(
            low=0, high=self.diffusion_steps, size=(batch_size // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.diffusion_steps - t - 1], dim=0)[:batch_size]
        c = self.alpha_bar.to(gt.device).gather(-1, t).reshape(-1, 1, 1)
        mean = c**0.5 * gt
        var = 1 - c
        eps = torch.randn_like(gt).to(gt.device)
        xt = mean + (var**0.5) * eps

        # forward
        out = self.forward(batch, xt, t)

        # loss
        loss = F.mse_loss(eps.float(), out)

        self.log(
            "train_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def _get_training_target(self, batch) -> torch.Tensor:
        if self.target == "loc":
            return self.loc_emb(batch["x_loc"])
        return batch["x"]

    def _reconstruct_idx(self, sample: torch.Tensor) -> torch.Tensor:
        if self.target == "loc":
            loc_emb = self.loc_emb.weight.clone().detach().cpu()
            distance = torch.cdist(
                sample, loc_emb, compute_mode="donot_use_mm_for_euclid_dist"
            )
            sample_idx = torch.argmin(distance, dim=-1)
            return sample_idx
        elif self.target == "xy":
            sample = torch.stack(
                [
                    sample[:, :, 0] * (MAXX - MINX) + MINX,
                    sample[:, :, 1] * (MAXY - MINY) + MINY,
                ],
                dim=-1,
            ).to(torch.float32)
            loc = torch.tensor(INDEX2XY, dtype=torch.float32)
            distance = torch.cdist(
                sample, loc, compute_mode="donot_use_mm_for_euclid_dist"
            )
            sample_idx = torch.argmin(distance, dim=-1)
            return sample_idx
        elif self.target == "lonlat":
            loc = torch.tensor(INDEX2LONLAT, dtype=torch.float32)
            sample = (sample * LONLAT_STD + LONLAT_MEAN).to(torch.float32)
            distance = torch.sum((sample[:, :, None] - loc) ** 2, dim=3)
            sample_idx = torch.argmin(distance, dim=-1)
            return sample_idx

    def validation_step(self, batch, batch_idx):
        x_idx = batch["x_loc"].cpu()
        sample = (
            self.sampling(data=batch, show_progress=False, num_steps=self.sample_steps)
            .detach()
            .cpu()
            .to(torch.float32)
        )
        gen_idx = self._reconstruct_idx(sample)
        # calculate metrics
        if "distance" in self.metrics:
            self.valildation_step_outputs["distance"]["real"].extend(
                travel_distance(x_idx)
            )
            self.valildation_step_outputs["distance"]["gen"].extend(
                travel_distance(gen_idx)
            )
        if "radius" in self.metrics:
            self.valildation_step_outputs["radius"]["real"].extend(
                gyration_radius(x_idx)
            )
            self.valildation_step_outputs["radius"]["gen"].extend(
                gyration_radius(gen_idx)
            )
        if "duration" in self.metrics:
            self.valildation_step_outputs["duration"]["real"].extend(duration(x_idx))
            self.valildation_step_outputs["duration"]["gen"].extend(duration(gen_idx))
        if "daily_loc" in self.metrics:
            self.valildation_step_outputs["daily_loc"]["real"].extend(daily_loc(x_idx))
            self.valildation_step_outputs["daily_loc"]["gen"].extend(daily_loc(gen_idx))
        if "cpc" in self.metrics or "mape" in self.metrics:
            self.valildation_step_outputs["transition"][
                "real"
            ] += complete_transition_matrix(x_idx)
            self.valildation_step_outputs["transition"][
                "gen"
            ] += complete_transition_matrix(gen_idx)

    def on_validation_epoch_end(self):
        if "distance" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["distance"]["real"],
                self.valildation_step_outputs["distance"]["gen"],
            )
            if len(gen) == 0:
                gen = [0]
            ks_stat = ks_test(real, gen)
            self.log(
                "distance_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["distance"]["real"].clear()
            self.valildation_step_outputs["distance"]["gen"].clear()
        if "radius" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["radius"]["real"],
                self.valildation_step_outputs["radius"]["gen"],
            )
            ks_stat = ks_test(real, gen)
            self.log(
                "radius_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["radius"]["real"].clear()
            self.valildation_step_outputs["radius"]["gen"].clear()
        if "duration" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["duration"]["real"],
                self.valildation_step_outputs["duration"]["gen"],
            )
            if len(gen) == 0:
                gen = [0]
            ks_stat = ks_test(real, gen)
            self.log(
                "duration_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["duration"]["real"].clear()
            self.valildation_step_outputs["duration"]["gen"].clear()
        if "daily_loc" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["daily_loc"]["real"],
                self.valildation_step_outputs["daily_loc"]["gen"],
            )
            if len(gen) == 0:
                gen = [0]
            ks_stat = ks_test(real, gen)
            self.log(
                "daily_loc_kstest",
                value=ks_stat,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.valildation_step_outputs["daily_loc"]["real"].clear()
            self.valildation_step_outputs["daily_loc"]["gen"].clear()
        if "cpc" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["transition"]["real"],
                self.valildation_step_outputs["transition"]["gen"],
            )
            cpc = (2 * torch.sum(torch.min(real, gen)) / torch.sum(real + gen)).item()
            self.log(
                "cpc",
                value=cpc,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
        if "mape" in self.metrics:
            real, gen = (
                self.valildation_step_outputs["transition"]["real"],
                self.valildation_step_outputs["transition"]["gen"],
            )
            transprob_real = (real + 1e-6) / torch.sum(real + 1e-6, dim=1, keepdim=True)
            transprob_gen = (gen + 1e-6) / torch.sum(gen + 1e-6, dim=1, keepdim=True)
            ae = torch.abs(transprob_real - transprob_gen)
            index = transprob_real > 0.01
            mape = torch.mean(ae[index] / transprob_real[index]).item()
            self.log(
                "mape",
                value=mape,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
        self.valildation_step_outputs["transition"]["real"].zero_()
        self.valildation_step_outputs["transition"]["gen"].zero_()

    @torch.no_grad()
    def sampling(
        self,
        data: Mapping[str, torch.Tensor],
        latent: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        eta: float = 0.0,
        return_his: bool = False,
        show_progress: bool = True,
    ) -> torch.Tensor | List[torch.Tensor]:
        if return_his:
            res = []
        device = next(self.parameters()).device
        # latent
        batch_size, seq_len, _ = self._get_training_target(data).shape
        output_dim = self.output_dim
        if latent is None:
            latent = torch.randn((batch_size, seq_len, output_dim), dtype=torch.float64)
            if self.noise_prior:
                latent = latent * self.noise_std[None, :, None]
                # latent = self.noise_sampling(data)
        latent = latent.to(device)
        # denoising time steps
        t_steps = range(0, self.diffusion_steps, self.diffusion_steps // num_steps)
        t_next = [-1] + list(t_steps[:-1])
        beta = torch.cat([torch.zeros(1).to(device), self.beta.to(device)], dim=0).to(
            torch.float64
        )
        alpha_cumprod = (1 - beta).cumprod(dim=0)

        x_next = latent
        if show_progress:
            bar = tqdm.tqdm(
                total=num_steps, unit="step", desc="Sampling", dynamic_ncols=True
            )
        for t_c, t_n in zip(reversed(t_steps), reversed(t_next)):
            t_cur = torch.ones((batch_size,), dtype=torch.long, device=device) * t_c
            t_next = torch.ones((batch_size,), dtype=torch.long, device=device) * t_n
            pre_noise = self.forward(data, x_next, t_cur).to(torch.float64)

            at = alpha_cumprod.index_select(0, t_cur + 1).view(-1, 1, 1)
            at_next = alpha_cumprod.index_select(0, t_next + 1).view(-1, 1, 1)

            x0_t = (x_next - pre_noise * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = (1 - at_next - c1**2).sqrt()
            eps = torch.randn(x_next.shape, device=x_next.device)
            x_next = at_next.sqrt() * x0_t + c1 * eps + c2 * pre_noise

            if return_his:
                res.append(x_next.cpu().numpy())
            if show_progress:
                bar.update(1)
        if show_progress:
            bar.close()
        if return_his:
            return res
        return x_next

    def inverse_sampling(
        self,
        data: Mapping[str, torch.Tensor],
        latent: Optional[torch.Tensor] = None,
        num_steps: int = 500,
        return_his: bool = False,
        show_progress: bool = True,
    ) -> torch.Tensor | List[torch.Tensor]:
        if return_his:
            res = []
        device = next(self.parameters()).device
        # latent
        if latent is None:
            latent = self._get_training_target(data)
        latent = latent.to(device)
        # noising time steps
        t_steps = range(0, self.diffusion_steps, self.diffusion_steps // num_steps)
        beta = torch.cat([torch.zeros(1).to(device), self.beta.to(device)], dim=0).to(
            torch.float64
        )
        alpha_cumprod = (1 - beta).cumprod(dim=0)

        if show_progress:
            bar = tqdm.tqdm(
                total=num_steps - 1,
                unit="step",
                desc="Inverse Sampling",
                dynamic_ncols=True,
            )
        for i in range(1, num_steps):
            t_n = t_steps[i]
            t_c = max(0, t_n - (self.diffusion_steps // num_steps))
            t_cur = (
                torch.ones((latent.shape[0],), dtype=torch.long, device=device) * t_c
            )
            t_next = (
                torch.ones((latent.shape[0],), dtype=torch.long, device=device) * t_n
            )
            pre_noise = self.forward(data, latent, t_next)

            at = alpha_cumprod.index_select(0, t_cur).view(-1, 1, 1)
            at_next = alpha_cumprod.index_select(0, t_next).view(-1, 1, 1)

            latent = (latent - (1 - at).sqrt() * pre_noise) * (
                at_next.sqrt() / at.sqrt()
            ) + (1 - at_next).sqrt() * pre_noise
            if return_his:
                res.append(latent.cpu().numpy())
            if show_progress:
                bar.update(1)
        if show_progress:
            bar.close()
        if return_his:
            return res
        return latent

    def noise_sampling(self, data: Mapping[str, torch.Tensor]):
        latent = data["latent"]
        sample_noise = torch.zeros_like(latent)
        sample_noise[:, 0] = torch.randn_like(latent[:, 0])
        for i in range(1, latent.size(1)):
            x = sample_noise[:, :i]
            t_pred, s_pred = self.noise_sampler(data, x)
            t_pred, s_pred = t_pred[:, -1], s_pred[:, -1]
            norm, angle = s_pred[:, 0], s_pred[:, 1]
            move = torch.multinomial(torch.nn.functional.softmax(t_pred, dim=-1), 1)
            dx, dy = norm * torch.cos(angle), norm * torch.sin(angle)
            cur = sample_noise[:, i - 1]
            next_pos = cur + torch.stack([dx, dy], dim=1)
            sample_noise[:, i] = move * next_pos + (1 - move) * cur
        noise_std = torch.std(sample_noise, dim=0).mean(dim=1)
        sample_noise = sample_noise * (1 / noise_std.max())
        return sample_noise

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.LSTMCell,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.Embedding,
            nn.GroupNorm,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        params = {
            "optimizer": optimizer,
            "T_max": self.T_max,
            "total_steps": self.T_max,
            "max_lr": self.lr,
            "pct_start": 0.15,
        }
        s = eval(self.lr_scheduler)
        scheduler = s(
            **{k: v for k, v in params.items() if k in inspect.signature(s).parameters}
        )
        return [optimizer], [scheduler]
