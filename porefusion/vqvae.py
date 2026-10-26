"""Stage I: 3D VQ-VAE that compresses a binary pore/solid volume into a latent grid."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from porefusion.losses import VQLoss
from porefusion.networks.vqvae.network import VQVAE


def build_vqvae(cfg_path: str) -> tuple[VQVAE, VQLoss]:
    cfg = OmegaConf.load(cfg_path)
    p = cfg.model.params
    net = VQVAE(p.ddconfig, p.n_embed, p.embed_dim)
    loss = VQLoss(codebook_weight=cfg.lossconfig.params.codebook_weight)
    return net, loss


def latent_shape(cfg_path: str) -> tuple[int, int, int, int]:
    """Shape (C, D, H, W) of the latent grid implied by a VQ-VAE config."""
    dd = OmegaConf.load(cfg_path).model.params.ddconfig
    n_down = len(dd.ch_mult) - 1
    s = dd.resolution // (2 ** n_down)
    return (dd.z_channels, s, s, s)


class VQVAETrainer:
    """Minimal training wrapper (one optimiser step per call)."""

    def __init__(self, cfg_path: str, lr: float = 1e-4, device: str = "cuda"):
        self.device = torch.device(device)
        self.net, self.loss_fn = build_vqvae(cfg_path)
        self.net.to(self.device)
        self.loss_fn.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.9))
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, 1000, 0.9)

    def step(self, x: torch.Tensor) -> dict:
        """x: float tensor (B, C, R, R, R); C must equal ddconfig.in_channels."""
        self.net.train()
        x = x.to(self.device)
        x_rec, qloss = self.net(x)
        loss, log = self.loss_fn(qloss, x, x_rec)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.sched.step()
        return {k: float(v) for k, v in log.items()}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        z = self.net(x.to(self.device), forward_no_quant=True, encode_only=True)
        return self.net.decode_no_quant(z)

    def save(self, path: str, step: int = 0):
        torch.save({"vqvae": self.net.state_dict(), "global_step": step}, path)

    def load(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.net.load_state_dict(sd["vqvae"] if "vqvae" in sd else sd)
