"""Stage II: text-conditioned latent diffusion (3D UNet, cross-attention on SciBERT features).

The model works on the *continuous* (pre-quantisation) VQ-VAE latent. SciBERT is frozen. The
768x768 linear projection ``cond_layer`` is **not** passed to the optimiser (same as in the
experiments reported in the paper), i.e. it stays a fixed random projection; set
``train_cond_layer=True`` to change that.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from porefusion.networks.diffusion.ldm_diffusion_util import default, extract_into_tensor, make_beta_schedule
from porefusion.networks.diffusion.network import DiffusionUNet
from porefusion.networks.diffusion.samplers.ddim import DDIMSampler
from porefusion.networks.vqvae.network import VQVAE
from porefusion.text_encoder import create_text_encoder
from porefusion.vqvae import latent_shape


class TextLatentDiffusion:
    def __init__(self, df_cfg: str, vq_cfg: str, vq_ckpt: str | None = None, device: str = "cuda",
                 lr: float = 1e-4, text_encoder=None, max_text_len: int = 256,
                 train_cond_layer: bool = False):
        self.device = torch.device(device)
        self.max_text_len = max_text_len
        df_conf, vq_conf = OmegaConf.load(df_cfg), OmegaConf.load(vq_cfg)
        self.z_shape = latent_shape(vq_cfg)

        # networks
        p = vq_conf.model.params
        self.vqvae = VQVAE(p.ddconfig, p.n_embed, p.embed_dim)
        if vq_ckpt is not None:
            sd = torch.load(vq_ckpt, map_location="cpu")
            self.vqvae.load_state_dict(sd["vqvae"] if "vqvae" in sd else sd)
        self.vqvae.to(self.device).eval()
        for q in self.vqvae.parameters():
            q.requires_grad = False

        self.df = DiffusionUNet(df_conf.unet.params, vq_conf=vq_conf,
                                conditioning_key=df_conf.model.params.conditioning_key).to(self.device)
        if text_encoder is None:
            text_encoder = create_text_encoder()
        self.cond_model, self.tokenizer = text_encoder
        self.cond_model.to(self.device).eval()
        for q in self.cond_model.parameters():
            q.requires_grad = False
        h = self.cond_model.config.hidden_size
        self.cond_layer = nn.Linear(h, h).to(self.device)

        params = list(self.df.parameters())
        if train_cond_layer:
            params += list(self.cond_layer.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr)

        self._register_schedule(df_conf.model.params)
        self.parameterization = "eps"

    # ------------------------------------------------------------------ schedule
    def _register_schedule(self, mp):
        betas = make_beta_schedule("linear", mp.timesteps, linear_start=mp.linear_start, linear_end=mp.linear_end)
        alphas = 1.0 - betas
        ac = np.cumprod(alphas, axis=0)
        ac_prev = np.append(1.0, ac[:-1])
        self.num_timesteps = int(betas.shape[0])
        f = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.betas, self.alphas_cumprod, self.alphas_cumprod_prev = f(betas), f(ac), f(ac_prev)
        self.sqrt_alphas_cumprod = f(np.sqrt(ac))
        self.sqrt_one_minus_alphas_cumprod = f(np.sqrt(1.0 - ac))

    def q_sample(self, x0, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def apply_model(self, x_noisy, t, cond):
        out = self.df(x_noisy, t, c_crossattn=[cond])
        return out[0] if isinstance(out, tuple) else out

    # ------------------------------------------------------------------ conditioning
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                             max_length=self.max_text_len)
        with torch.no_grad():
            c = self.cond_model(input_ids=tok["input_ids"].to(self.device),
                                attention_mask=tok["attention_mask"].to(self.device))["last_hidden_state"]
        return self.cond_layer(c)

    # ------------------------------------------------------------------ training
    def train_step(self, x: torch.Tensor, texts: list[str]) -> dict:
        """x: float tensor (B, C, R, R, R) (one-hot pore/solid); texts: B prompts."""
        self.df.train()
        x = x.to(self.device)
        c = self.encode_text(texts)
        with torch.no_grad():
            z = self.vqvae(x, forward_no_quant=True, encode_only=True)
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        noise = torch.randn_like(z)
        pred = self.apply_model(self.q_sample(z, t, noise), t, c)
        loss = torch.nn.functional.mse_loss(pred, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach())}

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def txt2shape(self, prompt: str, ngen: int = 4, ddim_steps: int = 200, ddim_eta: float = 0.0):
        """Generate ``ngen`` volumes for one prompt. Returns logits of shape (ngen, C, R, R, R)."""
        self.df.eval()
        c = self.encode_text([prompt] * ngen)
        sampler = DDIMSampler(self)
        z, _ = sampler.sample(S=ddim_steps, batch_size=ngen, shape=self.z_shape, conditioning=c,
                              verbose=False, eta=ddim_eta)
        return self.vqvae.decode_no_quant(z)

    @staticmethod
    def to_binary(logits: torch.Tensor) -> np.ndarray:
        """argmax over the channel axis -> uint8 volumes (B, R, R, R); channel 1 = pore."""
        return torch.argmax(logits, dim=1).to(torch.uint8).cpu().numpy()

    # ------------------------------------------------------------------ io
    def save(self, path: str, step: int = 0):
        torch.save(OrderedDict(vqvae=self.vqvae.state_dict(), df=self.df.state_dict(),
                               cond_model=self.cond_model.state_dict(),
                               cond_layer=self.cond_layer.state_dict(),
                               optimizer=self.optimizer.state_dict(), global_step=step), path)

    def load(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.vqvae.load_state_dict(sd["vqvae"])
        self.df.load_state_dict(sd["df"])
        self.cond_model.load_state_dict(sd["cond_model"])
        self.cond_layer.load_state_dict(sd["cond_layer"])
