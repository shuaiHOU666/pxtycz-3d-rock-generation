#!/usr/bin/env python
"""Stage II: train the text-conditioned latent diffusion model (VQ-VAE frozen).

    python scripts/train_diffusion.py --root data/ --res 128 --vq-cfg configs/vqvae_128.yaml \
        --vq-ckpt runs/vq128/vqvae_steps-5000.pth --out runs/df128
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from porefusion.data import RockDataset  # noqa: E402
from porefusion.diffusion import TextLatentDiffusion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--vq-cfg", default="configs/vqvae_128.yaml")
    ap.add_argument("--vq-ckpt", required=True)
    ap.add_argument("--df-cfg", default="configs/diffusion_text.yaml")
    ap.add_argument("--text-encoder", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--out", default="runs/diffusion")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from porefusion.text_encoder import create_text_encoder
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RockDataset(a.root, a.res)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=len(ds) >= a.batch_size)
    model = TextLatentDiffusion(a.df_cfg, a.vq_cfg, a.vq_ckpt, device=device, lr=a.lr,
                                text_encoder=create_text_encoder(a.text_encoder))

    it = 0
    while it < a.iters:
        for b in dl:
            log = model.train_step(b["x"], list(b["text"]))
            it += 1
            if it % 50 == 0:
                print(f"iter {it}: loss={log['loss']:.4f}")
            if it % a.save_every == 0 or it == a.iters:
                model.save(os.path.join(a.out, f"df_steps-{it}.pth"), it)
            if it >= a.iters:
                break


if __name__ == "__main__":
    main()
