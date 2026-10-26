#!/usr/bin/env python
"""Stage I: train the VQ-VAE.

    python scripts/train_vqvae.py --root data/ --res 128 --cfg configs/vqvae_128.yaml --out runs/vq128
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from porefusion.data import RockDataset  # noqa: E402
from porefusion.vqvae import VQVAETrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cfg", default="configs/vqvae_128.yaml")
    ap.add_argument("--out", default="runs/vqvae")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RockDataset(a.root, a.res)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=len(ds) >= a.batch_size)
    tr = VQVAETrainer(a.cfg, lr=a.lr, device=device)

    it = 0
    while it < a.iters:
        for b in dl:
            log = tr.step(b["x"])
            it += 1
            if it % 50 == 0:
                print(f"iter {it}: " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
            if it % a.save_every == 0 or it == a.iters:
                tr.save(os.path.join(a.out, f"vqvae_steps-{it}.pth"), it)
            if it >= a.iters:
                break


if __name__ == "__main__":
    main()
