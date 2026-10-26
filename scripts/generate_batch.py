#!/usr/bin/env python
"""Generate N volumes per specimen for partial text prompts and log the measured porosity.

For every row of the target table (``name``, ``Porosity``, ``grid_classification``) two prompts are used,
exactly as in the evaluation of the paper (they differ from the training prompts):

    "with a <PxTyCz> topology"        -> files topo_sample_<i>.npy
    "with a porosity of <phi>"        -> files poro_sample_<i>.npy

and a log ``<name>_generation_log.txt`` is written next to the volumes::

    MOF Name: <name>
    Target Topology: <PxTyCz>
    Target Porosity: <phi>
    ----------------------------------------
    --- Prompt: 'with a <PxTyCz> topology' ---
    topo_sample_0.npy: 0.0745
    ...

(The ``MOF Name`` header is kept for compatibility with the released logs.)
All N samples are saved. Selecting a subset (e.g. the sample closest to the target) is done afterwards, and
explicitly, by ``analysis/select_best_of_n.py``.

    python scripts/generate_batch.py --ckpt runs/df128/df_steps-5000.pth --vq-cfg configs/vqvae_128.yaml \
        --table data/example/combined_results.txt --n-samples 80 --out samples/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from porefusion.diffusion import TextLatentDiffusion  # noqa: E402
from porefusion.prompts import porosity_prompt, topology_prompt  # noqa: E402
from porefusion.text_encoder import create_text_encoder  # noqa: E402


def porosity(vol: np.ndarray) -> float:
    return float(vol.mean())  # fraction of pore voxels (channel 1 == pore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="diffusion checkpoint (contains VQ-VAE weights)")
    ap.add_argument("--vq-cfg", default="configs/vqvae_128.yaml")
    ap.add_argument("--df-cfg", default="configs/diffusion_text.yaml")
    ap.add_argument("--table", required=True, help="tab/space separated table with name, Porosity, grid_classification")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n-samples", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4, help="samples per forward pass")
    ap.add_argument("--ddim-steps", type=int, default=200)
    ap.add_argument("--ddim-eta", type=float, default=0.0)
    ap.add_argument("--text-encoder", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextLatentDiffusion(a.df_cfg, a.vq_cfg, None, device=device,
                                text_encoder=create_text_encoder(a.text_encoder))
    model.load(a.ckpt)

    table = pd.read_csv(a.table, sep=r"\s+", dtype=str)
    for _, row in table.iterrows():
        name, phi, topo = row["name"], row["Porosity"], row["grid_classification"]
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}_generation_log.txt"), "w") as log:
            log.write(f"MOF Name: {name}\nTarget Topology: {topo}\nTarget Porosity: {phi}\n" + "-" * 40 + "\n")
            for prefix, prompt in (("topo", topology_prompt(topo)), ("poro", porosity_prompt(phi))):
                log.write(f"--- Prompt: '{prompt}' ---\n")
                idx = 0
                while idx < a.n_samples:
                    n = min(a.batch, a.n_samples - idx)
                    vols = model.to_binary(model.txt2shape(prompt, ngen=n, ddim_steps=a.ddim_steps,
                                                           ddim_eta=a.ddim_eta))
                    for v in vols:
                        np.save(os.path.join(d, f"{prefix}_sample_{idx}.npy"), v.astype(np.uint8))
                        log.write(f"{prefix}_sample_{idx}.npy: {porosity(v):.4f}\n")
                        idx += 1
                    log.flush()
        print(f"{name}: done")


if __name__ == "__main__":
    main()
