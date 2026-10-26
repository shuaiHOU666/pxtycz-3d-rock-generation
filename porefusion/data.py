"""Datasets.

Expected layout (see docs/DATA.md)::

    <root>/resolution_<R>/<name>+<i>+<j>.npy      binary volume (R,R,R), 1 = pore, 0 = solid
    <root>/resolution_<R>/combined_results.txt    tab-separated table; columns ``name``, ``Porosity``,
                                                  ``grid_classification`` (PxTyCz label) are required
    <root>/resolution_<R>/text_data.txt           optional; lines ``<stem>, <prompt>`` (written by
                                                  scripts/make_text_prompts.py)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from porefusion.prompts import training_prompt


def to_one_hot(vol: np.ndarray) -> torch.Tensor:
    """(R,R,R) binary -> float tensor (2,R,R,R); channel 0 = solid, channel 1 = pore."""
    v = torch.from_numpy(np.ascontiguousarray(vol)).long()
    return torch.nn.functional.one_hot(v, 2).permute(3, 0, 1, 2).float()


class RockDataset(Dataset):
    def __init__(self, root: str, res: int = 128, max_size: int | None = None, return_text: bool = True):
        self.dir = os.path.join(root, f"resolution_{res}")
        self.res = res
        table = pd.read_csv(os.path.join(self.dir, "combined_results.txt"), sep=r"\s+")
        info = {str(r["name"]): (float(r["Porosity"]), str(r["grid_classification"]))
                for _, r in table.iterrows()}
        self.files = sorted(f for f in os.listdir(self.dir) if f.endswith(".npy"))
        self.files = [f for f in self.files if f.split("+")[0] in info]
        if max_size:
            self.files = self.files[:max_size]
        self.info = info
        self.return_text = return_text
        self.text = {}
        tpath = os.path.join(self.dir, "text_data.txt")
        if os.path.exists(tpath):
            for line in open(tpath, encoding="utf-8"):
                if "," in line:
                    k, v = line.split(",", 1)
                    self.text[k.strip()] = v.strip()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        stem = f[:-4]
        vol = np.load(os.path.join(self.dir, f)).astype(np.uint8)
        assert vol.shape == (self.res,) * 3, f"{f}: expected {(self.res,) * 3}, got {vol.shape}"
        name = stem.split("+")[0]
        phi, topo = self.info[name]
        text = self.text.get(stem) or training_prompt(stem, topo, f"{phi:.4f}")
        return {"x": to_one_hot(vol), "text": text, "name": stem, "porosity": phi, "topology": topo}
