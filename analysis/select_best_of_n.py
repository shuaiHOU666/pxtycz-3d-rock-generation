#!/usr/bin/env python
"""Summarise generation logs: unfiltered statistics AND best-of-N selection.

Reads every ``*_generation_log.txt`` below ``--samples`` (written by ``scripts/generate_batch.py``) and, per
specimen and prompt type (``poro`` = porosity prompt, ``topo`` = topology prompt), reports

* **all samples** (no selection): N, mean, std, mean absolute error (MAE) and median absolute error of the
  generated porosity w.r.t. the target;
* **best-of-N / oracle selection**: the ``--top`` samples whose measured porosity is closest to the target.
  This selection uses the target value, so it is an *upper bound* on what a user can obtain by generating N
  samples and keeping the best ones; it must be reported as such (never as unfiltered / random output).

Outputs (in ``--out``): ``all_samples.csv`` (one row per generated sample), ``summary_all_samples.csv``,
``selected_best_of_n.csv`` and a short ``README_selection.txt`` describing the rule.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def parse_log(path):
    name = target = None
    section = None
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("MOF Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Target Porosity:"):
            target = float(line.split(":", 1)[1])
        elif line.startswith("--- Prompt:"):
            section = "poro" if "porosity" in line else "topo"
        else:
            m = re.match(r"((?:topo|poro)_sample_\d+\.npy):\s+([\d.]+)", line)
            if m and name is not None:
                rows.append({"specimen": name, "prompt": m.group(1).split("_")[0], "file": m.group(1),
                             "target": target, "generated": float(m.group(2))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--top", type=int, default=2, help="number of samples kept per specimen/prompt (best-of-N)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(a.samples, "**", "*_generation_log.txt"), recursive=True)):
        rows += parse_log(p)
    df = pd.DataFrame(rows)
    df["abs_error"] = (df["generated"] - df["target"]).abs()
    df.to_csv(f"{a.out}/all_samples.csv", index=False)

    g = df.groupby(["specimen", "prompt"])["abs_error"]
    summ = pd.DataFrame({
        "N": g.size(),
        "mean_generated": df.groupby(["specimen", "prompt"])["generated"].mean(),
        "std_generated": df.groupby(["specimen", "prompt"])["generated"].std(),
        "target": df.groupby(["specimen", "prompt"])["target"].first(),
        "MAE": g.mean(), "median_abs_error": g.median(),
    }).reset_index()
    summ.to_csv(f"{a.out}/summary_all_samples.csv", index=False)

    sel = (df.sort_values("abs_error").groupby(["specimen", "prompt"]).head(a.top)
             .sort_values(["specimen", "prompt", "abs_error"]).copy())
    sel["rank"] = sel.groupby(["specimen", "prompt"]).cumcount() + 1
    sel["N_candidates"] = sel.set_index(["specimen", "prompt"]).index.map(summ.set_index(["specimen", "prompt"])["N"])
    sel.to_csv(f"{a.out}/selected_best_of_n.csv", index=False)

    open(f"{a.out}/README_selection.txt", "w").write(
        f"Selection rule: for each specimen and prompt type the {a.top} generated volumes whose measured porosity is\n"
        "closest to the TARGET porosity were kept (best-of-N, oracle selection using the target).\n"
        "These are not random draws. Unfiltered statistics are in summary_all_samples.csv.\n")

    pd.set_option("display.width", 160)
    for pr, t in summ.groupby("prompt"):
        print(f"\n== prompt '{pr}': all samples (unfiltered) ==")
        print(f"specimens={len(t)}  N per specimen={int(t['N'].min())}-{int(t['N'].max())}  "
              f"mean MAE={t['MAE'].mean():.4f}  median of MAE={t['MAE'].median():.4f}")
    for pr, t in sel.groupby("prompt"):
        print(f"best-of-N top-{a.top} ('{pr}'): mean |error| = {t['abs_error'].mean():.4f}")


if __name__ == "__main__":
    main()
