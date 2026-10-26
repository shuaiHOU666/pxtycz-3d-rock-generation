#!/usr/bin/env python
"""Symbolic regression (SISSO, via TorchSISSO) of elastic moduli and permeability on pore descriptors.

Reproduces the protocol used for the paper:

* inputs   : six descriptors (+ aspect ratio, kept as a candidate) taken from the descriptor table
             written by ``topology/crop_and_extract_network_features.py`` (``summary`` file)
* targets  : bulk modulus, shear modulus, permeability from ``--labels`` (columns ``k``, ``miu``,
             ``permeability``; names are historical and kept for compatibility with the data release)
* features are Min-Max scaled to [0, 1]. NOTE: as in the original experiments the scaler is fitted on the
  full table *before* the train/test split; use ``--scale-on-train-only`` to fit it on the training part only.
* split    : random 90/10 (seed 82); equation search with n_expansion = 2 (rung 2), n_term = 2, k = 40
* optional : ``--rung-cv`` repeats the search for rungs 1-3 with 10-fold CV (Table S3)

NOTE: samples are split at random at the *sample* level. If several samples derive from the same parent
specimen this does not measure generalisation to unseen specimens.

Usage
-----
    python sisso/fit_sisso.py --labels label.txt --descriptors combined_summary_merged.txt --out results/
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from TorchSisso import SissoModel

# descriptor column in the table -> short symbol used inside the discovered equations
FEATURES = {
    "Aspect_Ratio": "AR",
    "Mean_Throat_Radius": "Rt",
    "Mean_Coordination": "Z",
    "Sphericity": "Sp",
    "Tortuosity": "Tau",
    "Fractal_Dimension": "Df",
    "Porosity": "Phi",
}
TARGETS = ["k", "miu", "permeability"]  # bulk modulus, shear modulus, permeability
OPERATORS = ['+', '-', '*', '/', 'exp', 'pow(1/2)', 'pow(1/3)', 'log', 'ln', '^-1', 'pow(2)', 'pow(3)', 'exp(-1)']


def load_table(labels, descriptors):
    df1 = pd.read_csv(labels, sep=r"\s+")
    df2 = pd.read_csv(descriptors, sep="\t")
    df2 = df2.rename(columns={df2.columns[0]: "name"})
    df1["name"], df2["name"] = df1["name"].astype(str), df2["name"].astype(str)
    df = pd.merge(df1, df2, on="name").rename(columns=FEATURES)
    cols = list(FEATURES.values())
    return df[cols + TARGETS].dropna().copy(), cols


def evaluate_equation(equation, X, y, names):
    env = {"__builtins__": None, "sqrt": np.sqrt, "ln": np.log, "log": np.log, "exp": np.exp,
           "abs": np.abs, "pow": pow}
    pred = np.empty(len(X))
    for i in range(len(X)):
        try:
            pred[i] = eval(equation.strip(), {**env, **{n: X[i, j] for j, n in enumerate(names)}})
        except Exception:
            pred[i] = np.mean(y)
    return float(np.sqrt(mean_squared_error(y, pred))), float(r2_score(y, pred)), pred


def search(X, y, names, n_expansion, k, device):
    df = pd.DataFrame(np.column_stack([y, X]), columns=["y"] + names)
    model = SissoModel(df, OPERATORS, n_expansion=n_expansion, n_term=2, k=k, device=device,
                       use_gpu=device == "cuda")
    res = model.fit()
    return res[1]  # equation string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--descriptors", required=True)
    ap.add_argument("--out", default="sisso_results")
    ap.add_argument("--seed", type=int, default=82)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--scale-on-train-only", action="store_true")
    ap.add_argument("--rung-cv", action="store_true", help="also run rung 1-3 10-fold CV (slow, GPU advised)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, names = load_table(a.labels, a.descriptors)
    print(f"{len(df)} samples, device={device}")

    if not a.scale_on_train_only:
        df[names] = MinMaxScaler().fit_transform(df[names])

    report, cv_rows = [], []
    for tgt in TARGETS:
        X, y = df[names].values, df[tgt].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, random_state=a.seed)
        if a.scale_on_train_only:
            sc = MinMaxScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        eq = search(Xtr, ytr, names, n_expansion=2, k=40, device=device)
        rm_tr, r2_tr, _ = evaluate_equation(eq, Xtr, ytr, names)
        rm_te, r2_te, pred = evaluate_equation(eq, Xte, yte, names)
        report.append({"target": tgt, "equation": eq, "train_R2": r2_tr, "test_R2": r2_te,
                       "train_RMSE": rm_tr, "test_RMSE": rm_te})
        pd.DataFrame({"actual": yte, "predicted": pred}).to_csv(f"{a.out}/test_pred_{tgt}.csv", index=False)
        print(f"[{tgt}] {eq}  trainR2={r2_tr:.3f} testR2={r2_te:.3f}")

        if a.rung_cv:
            for rung in (1, 2, 3):
                te, tr = [], []
                for tri, tei in KFold(10, shuffle=True, random_state=a.seed).split(X):
                    e = search(X[tri], y[tri], names, rung, 40 if rung <= 2 else 20, device)
                    tr.append(evaluate_equation(e, X[tri], y[tri], names)[1])
                    te.append(evaluate_equation(e, X[tei], y[tei], names)[1])
                cv_rows.append({"target": tgt, "rung": rung, "train_R2": np.mean(tr), "test_R2": np.mean(te)})

    pd.DataFrame(report).to_csv(f"{a.out}/equations.csv", index=False)
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(f"{a.out}/rung_cv.csv", index=False)


if __name__ == "__main__":
    main()
