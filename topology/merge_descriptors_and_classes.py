#!/usr/bin/env python
"""Merge the descriptor table (``summary.txt``) with the PxTyCz labels written by
``pxtycz_grid_classification.py`` (``grid_classification_results.csv``).

The result (columns ``name, Porosity, ..., grid_classification, region_description``) is the
``combined_results.txt`` / ``combined_summary_merged.txt`` file used by the dataset, prompt and SISSO scripts.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="./grid_classification_results/grid_classification_results.csv")
    ap.add_argument("--summary", default="summary.txt")
    ap.add_argument("--out", default="combined_summary_merged.txt")
    a = ap.parse_args()

    cls = pd.read_csv(a.classes)[["sample_name", "grid_classification", "region_description"]]
    cls = cls.rename(columns={"sample_name": "name"})
    summ = pd.read_csv(a.summary, sep=r"\s+")
    merged = pd.merge(summ, cls, on="name", how="left")  # row set of summary.txt is kept
    merged.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out} ({len(merged)} rows; {merged['grid_classification'].isna().sum()} without a class)")


if __name__ == "__main__":
    main()
