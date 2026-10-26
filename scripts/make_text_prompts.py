#!/usr/bin/env python
"""Write the text prompts used for training.

For every ``<name>+<i>+<j>.npy`` volume in ``<root>/resolution_<R>/`` look up the specimen ``<name>`` in
``combined_results.txt`` and write one line per volume::

    <stem>, <stem> exhibits a <PxTyCz> topology and a porosity of <phi>

to ``text_data.txt`` (same folder).
"""
import argparse
import glob
import os

import pandas as pd

from porefusion.prompts import training_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing resolution_<R>/")
    ap.add_argument("--res", type=int, default=128)
    a = ap.parse_args()

    d = os.path.join(a.root, f"resolution_{a.res}")
    table = pd.read_csv(os.path.join(d, "combined_results.txt"), sep=r"\s+", dtype=str)
    info = {r["name"]: (r["Porosity"], r["grid_classification"]) for _, r in table.iterrows()}

    n = 0
    with open(os.path.join(d, "text_data.txt"), "w", encoding="utf-8") as out:
        for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            prefix = stem.split("+")[0]
            if prefix not in info:
                print(f"skip {stem}: '{prefix}' not in combined_results.txt")
                continue
            phi, topo = info[prefix]
            out.write(f"{stem}, {training_prompt(stem, topo, phi)}\n")
            n += 1
    print(f"wrote {n} prompts")


if __name__ == "__main__":
    main()
