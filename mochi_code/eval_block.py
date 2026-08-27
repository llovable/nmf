#!/usr/bin/env python3
"""Self-contained block-imputation evaluation for the NMF-Transformer.

For each modality fully missing (block missingness), impute it from the other
two and report, on the test split:
  - z-RMSE (data is z-scored; mean-imputation baseline ~= 1.0; lower better)
  - feature-wise Pearson r (mean over features; higher better)  [Axis B metric]
  - SD ratio = std(imputed)/std(true) per feature, mean (1.0 = truth; variance preservation)
Also prints each model's effective gamma per modality.

Usage: python eval_block.py <data_dir> NAME=ckpt [NAME2=ckpt2 ...]
"""
import sys
import numpy as np
import torch

from models_nmf_tf import load_nmf_tf, predict_nmf_tf, MODS
from train_gate import TripleSplitDataset

data_dir = sys.argv[1]
specs = sys.argv[2:]
device = torch.device("cpu")

train = TripleSplitDataset(data_dir, "train")
test = TripleSplitDataset(data_dir, "test", stats=train.stats)
tabs = {"protein": test.prot_f, "rna": test.rna_f, "methyl": test.methy_f}
obs = {"protein": test.m_prot < 0.5, "rna": test.m_rna < 0.5, "methyl": test.m_methy < 0.5}


def feat_corr(y, yhat, mask):
    rs = []
    for j in range(y.shape[1]):
        m = mask[:, j]
        if m.sum() < 3:
            continue
        a, b = y[m, j], yhat[m, j]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(rs)) if rs else float("nan")


def sd_ratio(y, yhat, mask):
    rr = []
    for j in range(y.shape[1]):
        m = mask[:, j]
        if m.sum() < 3:
            continue
        sy = y[m, j].std()
        if sy < 1e-8:
            continue
        rr.append(yhat[m, j].std() / sy)
    return float(np.nanmean(rr)) if rr else float("nan")


for spec in specs:
    name, path = spec.split("=", 1)
    model = load_nmf_tf(path, device)
    geff = model.effective_gamma().detach().cpu().numpy()
    gmap = {m: float(geff[i]) for i, m in enumerate(MODS)}
    print(f"\n=== {name} ===")
    print("  effective gamma: " + ", ".join(f"{m}={gmap[m]:.3f}" for m in MODS))
    print(f"  {'modality':8s} {'z-RMSE':>8s} {'feat_r':>8s} {'sd_ratio':>9s}")
    for tgt in MODS:
        hat = predict_nmf_tf(model, tabs, device, missing=tgt)[tgt]
        y = tabs[tgt]
        m = obs[tgt]
        z = float(np.sqrt(np.mean((y[m] - hat[m]) ** 2)))
        fr = feat_corr(y, hat, m)
        sr = sd_ratio(y, hat, m)
        print(f"  {tgt:8s} {z:8.4f} {fr:8.4f} {sr:9.4f}")
