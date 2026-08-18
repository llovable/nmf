#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5 기준, 종양 631명 split 위에서
가우시안 노이즈 + 칸 마스킹 10/30/50% 평가.

채점 칸만 본다. 관측 칸·원래 NaN은 제외.
2→1 / 1→1, MAE·RMSE·R²(원척도) + z-RMSE(z-점수).
Ridge·mean은 같은 마스크에서 같이 본다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader

from compare_gate import (
    PAIRS_1TO1, PAIRS_2TO1, RIDGE_ALPHAS,
    make_mochi_1to1, make_mochi_2to1, load_v5,
)
from models import ConditionalCritic
from train_gate import TripleSplitDataset, WGAN_GP_Trainer_Tri

RATES = (0.1, 0.3, 0.5)
SEEDS = (0, 1, 2)
MODS = ("protein", "rna", "methyl")


def _get(ds, name):
    return {"rna": ds.rna_f, "protein": ds.prot_f, "methyl": ds.methy_f}[name]


def _obs(ds, name):
    # Dataset stores m_* as 1 = originally missing
    miss = {"rna": ds.m_rna, "protein": ds.m_prot, "methyl": ds.m_methy}[name]
    return miss < 0.5


def _stats(ds, name):
    key = {"rna": "rna", "protein": "prot", "methyl": "methy"}[name]
    mu, sd = ds.stats[key]
    return np.asarray(mu, dtype=np.float32), np.asarray(sd, dtype=np.float32)


def to_raw(z, mu, sd):
    return np.asarray(z, dtype=np.float32) * sd + mu


def mcar_mask(obs, rate, rng):
    cand = np.flatnonzero(obs.ravel())
    n = int(len(cand) * rate)
    chosen = rng.choice(cand, size=max(n, 1), replace=False) if n else np.array([], dtype=int)
    mask = np.zeros(obs.shape, dtype=bool)
    if len(chosen):
        mask.ravel()[chosen] = True
    return mask


def corrupt(X, obs, rate, rng):
    """관측 칸에 N(0, rate²) 노이즈, rate 비율은 0으로 가림."""
    eval_mask = mcar_mask(obs, rate, rng)
    noise = rng.normal(0.0, rate, size=X.shape).astype(np.float32)
    Xc = X + noise * obs.astype(np.float32)
    Xc = np.where(eval_mask, 0.0, Xc).astype(np.float32)
    return Xc, eval_mask


def masked_metrics(y_z, yhat_z, mask, mu, sd):
    if mask.sum() == 0:
        return {k: np.nan for k in ("mae", "rmse", "r2", "z_rmse", "n_cells")}
    yt_z, yp_z = y_z[mask], yhat_z[mask]
    yt, yp = to_raw(y_z, mu, sd)[mask], to_raw(yhat_z, mu, sd)[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan,
        "z_rmse": float(np.sqrt(np.mean((yt_z - yp_z) ** 2))),
        "n_cells": int(mask.sum()),
    }


def avg_row(per_mod):
    keys = ("mae", "rmse", "r2", "z_rmse")
    out = {k: float(np.nanmean([per_mod[m][k] for m in MODS])) for k in keys}
    out["n_cells"] = int(sum(per_mod[m]["n_cells"] for m in MODS))
    return out


@torch.no_grad()
def predict_2to1(trainer, rna, prot, methy, device):
    trainer.Gp.eval(); trainer.Gr.eval(); trainer.Gm.eval()
    n = rna.shape[0]
    yp, yr, ym = [], [], []
    for i in range(0, n, 32):
        r = torch.from_numpy(rna[i:i + 32]).to(device)
        p = torch.from_numpy(prot[i:i + 32]).to(device)
        m = torch.from_numpy(methy[i:i + 32]).to(device)
        a, b, c = trainer._forward(r, p, m)
        yp.append(a.cpu().numpy()); yr.append(b.cpu().numpy()); ym.append(c.cpu().numpy())
    return {
        "protein": np.concatenate(yp),
        "rna": np.concatenate(yr),
        "methyl": np.concatenate(ym),
    }


def load_1to1(ckpt_path, dim_r, dim_p, dim_m, device):
    Gp, Gr, Gm, Dp, Dr, Dm = make_mochi_1to1(dim_r, dim_p, dim_m)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    Gp.load_state_dict(ckpt["Gp"])
    Gr.load_state_dict(ckpt["Gr"])
    Gm.load_state_dict(ckpt["Gm"])
    tr = WGAN_GP_Trainer_Tri(Gp, Gr, Gm, Dp, Dr, Dm, device)

    class T1(WGAN_GP_Trainer_Tri):
        def _conds(self, rna, prot, methy):
            return rna, prot, rna

        def _forward(self, rna, prot, methy):
            return self.Gp(rna, src=None), self.Gr(prot, src=None), self.Gm(rna, src=None)

    t = T1(Gp, Gr, Gm, Dp, Dr, Dm, device)
    return t


def fit_ridges(train):
    r1, r2 = {}, {}
    for tgt, src in PAIRS_1TO1:
        r1[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(_get(train, src), _get(train, tgt))
    for tgt, srcs in PAIRS_2TO1:
        X = np.concatenate([_get(train, s) for s in srcs], 1)
        r2[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, _get(train, tgt))
    return r1, r2


def pred_ridge_1(r1, corrupted):
    return {tgt: r1[tgt].predict(corrupted[src]) for tgt, src in PAIRS_1TO1}


def pred_ridge_2(r2, corrupted):
    out = {}
    for tgt, srcs in PAIRS_2TO1:
        X = np.concatenate([corrupted[s] for s in srcs], 1)
        out[tgt] = r2[tgt].predict(X)
    return out


def score_method(name, setting, split, rate, seed, ytrue, yhat, masks, stats):
    rows = []
    per = {}
    for m in MODS:
        met = masked_metrics(ytrue[m], yhat[m], masks[m], *stats[m])
        per[m] = met
        rows.append({"method": name, "setting": setting, "split": split,
                     "rate": rate, "seed": seed, "modality": m, **met})
    avg = avg_row(per)
    rows.append({"method": name, "setting": setting, "split": split,
                 "rate": rate, "seed": seed, "modality": "avg", **avg})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--v5_ckpt", default="/home/dyan/nmf/mochi_code/results/current/gate_tri_v5/tri_best.ckpt")
    ap.add_argument("--v5_1to1", default="/home/dyan/nmf/mochi_code/results/current/gate_compare/mochi_1to1.ckpt")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_v5_noise")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("protocol: MCAR cell mask + N(0, rate^2) on observed z-cells; score masked cells only")

    train = TripleSplitDataset(args.data_dir, "train")
    splits = {
        "val": TripleSplitDataset(args.data_dir, "val", stats=train.stats),
        "test": TripleSplitDataset(args.data_dir, "test", stats=train.stats),
    }
    dim_r, dim_p, dim_m = train.rna_f.shape[1], train.prot_f.shape[1], train.methy_f.shape[1]
    print(f"n train/val/test={len(train)}/{len(splits['val'])}/{len(splits['test'])}  R/P/M={dim_r}/{dim_p}/{dim_m}")

    print("fitting Ridge on clean train...")
    ridge1, ridge2 = fit_ridges(train)
    tri = load_v5(args.v5_ckpt, dim_r, dim_p, dim_m, device)
    mochi1 = load_1to1(args.v5_1to1, dim_r, dim_p, dim_m, device)

    rows = []
    for split, ds in splits.items():
        ytrue = {m: _get(ds, m) for m in MODS}
        obs = {m: _obs(ds, m) for m in MODS}
        stats = {m: _stats(ds, m) for m in MODS}
        for rate in RATES:
            for seed in SEEDS:
                rng = np.random.default_rng(10_000 + int(1000 * rate) + seed)
                corrupted, masks = {}, {}
                for m in MODS:
                    corrupted[m], masks[m] = corrupt(ytrue[m], obs[m], rate, rng)

                mean_hat = {m: np.zeros_like(ytrue[m]) for m in MODS}
                rows += score_method("mean", "none", split, rate, seed, ytrue, mean_hat, masks, stats)
                rows += score_method("Ridge", "1to1", split, rate, seed, ytrue,
                                     pred_ridge_1(ridge1, corrupted), masks, stats)
                rows += score_method("Ridge", "2to1", split, rate, seed, ytrue,
                                     pred_ridge_2(ridge2, corrupted), masks, stats)
                rows += score_method("MOCHI-v5", "1to1", split, rate, seed, ytrue,
                                     predict_2to1(mochi1, corrupted["rna"], corrupted["protein"],
                                                  corrupted["methyl"], device),
                                     masks, stats)
                rows += score_method("MOCHI-v5", "2to1", split, rate, seed, ytrue,
                                     predict_2to1(tri, corrupted["rna"], corrupted["protein"],
                                                  corrupted["methyl"], device),
                                     masks, stats)
                print(f"[{split}] rate={rate:.0%} seed={seed} done")

    raw = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "eval_raw.tsv", sep="\t", index=False, float_format="%.6f")

    g = ["method", "setting", "split", "rate", "modality"]
    summary = (raw.groupby(g)
               .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"),
                    z_rmse=("z_rmse", "mean"), z_rmse_sd=("z_rmse", "std"),
                    n_cells=("n_cells", "mean"))
               .reset_index())
    summary.to_csv(out / "eval_summary.tsv", sep="\t", index=False, float_format="%.4f")

    show = summary[(summary["modality"] == "avg")].sort_values(["split", "rate", "z_rmse"])
    print("\n=== avg over protein/rna/methyl (seed mean) ===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {out / 'eval_summary.tsv'}")


if __name__ == "__main__":
    main()
