#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRCA 631 스플릿, 칸 결측 MCAR/MNAR + 블록 결측.

같은 마스크에서 mean, TOBMI, Ridge, MOCHI-v5, 공식 OmicsNMF/OmiTrans,
PIMMS-DAE, MIMIR을 비교한다. 채점은 가린 칸만.
MNAR은 MIMIR과 같이 낮은 값에 가중.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_gate import PAIRS_1TO1, PAIRS_2TO1, load_v5
from eval_survival import fit_ridges, load_official, tobmi_predict, SRC_1TO1, SRC_2TO1
from eval_v5_noise import (
    _get, _obs, _stats, load_1to1, masked_metrics, predict_2to1,
)
from mimir_wrap import frames_from_dir, load_mimir, predict_block, predict_values, train_mimir
from missingness import block_mask, mcar_mask, mnar_mask
from pimms_wrap import PimmsDAE
from train_gate import TripleSplitDataset

MODS = ("protein", "rna", "methyl")
RATES = (0.1, 0.3, 0.5)
SEEDS = (0, 1, 2)


def pack(tabs, dims=None):
    return np.concatenate([tabs[m] for m in MODS], 1)


def unpack(X, dims):
    out, i = {}, 0
    for m in MODS:
        d = dims[m]
        out[m] = X[:, i:i + d]
        i += d
    return out


def to_nan(filled, orig_obs, masks):
    """번역기용 0-채움 배열을 PIMMS/MIMIR용 NaN 행렬로."""
    out = {}
    for m in MODS:
        x = filled[m].copy()
        x[~orig_obs[m]] = np.nan
        x[masks[m]] = np.nan
        out[m] = x
    return out


def dfs_from_nan(nan_tabs, index, columns):
    return {m: pd.DataFrame(nan_tabs[m], index=index, columns=columns[m]) for m in MODS}


def pred_tobmi(train_tabs, ev_tabs, setting):
    out = {}
    if setting == "1to1":
        for tgt, src in PAIRS_1TO1:
            out[tgt] = tobmi_predict(train_tabs[src], train_tabs[tgt], ev_tabs[src])
    else:
        for tgt, srcs in PAIRS_2TO1:
            Xtr = np.concatenate([train_tabs[s] for s in srcs], 1)
            Xev = np.concatenate([ev_tabs[s] for s in srcs], 1)
            out[tgt] = tobmi_predict(Xtr, train_tabs[tgt], Xev)
    return out


def pred_ridge(ridges, ev_tabs, setting):
    out = {}
    if setting == "1to1":
        for tgt, src in PAIRS_1TO1:
            out[tgt] = ridges[tgt].predict(ev_tabs[src])
    else:
        for tgt, srcs in PAIRS_2TO1:
            X = np.concatenate([ev_tabs[s] for s in srcs], 1)
            out[tgt] = ridges[tgt].predict(X)
    return out


def score_method(name, setting, mechanism, split, rate, seed, ytrue, yhat, masks, stats):
    rows = []
    per = {}
    for m in MODS:
        met = masked_metrics(ytrue[m], yhat[m], masks[m], *stats[m])
        per[m] = met
        rows.append({
            "method": name, "setting": setting, "mechanism": mechanism,
            "split": split, "rate": rate, "seed": seed, "modality": m, **met,
        })
    scored = [m for m in MODS if per[m]["n_cells"] == per[m]["n_cells"] and per[m]["n_cells"] > 0]
    if scored:
        avg = {k: float(np.nanmean([per[m][k] for m in scored])) for k in ("mae", "rmse", "r2", "z_rmse")}
        avg["n_cells"] = int(sum(per[m]["n_cells"] for m in scored))
    else:
        avg = {k: np.nan for k in ("mae", "rmse", "r2", "z_rmse")}
        avg["n_cells"] = 0
    rows.append({
        "method": name, "setting": setting, "mechanism": mechanism,
        "split": split, "rate": rate, "seed": seed, "modality": "avg", **avg,
    })
    return rows


def make_cell_masks(ytrue, obs, mechanism, rate, rng):
    masks = {}
    for m in MODS:
        if mechanism == "mcar":
            masks[m] = mcar_mask(obs[m], rate, rng)
        elif mechanism == "mnar":
            masks[m] = mnar_mask(ytrue[m], obs[m], rate, rng)
        else:
            raise ValueError(mechanism)
    return masks


def apply_zero(ytrue, masks):
    return {m: np.where(masks[m], 0.0, ytrue[m]).astype(np.float32) for m in MODS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--v5_ckpt", default="/home/dyan/nmf/mochi_code/results/current/gate_tri_v5/tri_best.ckpt")
    ap.add_argument("--v5_1to1", default="/home/dyan/nmf/mochi_code/results/current/gate_compare/mochi_1to1.ckpt")
    ap.add_argument("--official_dir", default="/home/dyan/nmf/mochi_code/results/current/official")
    ap.add_argument("--mimir_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_mcar_mnar")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip_block", action="store_true")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("protocol: cell MCAR / MNAR (low-value rank) + optional block; score masked cells only")

    train = TripleSplitDataset(args.data_dir, "train")
    splits = {
        "val": TripleSplitDataset(args.data_dir, "val", stats=train.stats),
        "test": TripleSplitDataset(args.data_dir, "test", stats=train.stats),
    }
    dim_r, dim_p, dim_m = train.rna_f.shape[1], train.prot_f.shape[1], train.methy_f.shape[1]
    dims = {"protein": dim_p, "rna": dim_r, "methyl": dim_m}
    print(f"n train/val/test={len(train)}/{len(splits['val'])}/{len(splits['test'])}  "
          f"R/P/M={dim_r}/{dim_p}/{dim_m}")

    train_tabs = {m: _get(train, m) for m in MODS}
    tr_frames, stats_df, _ = frames_from_dir(args.data_dir, "train")
    split_frames = {
        "val": frames_from_dir(args.data_dir, "val", stats=stats_df)[0],
        "test": frames_from_dir(args.data_dir, "test", stats=stats_df)[0],
    }
    columns = {m: tr_frames[m].columns for m in MODS}

    print("fitting Ridge...")
    ridge1, ridge2 = fit_ridges(train)
    print("loading MOCHI / official...")
    mochi2 = load_v5(args.v5_ckpt, dim_r, dim_p, dim_m, device)
    mochi1 = load_1to1(args.v5_1to1, dim_r, dim_p, dim_m, device)
    onmf, ot = load_official(args.official_dir, dim_r, dim_p, dim_m, device)

    print("fitting PIMMS-DAE on concat train...")
    pimms = PimmsDAE(device=device)
    Xtr = pack({m: tr_frames[m].to_numpy(np.float32) for m in MODS})
    Xva = pack({m: split_frames["val"][m].to_numpy(np.float32) for m in MODS})
    pimms.fit(Xtr, Xva)

    mimir_ckpt = Path(args.mimir_dir) / "shared_best.pt"
    if mimir_ckpt.exists():
        print(f"loading MIMIR from {mimir_ckpt}")
        mimir, mimir_mv = load_mimir(args.mimir_dir, device)
    else:
        print("training MIMIR (no checkpoint)...")
        mimir, mimir_mv = train_mimir(args.data_dir, args.mimir_dir, device)

    rows = []

    def run_methods(mechanism, split, rate, seed, ytrue, obs, stats, masks, index):
        filled = apply_zero(ytrue, masks)
        nan_tabs = to_nan(ytrue, obs, masks)
        local = []
        local += score_method("mean", "none", mechanism, split, rate, seed, ytrue,
                              {m: np.zeros_like(ytrue[m]) for m in MODS}, masks, stats)
        local += score_method("TOBMI", "1to1", mechanism, split, rate, seed, ytrue,
                              pred_tobmi(train_tabs, filled, "1to1"), masks, stats)
        local += score_method("TOBMI", "2to1", mechanism, split, rate, seed, ytrue,
                              pred_tobmi(train_tabs, filled, "2to1"), masks, stats)
        local += score_method("Ridge", "1to1", mechanism, split, rate, seed, ytrue,
                              pred_ridge(ridge1, filled, "1to1"), masks, stats)
        local += score_method("Ridge", "2to1", mechanism, split, rate, seed, ytrue,
                              pred_ridge(ridge2, filled, "2to1"), masks, stats)
        local += score_method("MOCHI-v5", "1to1", mechanism, split, rate, seed, ytrue,
                              predict_2to1(mochi1, filled["rna"], filled["protein"],
                                           filled["methyl"], device), masks, stats)
        local += score_method("MOCHI-v5", "2to1", mechanism, split, rate, seed, ytrue,
                              predict_2to1(mochi2, filled["rna"], filled["protein"],
                                           filled["methyl"], device), masks, stats)
        onmf_hat = {tgt: onmf[tgt].predict_z(filled[SRC_1TO1[tgt]]) for tgt in MODS}
        ot_hat = {tgt: ot[tgt].predict(filled[SRC_1TO1[tgt]]) for tgt in MODS}
        local += score_method("OmicsNMF-official", "1to1", mechanism, split, rate, seed,
                              ytrue, onmf_hat, masks, stats)
        local += score_method("OmiTrans-official", "1to1", mechanism, split, rate, seed,
                              ytrue, ot_hat, masks, stats)

        xhat = unpack(pimms.transform(pack(nan_tabs)), dims)
        local += score_method("PIMMS-DAE", "concat", mechanism, split, rate, seed,
                              ytrue, xhat, masks, stats)

        corrupted_dfs = dfs_from_nan(nan_tabs, index, columns)
        mimir_hat = predict_values(mimir, mimir_mv, corrupted_dfs, device)
        local += score_method("MIMIR", "shared", mechanism, split, rate, seed,
                              ytrue, mimir_hat, masks, stats)
        return local

    for split, ds in splits.items():
        ytrue = {m: _get(ds, m) for m in MODS}
        obs = {m: _obs(ds, m) for m in MODS}
        stats = {m: _stats(ds, m) for m in MODS}
        index = split_frames[split]["rna"].index
        for mechanism in ("mcar", "mnar"):
            for rate in RATES:
                for seed in SEEDS:
                    rng = np.random.default_rng(20_000 + int(1000 * rate) + seed + (0 if mechanism == "mcar" else 17))
                    masks = make_cell_masks(ytrue, obs, mechanism, rate, rng)
                    rows += run_methods(mechanism, split, rate, seed, ytrue, obs, stats, masks, index)
                    print(f"[{split}] {mechanism} rate={rate:.0%} seed={seed} done")

        if not args.skip_block:
            for tgt in MODS:
                masks = {m: block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool)
                         for m in MODS}
                filled = apply_zero(ytrue, masks)
                nan_tabs = to_nan(ytrue, obs, masks)
                present = {m: pd.DataFrame(nan_tabs[m], index=index, columns=columns[m])
                           for m in MODS if m != tgt}
                block_rows = []
                block_rows += score_method("mean", "none", "block", split, 1.0, 0, ytrue,
                                           {m: np.zeros_like(ytrue[m]) for m in MODS}, masks, stats)
                block_rows += score_method("TOBMI", "1to1", "block", split, 1.0, 0, ytrue,
                                           pred_tobmi(train_tabs, filled, "1to1"), masks, stats)
                block_rows += score_method("TOBMI", "2to1", "block", split, 1.0, 0, ytrue,
                                           pred_tobmi(train_tabs, filled, "2to1"), masks, stats)
                block_rows += score_method("Ridge", "1to1", "block", split, 1.0, 0, ytrue,
                                           pred_ridge(ridge1, filled, "1to1"), masks, stats)
                block_rows += score_method("Ridge", "2to1", "block", split, 1.0, 0, ytrue,
                                           pred_ridge(ridge2, filled, "2to1"), masks, stats)
                block_rows += score_method("MOCHI-v5", "1to1", "block", split, 1.0, 0, ytrue,
                                           predict_2to1(mochi1, filled["rna"], filled["protein"],
                                                        filled["methyl"], device), masks, stats)
                block_rows += score_method("MOCHI-v5", "2to1", "block", split, 1.0, 0, ytrue,
                                           predict_2to1(mochi2, filled["rna"], filled["protein"],
                                                        filled["methyl"], device), masks, stats)
                onmf_hat = {t: onmf[t].predict_z(filled[SRC_1TO1[t]]) for t in MODS}
                ot_hat = {t: ot[t].predict(filled[SRC_1TO1[t]]) for t in MODS}
                block_rows += score_method("OmicsNMF-official", "1to1", "block", split, 1.0, 0,
                                           ytrue, onmf_hat, masks, stats)
                block_rows += score_method("OmiTrans-official", "1to1", "block", split, 1.0, 0,
                                           ytrue, ot_hat, masks, stats)
                xhat = unpack(pimms.transform(pack(nan_tabs)), dims)
                block_rows += score_method("PIMMS-DAE", "concat", "block", split, 1.0, 0,
                                           ytrue, xhat, masks, stats)
                mimir_hat = {m: filled[m].copy() for m in MODS}
                mimir_hat[tgt] = predict_block(
                    mimir, mimir_mv, present, tgt, columns[tgt], index, device)
                block_rows += score_method("MIMIR", "shared", "block", split, 1.0, 0,
                                           ytrue, mimir_hat, masks, stats)
                for r in block_rows:
                    r["missing"] = tgt
                rows += block_rows
                print(f"[{split}] block missing-{tgt} done")

    raw = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "eval_raw.tsv", sep="\t", index=False, float_format="%.6f")

    g = ["method", "setting", "mechanism", "split", "rate", "missing", "modality"]
    keep = raw[raw["n_cells"].fillna(0) > 0].copy()
    if "missing" not in keep.columns:
        keep["missing"] = "cell"
    keep["missing"] = keep["missing"].fillna("cell")
    summary = (keep.groupby(g, dropna=False)
               .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"),
                    z_rmse=("z_rmse", "mean"), z_rmse_sd=("z_rmse", "std"),
                    n_cells=("n_cells", "mean"))
               .reset_index())
    summary.to_csv(out / "eval_summary.tsv", sep="\t", index=False, float_format="%.4f")

    show = summary[(summary["modality"] == "avg") & (summary["split"] == "test")]
    show = show.sort_values(["mechanism", "rate", "z_rmse"])
    print("\n=== test avg z-RMSE (seed mean) ===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {out / 'eval_summary.tsv'}")


if __name__ == "__main__":
    main()
