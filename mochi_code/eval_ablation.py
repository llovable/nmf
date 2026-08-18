#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""보고 모델 ablation: NMF 토큰 / Transformer / GAN / 자기 히든 가중."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_mcar_mnar import (
    MODS, RATES, SEEDS, apply_zero, make_cell_masks, score_method,
)
from eval_v5_noise import _get, _obs, _stats
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset


def eval_one(name, setting, model, device, splits, self_weight=10.0):
    rows = []
    for split, ds in splits.items():
        ytrue = {m: _get(ds, m) for m in MODS}
        obs = {m: _obs(ds, m) for m in MODS}
        stats = {m: _stats(ds, m) for m in MODS}
        for mechanism in ("mcar", "mnar"):
            for rate in RATES:
                for seed in SEEDS:
                    rng = np.random.default_rng(
                        20_000 + int(1000 * rate) + seed + (0 if mechanism == "mcar" else 17))
                    masks = make_cell_masks(ytrue, obs, mechanism, rate, rng)
                    filled = apply_zero(ytrue, masks)
                    hat = predict_nmf_tf(model, filled, device, missing=None,
                                         self_weight=self_weight)
                    rows += score_method(name, setting, mechanism, split, rate, seed,
                                         ytrue, hat, masks, stats)
        for tgt in MODS:
            masks = {m: block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool)
                     for m in MODS}
            filled = apply_zero(ytrue, masks)
            hat = predict_nmf_tf(model, filled, device, missing=tgt)
            block = score_method(name, setting, "block", split, 1.0, 0, ytrue, hat, masks, stats)
            for r in block:
                r["missing"] = tgt
            rows += block
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_ablate")
    ap.add_argument("--runs", nargs="+", default=[
        "full=/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt",
        "nogan=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nogan/nmf_tf_best.ckpt",
        "mean=/home/dyan/nmf/mochi_code/results/current/gate_ablate_mean/nmf_tf_best.ckpt",
        "nonmf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nonmf/nmf_tf_best.ckpt",
    ])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    splits = {
        "val": TripleSplitDataset(args.data_dir, "val", stats=train.stats),
        "test": TripleSplitDataset(args.data_dir, "test", stats=train.stats),
    }
    print(f"device={device} n={len(train)}/{len(splits['val'])}/{len(splits['test'])}")

    rows = []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        ckpt = Path(path)
        if not ckpt.exists():
            print(f"skip {name}: {ckpt} 없음")
            continue
        model = load_nmf_tf(ckpt, device)
        print(f"eval {name} {ckpt}")
        if name == "full":
            rows += eval_one("MOCHI", "self-w10", model, device, splits, self_weight=10.0)
            rows += eval_one("MOCHI", "self-w0", model, device, splits, self_weight=0.0)
        else:
            rows += eval_one(f"MOCHI-{name}", "ablation", model, device, splits, self_weight=10.0)

    if not rows:
        raise SystemExit("평가할 체크포인트가 없습니다.")
    raw = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "eval_raw.tsv", sep="\t", index=False, float_format="%.6f")
    keep = raw[raw["n_cells"].fillna(0) > 0].copy()
    if "missing" not in keep.columns:
        keep["missing"] = "cell"
    keep["missing"] = keep["missing"].fillna("cell")
    g = ["method", "setting", "mechanism", "split", "rate", "missing", "modality"]
    summary = (keep.groupby(g, dropna=False)
               .agg(z_rmse=("z_rmse", "mean"), z_rmse_sd=("z_rmse", "std"),
                    r2=("r2", "mean"), n_cells=("n_cells", "mean"))
               .reset_index())
    summary.to_csv(out / "eval_summary.tsv", sep="\t", index=False, float_format="%.4f")
    show = summary[(summary["modality"] == "avg") & (summary["split"] == "test")]
    print("\n=== test avg z-RMSE ===")
    print(show.sort_values(["mechanism", "rate", "z_rmse"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {out / 'eval_summary.tsv'}")


if __name__ == "__main__":
    main()
