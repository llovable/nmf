#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mean / Ridge / MOCHI-v5 / MOCHI-shared / NMF-Transformer / MIMIR 비교."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baselines_self import RidgeSelf, all_self_predictions, choose_rank_on_val
from compare_gate import load_v5
from eval_mcar_mnar import (
    MODS, RATES, SEEDS, apply_zero, make_cell_masks, pred_ridge, score_method,
)
from eval_survival import fit_ridges
from eval_v5_noise import _get, _obs, _stats, predict_2to1
from mimir_wrap import frames_from_dir, load_mimir, predict_block, predict_values
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from models_shared import load_shared, predict_shared
from train_gate import TripleSplitDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--v5_ckpt", default="/home/dyan/nmf/mochi_code/results/current/gate_tri_v5/tri_best.ckpt")
    ap.add_argument("--mimir_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--shared_ckpt",
                    default="/home/dyan/nmf/mochi_code/results/current/gate_shared_twopath_mean/shared_best.ckpt")
    ap.add_argument("--nmf_tf_ckpt",
                    default="/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf/nmf_tf_best.ckpt")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="자기정보 베이스라인의 마스킹 증강·rank 선택 시드")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    splits = {
        "val": TripleSplitDataset(args.data_dir, "val", stats=train.stats),
        "test": TripleSplitDataset(args.data_dir, "test", stats=train.stats),
    }
    dims = {"rna": train.rna_f.shape[1], "protein": train.prot_f.shape[1],
            "methyl": train.methy_f.shape[1]}
    print(f"device={device} n={len(train)}/{len(splits['val'])}/{len(splits['test'])}")

    print("fitting Ridge...")
    ridge1, ridge2 = fit_ridges(train)

    # --- 타깃 자기 정보를 쓰는 베이스라인 (MOCHI의 ω=10과 같은 정보에 접근) ---
    train_tabs = {m: _get(train, m) for m in MODS}
    train_obs = {m: _obs(train, m) for m in MODS}
    print("fitting Ridge-self (all three omics as input, mask-augmented)...")
    ridge_self = RidgeSelf(seed=args.seed).fit(train_tabs, train_obs)
    val_tabs = {m: _get(splits["val"], m) for m in MODS}
    val_obs = {m: _obs(splits["val"], m) for m in MODS}
    si_rank, si_scores = choose_rank_on_val(val_tabs, val_obs, seed=args.seed,
                                            anchor_tabs=train_tabs)
    print(f"softImpute rank chosen on val: {si_rank}  (val z-RMSE by rank: "
          + ", ".join(f"{r}:{s:.4f}" for r, s in si_scores.items()) + ")")
    print("loading v5 / MIMIR / shared / NMF-TF...")
    mochi2 = None
    if Path(args.v5_ckpt).exists():
        mochi2 = load_v5(args.v5_ckpt, dims["rna"], dims["protein"], dims["methyl"], device)
    else:
        print(f"skip MOCHI-v5: {args.v5_ckpt} 없음")
    mimir = mimir_mv = None
    if args.mimir_dir and (Path(args.mimir_dir) / "shared_best.pt").exists():
        mimir, mimir_mv = load_mimir(args.mimir_dir, device)
    else:
        print(f"skip MIMIR: {args.mimir_dir} 에 shared_best.pt 없음")
    shared = None
    shared_tag = "mean-z"
    if Path(args.shared_ckpt).exists():
        shared = load_shared(args.shared_ckpt, dims, device)
        shared_tag = f"{getattr(shared, 'fuse_name', 'mean')}-z"
    ckpt = Path(args.nmf_tf_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"train_nmf_tf.py 먼저 실행: {ckpt}")
    nmftf = load_nmf_tf(ckpt, device)

    tr_frames, stats_df, _ = frames_from_dir(args.data_dir, "train")
    split_frames = {
        "val": frames_from_dir(args.data_dir, "val", stats=stats_df)[0],
        "test": frames_from_dir(args.data_dir, "test", stats=stats_df)[0],
    }
    columns = {m: tr_frames[m].columns for m in MODS}

    rows = []
    for split, ds in splits.items():
        ytrue = {m: _get(ds, m) for m in MODS}
        obs = {m: _obs(ds, m) for m in MODS}
        stats = {m: _stats(ds, m) for m in MODS}
        index = split_frames[split]["rna"].index
        for mechanism in ("mcar", "mnar"):
            for rate in RATES:
                for seed in SEEDS:
                    rng = np.random.default_rng(
                        20_000 + int(1000 * rate) + seed + (0 if mechanism == "mcar" else 17))
                    masks = make_cell_masks(ytrue, obs, mechanism, rate, rng)
                    filled = apply_zero(ytrue, masks)
                    nan_tabs = {}
                    for m in MODS:
                        x = ytrue[m].copy()
                        x[~obs[m]] = np.nan
                        x[masks[m]] = np.nan
                        nan_tabs[m] = x
                    rows += score_method("mean", "none", mechanism, split, rate, seed, ytrue,
                                         {m: np.zeros_like(ytrue[m]) for m in MODS}, masks, stats)
                    rows += score_method("Ridge", "1to1", mechanism, split, rate, seed, ytrue,
                                         pred_ridge(ridge1, filled, "1to1"), masks, stats)
                    rows += score_method("Ridge", "2to1", mechanism, split, rate, seed, ytrue,
                                         pred_ridge(ridge2, filled, "2to1"), masks, stats)
                    if mochi2 is not None:
                        rows += score_method("MOCHI-v5", "2to1", mechanism, split, rate, seed, ytrue,
                                             predict_2to1(mochi2, filled["rna"], filled["protein"],
                                                          filled["methyl"], device), masks, stats)
                    if shared is not None:
                        rows += score_method("MOCHI-shared", shared_tag, mechanism, split, rate, seed,
                                             ytrue, predict_shared(shared, filled, device, missing=None),
                                             masks, stats)
                    rows += score_method("MOCHI-NMFTF", "nmf-tf", mechanism, split, rate, seed, ytrue,
                                         predict_nmf_tf(nmftf, filled, device, missing=None),
                                         masks, stats)
                    if mimir is not None:
                        dfs = {m: pd.DataFrame(nan_tabs[m], index=index, columns=columns[m]) for m in MODS}
                        rows += score_method("MIMIR", "shared", mechanism, split, rate, seed, ytrue,
                                             predict_values(mimir, mimir_mv, dfs, device),
                                             masks, stats)
                    for nm, hat in all_self_predictions(
                            nan_tabs, ridge_self=ridge_self, filled_tabs=filled,
                            si_rank=si_rank, anchor_tabs=train_tabs).items():
                        rows += score_method(nm, "self-info", mechanism, split, rate, seed,
                                             ytrue, hat, masks, stats)
                    print(f"[{split}] {mechanism} {rate:.0%} seed={seed}")

        for tgt in MODS:
            masks = {m: block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool)
                     for m in MODS}
            filled = apply_zero(ytrue, masks)
            nan_tabs = {}
            for m in MODS:
                x = ytrue[m].copy()
                x[~obs[m]] = np.nan
                x[masks[m]] = np.nan
                nan_tabs[m] = x
            present = {m: pd.DataFrame(nan_tabs[m], index=index, columns=columns[m])
                       for m in MODS if m != tgt}
            block = []
            block += score_method("mean", "none", "block", split, 1.0, 0, ytrue,
                                  {m: np.zeros_like(ytrue[m]) for m in MODS}, masks, stats)
            block += score_method("Ridge", "1to1", "block", split, 1.0, 0, ytrue,
                                  pred_ridge(ridge1, filled, "1to1"), masks, stats)
            block += score_method("Ridge", "2to1", "block", split, 1.0, 0, ytrue,
                                  pred_ridge(ridge2, filled, "2to1"), masks, stats)
            if mochi2 is not None:
                block += score_method("MOCHI-v5", "2to1", "block", split, 1.0, 0, ytrue,
                                      predict_2to1(mochi2, filled["rna"], filled["protein"],
                                                   filled["methyl"], device), masks, stats)
            if shared is not None:
                block += score_method("MOCHI-shared", shared_tag, "block", split, 1.0, 0, ytrue,
                                      predict_shared(shared, filled, device, missing=tgt),
                                      masks, stats)
            block += score_method("MOCHI-NMFTF", "nmf-tf", "block", split, 1.0, 0, ytrue,
                                  predict_nmf_tf(nmftf, filled, device, missing=tgt),
                                  masks, stats)
            if mimir is not None:
                mh = {m: filled[m].copy() for m in MODS}
                mh[tgt] = predict_block(mimir, mimir_mv, present, tgt, columns[tgt], index, device)
                block += score_method("MIMIR", "shared", "block", split, 1.0, 0, ytrue, mh, masks, stats)
            # 블록에서 오믹스별 KNN/softImpute는 열 평균(z=0)으로 퇴화한다. 그대로 보고한다:
            # '블록 결측은 교차 오믹스 없이 풀리지 않는다'는 직접 근거가 된다.
            for nm, hat in all_self_predictions(
                    nan_tabs, ridge_self=ridge_self, filled_tabs=filled,
                    si_rank=si_rank, anchor_tabs=train_tabs).items():
                block += score_method(nm, "self-info", "block", split, 1.0, 0,
                                      ytrue, hat, masks, stats)
            for r in block:
                r["missing"] = tgt
            rows += block
            print(f"[{split}] block missing-{tgt}")

    raw = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "eval_raw.tsv", sep="\t", index=False, float_format="%.6f")
    keep = raw[raw["n_cells"].fillna(0) > 0].copy()
    keep["missing"] = keep["missing"].fillna("cell") if "missing" in keep.columns else "cell"
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
