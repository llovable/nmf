#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
적대항·Transformer가 이득을 내는 조건 탐색.

세 가지 스트레스 축을 본다.
  1) 극단 칸 결측률 0.7 / 0.9 (MCAR, MNAR)
  2) 이중 블록 결측 — 한 오믹스만 남기고 나머지 둘을 채운다
  3) 분포 보존 — 분산비, 특징별 평균 이동, 상관 구조, KS 통계량

채점은 재구성 정확도(z-RMSE)와 분포 지표를 함께 낸다.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.linear_model import RidgeCV

from compare_gate import RIDGE_ALPHAS
from eval_mcar_mnar import MODS, apply_zero, make_cell_masks, score_method
from eval_v5_noise import _get, _obs, _stats
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset

HARD_RATES = (0.7, 0.9)
SEEDS = (0, 1, 2)
N_DIST_FEAT = 300


def fit_all_pairs(train):
    """단일 소스 -> 타깃 6쌍 + 이중 소스 3쌍."""
    single, double = {}, {}
    for tgt, src in itertools.permutations(MODS, 2):
        single[(tgt, src)] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(_get(train, src), _get(train, tgt))
    for tgt in MODS:
        srcs = tuple(m for m in MODS if m != tgt)
        X = np.concatenate([_get(train, s) for s in srcs], 1)
        double[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, _get(train, tgt))
    return single, double


@torch.no_grad()
def predict_blocks(model, tabs, device, missing, batch_size=64):
    """missing 안의 모든 오믹스를 남은 오믹스로부터 채운다."""
    missing = tuple(missing)
    present_mods = [m for m in MODS if m not in missing]
    n = tabs[present_mods[0]].shape[0]
    acc = {m: [] for m in MODS}
    model.eval()
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        xs, present = {}, {}
        b = None
        for m in present_mods:
            t = torch.from_numpy(np.asarray(tabs[m][sl], dtype=np.float32)).to(device)
            xs[m] = t
            present[m] = torch.ones(t.size(0), dtype=torch.bool, device=device)
            b = t.size(0)
        for m in missing:
            present[m] = torch.zeros(b, dtype=torch.bool, device=device)
        hat = {}
        for m in present_mods:
            hat[m] = model.decoders[m](model.encoders[m](xs[m]))
        for m in missing:
            hat[m] = model.loo_reconstruct(xs, present, m)
        for m in MODS:
            acc[m].append(hat[m].cpu().numpy())
    return {m: np.concatenate(acc[m], 0) for m in MODS}


def dist_metrics(ytrue, yhat, rng):
    """z 공간에서 분포 보존을 잰다. 값은 특징 축으로 집계한다."""
    f = ytrue.shape[1]
    idx = rng.choice(f, size=min(N_DIST_FEAT, f), replace=False)
    a, b = ytrue[:, idx], yhat[:, idx]
    sd_t, sd_p = a.std(0), b.std(0)
    corr_t = np.corrcoef(a, rowvar=False)
    corr_p = np.corrcoef(b, rowvar=False)
    corr_t = np.nan_to_num(corr_t)
    corr_p = np.nan_to_num(corr_p)
    denom = np.linalg.norm(corr_t) or 1.0
    ks = [ks_2samp(a[:, j], b[:, j]).statistic for j in range(a.shape[1])]
    return {
        "sd_ratio": float(np.mean(sd_p) / max(1e-8, float(np.mean(sd_t)))),
        "mean_shift": float(np.mean(np.abs(b.mean(0) - a.mean(0)))),
        "corr_err": float(np.linalg.norm(corr_p - corr_t) / denom),
        "ks": float(np.mean(ks)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mimir_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_stress")
    ap.add_argument("--runs", nargs="+", default=[
        "MOCHI=/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt",
        "MOCHI-nogan=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nogan/nmf_tf_best.ckpt",
        "MOCHI-notf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_mean/nmf_tf_best.ckpt",
        "MOCHI-nonmf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nonmf/nmf_tf_best.ckpt",
    ])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    test = TripleSplitDataset(args.data_dir, "test", stats=train.stats)
    print(f"device={device} n_train={len(train)} n_test={len(test)}")

    ytrue = {m: _get(test, m) for m in MODS}
    obs = {m: _obs(test, m) for m in MODS}
    stats = {m: _stats(test, m) for m in MODS}

    print("fitting Ridge (6 single + 3 double)...")
    ridge_s, ridge_d = fit_all_pairs(train)

    models = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            models[name] = load_nmf_tf(path, device)
        else:
            print(f"skip {name}: {path} 없음")

    mimir = mimir_mv = None
    columns = index = None
    if Path(args.mimir_dir).exists():
        from mimir_wrap import frames_from_dir, load_mimir, predict_block, predict_values
        mimir, mimir_mv = load_mimir(args.mimir_dir, device)
        tr_frames, stats_df, _ = frames_from_dir(args.data_dir, "train")
        te_frames = frames_from_dir(args.data_dir, "test", stats=stats_df)[0]
        columns = {m: tr_frames[m].columns for m in MODS}
        index = te_frames["rna"].index

    def to_nan(masks):
        out = {}
        for m in MODS:
            x = ytrue[m].copy()
            x[~obs[m]] = np.nan
            x[masks[m]] = np.nan
            out[m] = x
        return out

    rows = []

    # ---- 1) 극단 칸 결측률 ----
    for mechanism in ("mcar", "mnar"):
        for rate in HARD_RATES:
            for seed in SEEDS:
                rng = np.random.default_rng(
                    30_000 + int(1000 * rate) + seed + (0 if mechanism == "mcar" else 17))
                masks = make_cell_masks(ytrue, obs, mechanism, rate, rng)
                filled = apply_zero(ytrue, masks)
                rows += score_method("mean", "none", mechanism, "test", rate, seed, ytrue,
                                     {m: np.zeros_like(ytrue[m]) for m in MODS}, masks, stats)
                r2hat = {tgt: ridge_d[tgt].predict(
                    np.concatenate([filled[s] for s in MODS if s != tgt], 1)) for tgt in MODS}
                rows += score_method("Ridge", "2to1", mechanism, "test", rate, seed, ytrue,
                                     r2hat, masks, stats)
                for name, mdl in models.items():
                    rows += score_method(name, "cell", mechanism, "test", rate, seed, ytrue,
                                         predict_nmf_tf(mdl, filled, device, missing=None),
                                         masks, stats)
                if mimir is not None:
                    dfs = {m: pd.DataFrame(to_nan(masks)[m], index=index, columns=columns[m])
                           for m in MODS}
                    rows += score_method("MIMIR", "shared", mechanism, "test", rate, seed, ytrue,
                                         predict_values(mimir, mimir_mv, dfs, device), masks, stats)
                print(f"[cell] {mechanism} rate={rate:.0%} seed={seed} done")

    # ---- 2) 이중 블록 결측 ----
    for keep in MODS:
        miss = tuple(m for m in MODS if m != keep)
        masks = {m: (block_mask(obs[m]) if m in miss else np.zeros_like(obs[m], dtype=bool))
                 for m in MODS}
        filled = apply_zero(ytrue, masks)
        local = []
        local += score_method("mean", "none", "block2", "test", 1.0, 0, ytrue,
                              {m: np.zeros_like(ytrue[m]) for m in MODS}, masks, stats)
        rhat = {m: filled[m].copy() for m in MODS}
        for m in miss:
            rhat[m] = ridge_s[(m, keep)].predict(filled[keep])
        local += score_method("Ridge", "1to1", "block2", "test", 1.0, 0, ytrue, rhat, masks, stats)
        for name, mdl in models.items():
            local += score_method(name, "block2", "block2", "test", 1.0, 0, ytrue,
                                  predict_blocks(mdl, filled, device, miss), masks, stats)
        if mimir is not None:
            present = {keep: pd.DataFrame(to_nan(masks)[keep], index=index, columns=columns[keep])}
            mh = {m: filled[m].copy() for m in MODS}
            for m in miss:
                mh[m] = predict_block(mimir, mimir_mv, present, m, columns[m], index, device)
            local += score_method("MIMIR", "shared", "block2", "test", 1.0, 0, ytrue, mh, masks, stats)
        for r in local:
            r["missing"] = "+".join(miss)
        rows += local
        print(f"[block2] keep={keep} done")

    # ---- 3) 단일 블록에서 분포 보존 ----
    dist_rows = []
    for tgt in MODS:
        masks = {m: (block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool))
                 for m in MODS}
        filled = apply_zero(ytrue, masks)
        cand = {"mean": {m: np.zeros_like(ytrue[m]) for m in MODS}}
        cand["Ridge-2to1"] = {tgt: ridge_d[tgt].predict(
            np.concatenate([filled[s] for s in MODS if s != tgt], 1))}
        for name, mdl in models.items():
            cand[name] = predict_nmf_tf(mdl, filled, device, missing=tgt)
        if mimir is not None:
            present = {m: pd.DataFrame(to_nan(masks)[m], index=index, columns=columns[m])
                       for m in MODS if m != tgt}
            cand["MIMIR"] = {tgt: predict_block(mimir, mimir_mv, present, tgt,
                                                columns[tgt], index, device)}
        for name, hat in cand.items():
            rng = np.random.default_rng(777)
            met = dist_metrics(ytrue[tgt], np.asarray(hat[tgt]), rng)
            zr = float(np.sqrt(np.mean((ytrue[tgt] - np.asarray(hat[tgt])) ** 2)))
            dist_rows.append({"method": name, "missing": tgt, "z_rmse": zr, **met})
        print(f"[dist] missing={tgt} done")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "stress_raw.tsv", sep="\t", index=False, float_format="%.6f")
    keep_rows = raw[raw["n_cells"].fillna(0) > 0].copy()
    keep_rows["missing"] = keep_rows.get("missing", pd.Series(dtype=object)).fillna("cell")
    g = ["method", "setting", "mechanism", "rate", "missing", "modality"]
    summary = (keep_rows.groupby(g, dropna=False)
               .agg(z_rmse=("z_rmse", "mean"), z_rmse_sd=("z_rmse", "std"), r2=("r2", "mean"))
               .reset_index())
    summary.to_csv(out / "stress_summary.tsv", sep="\t", index=False, float_format="%.4f")

    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out / "dist_metrics.tsv", sep="\t", index=False, float_format="%.4f")

    avg = summary[summary["modality"] == "avg"]
    print("\n=== 극단 칸 결측 (test avg z-RMSE) ===")
    print(avg[avg["mechanism"].isin(("mcar", "mnar"))]
          .pivot_table(index="method", columns=["mechanism", "rate"], values="z_rmse")
          .round(4).to_string())
    print("\n=== 이중 블록 결측 (test avg z-RMSE) ===")
    print(avg[avg["mechanism"] == "block2"]
          .pivot_table(index="method", columns="missing", values="z_rmse")
          .round(4).to_string())
    print("\n=== 분포 보존 (단일 블록, 모달리티 평균) ===")
    print(dist.groupby("method")[["z_rmse", "sd_ratio", "mean_shift", "corr_err", "ks"]]
          .mean().round(4).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
