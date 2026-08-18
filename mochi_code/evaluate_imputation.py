#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멀티오믹스 결측치 보정 평가 (엄밀 버전)

핵심 원칙
---------
1) 마스킹된(인위적으로 가린) 셀에서만 평가한다. 관측/보존 셀은 평가에서 제외.
2) 두 가지 결측 시나리오: MCAR(원소 단위) + block-missing(모달리티 단위).
3) 동일한 마스크에서 베이스라인(평균/중앙값/KNN)과 MOCHI를 함께 비교한다.
4) feature-wise 상관(주력) + sample-wise 상관 + RMSE/MAE/NRMSE 보고.
5) 여러 시드로 반복 → 평균 ± 표준편차.
6) (권장) 샘플 단위 train/test split. test 샘플에서만 평가하여 누수를 줄인다.

사용 예
-------
# 베이스라인만 빠르게 (모델 불필요)
python evaluate_imputation.py --data_dir <DIR> --out eval_out --seeds 5

# MOCHI까지 포함 (CPU 가능, 느릴 수 있음)
python evaluate_imputation.py --data_dir <DIR> --out eval_out --ckpt <tri_best.ckpt> --use_mochi
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

MODALITIES = ["rna", "protein", "methy"]


# ----------------------------------------------------------------------
# 데이터 로딩: 파일은 [features x samples] → [samples x features]로 전치
# ----------------------------------------------------------------------
def load_omics(data_dir: str, prefix: str = "BRCA_PAM50") -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    tabs = {}
    for m in MODALITIES:
        fp = data_dir / f"{prefix}.{m}.original.tsv"
        if not fp.exists():
            raise FileNotFoundError(f"파일 없음: {fp}")
        df = pd.read_csv(fp, sep="\t", index_col=0).T  # [samples x features]
        tabs[m] = df.astype(np.float32)
    # 공통 샘플 정렬
    common = None
    for m in MODALITIES:
        common = tabs[m].index if common is None else common.intersection(tabs[m].index)
    common = sorted(common)
    for m in MODALITIES:
        tabs[m] = tabs[m].loc[common]
    print(f"공통 샘플 수: {len(common)}")
    for m in MODALITIES:
        print(f"  {m}: {tabs[m].shape} (samples x features)")
    return tabs


# ----------------------------------------------------------------------
# 마스크 생성 (True = 평가 대상으로 가린 위치). 관측치(원래 NaN)는 가리지 않음.
# ----------------------------------------------------------------------
def make_mcar_mask(shape, rate, observed_mask, rng) -> np.ndarray:
    """observed_mask: True=관측됨(가릴 수 있음). 원래 NaN은 False."""
    cand = observed_mask
    n_cand = cand.sum()
    n_mask = int(n_cand * rate)
    flat = np.flatnonzero(cand.ravel())
    chosen = rng.choice(flat, size=n_mask, replace=False)
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[chosen] = True
    return mask


def make_block_mask(shape, frac_samples, observed_mask, rng) -> np.ndarray:
    """일부 샘플의 해당 모달리티 전체를 가린다(관측된 셀만)."""
    n_samples = shape[0]
    n_block = max(1, int(n_samples * frac_samples))
    rows = rng.choice(n_samples, size=n_block, replace=False)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, :] = True
    mask &= observed_mask  # 원래 NaN은 평가 제외
    return mask


# ----------------------------------------------------------------------
# 베이스라인 보정 (관측치로만 통계 추정 → 가린 위치 채움)
# ----------------------------------------------------------------------
def impute_mean(X_obs: np.ndarray) -> np.ndarray:
    col_mean = np.nanmean(X_obs, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    out = np.where(np.isnan(X_obs), col_mean, X_obs)
    return out


def impute_median(X_obs: np.ndarray) -> np.ndarray:
    col_med = np.nanmedian(X_obs, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    out = np.where(np.isnan(X_obs), col_med, X_obs)
    return out


def impute_knn(X_obs: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    from sklearn.impute import KNNImputer
    imp = KNNImputer(n_neighbors=min(n_neighbors, max(2, X_obs.shape[0] - 1)))
    return imp.fit_transform(X_obs)


# ----------------------------------------------------------------------
# 지표 계산 (가린 위치에서만)
# ----------------------------------------------------------------------
def feature_std(Y_true: np.ndarray) -> np.ndarray:
    s = np.nanstd(Y_true, axis=0)
    s = np.where((s == 0) | np.isnan(s), 1.0, s)
    return s


def compute_metrics(Y_true, Y_pred, eval_mask) -> Dict[str, float]:
    """Y_*: [samples x features], eval_mask: bool 같은 모양."""
    fstd = feature_std(Y_true)  # NRMSE 정규화용(feature 표준편차)

    yt = Y_true[eval_mask]
    yp = Y_pred[eval_mask]
    if yt.size == 0:
        return {k: np.nan for k in ["rmse", "mae", "nrmse", "feat_pearson",
                                    "feat_spearman", "samp_pearson", "n_cells"]}

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    # NRMSE: feature 표준편차로 정규화한 잔차의 RMSE
    res_norm = (Y_true - Y_pred) / fstd[None, :]
    nrmse = float(np.sqrt(np.mean(res_norm[eval_mask] ** 2)))

    # feature-wise: 각 feature(열)에서 가려진 샘플들에 대해 상관
    feat_p, feat_s = [], []
    for j in range(Y_true.shape[1]):
        rows = np.flatnonzero(eval_mask[:, j])
        if rows.size >= 3:
            a, b = Y_true[rows, j], Y_pred[rows, j]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                feat_p.append(pearsonr(a, b)[0])
                feat_s.append(spearmanr(a, b)[0])
    # sample-wise: 각 샘플(행)에서 가려진 feature들에 대해 상관
    samp_p = []
    for i in range(Y_true.shape[0]):
        cols = np.flatnonzero(eval_mask[i, :])
        if cols.size >= 3:
            a, b = Y_true[i, cols], Y_pred[i, cols]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                samp_p.append(pearsonr(a, b)[0])

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "feat_pearson": float(np.nanmean(feat_p)) if feat_p else np.nan,
        "feat_spearman": float(np.nanmean(feat_s)) if feat_s else np.nan,
        "samp_pearson": float(np.nanmean(samp_p)) if samp_p else np.nan,
        "n_cells": int(eval_mask.sum()),
    }


# ----------------------------------------------------------------------
# MOCHI 보정 (선택)
# ----------------------------------------------------------------------
def mochi_impute(tabs_filled: Dict[str, np.ndarray], ckpt_path: str, device="cpu") -> Dict[str, np.ndarray]:
    """tabs_filled: 가린 위치를 0으로 채운 [samples x features] 배열."""
    import torch
    from models import Generator

    ck = torch.load(ckpt_path, map_location=device)
    dr = tabs_filled["rna"].shape[1]
    dp = tabs_filled["protein"].shape[1]
    dm = tabs_filled["methy"].shape[1]

    Gp = Generator(input_size=dr + dm, output_size=dp, use_attn=True, n_heads=4, d_head=64, target_type="protein", src_size=dr + dm)
    Gr = Generator(input_size=dp + dm, output_size=dr, use_attn=True, n_heads=4, d_head=64, target_type="rna", src_size=dp + dm)
    Gm = Generator(input_size=dr + dp, output_size=dm, use_attn=True, n_heads=4, d_head=64, target_type="methyl", src_size=dr + dp)
    Gp.load_state_dict(ck["Gp"]); Gr.load_state_dict(ck["Gr"]); Gm.load_state_dict(ck["Gm"])
    Gp.eval(); Gr.eval(); Gm.eval()

    rna = torch.tensor(tabs_filled["rna"], dtype=torch.float32)
    prot = torch.tensor(tabs_filled["protein"], dtype=torch.float32)
    methy = torch.tensor(tabs_filled["methy"], dtype=torch.float32)
    with torch.no_grad():
        p = Gp(torch.cat([rna, methy], dim=1), src=None).numpy()
        r = Gr(torch.cat([prot, methy], dim=1), src=None).numpy()
        m = Gm(torch.cat([rna, prot], dim=1), src=None).numpy()
    return {"rna": r, "protein": p, "methy": m}


# ----------------------------------------------------------------------
# 메인 평가 루프
# ----------------------------------------------------------------------
def run(args):
    tabs = load_omics(args.data_dir, args.prefix)
    samples = list(tabs["rna"].index)
    n = len(samples)

    # 샘플 단위 test split (권장). 기본은 전체 평가(빠른 진단) + 경고.
    rng0 = np.random.default_rng(args.split_seed)
    if args.test_frac > 0:
        idx = rng0.permutation(n)
        n_test = max(1, int(n * args.test_frac))
        test_rows = np.sort(idx[:n_test])
        print(f"[split] test 샘플 {len(test_rows)}/{n}개에서만 평가 (권장)")
    else:
        test_rows = np.arange(n)
        print(f"[split] ⚠️ 전체 {n}개 샘플에서 평가 (in-sample, 누수 가능 — 빠른 진단용)")

    arr = {m: tabs[m].values for m in MODALITIES}
    obs = {m: ~np.isnan(arr[m]) for m in MODALITIES}  # 관측 위치

    # 평가 시나리오 구성
    scenarios = []
    if args.scenario in ("mcar", "both"):
        for rate in [float(x) for x in args.rates.split(",")]:
            scenarios.append(("mcar", rate))
    if args.scenario in ("block", "both"):
        scenarios.append(("block", args.block_frac))

    # 베이스라인 메서드
    methods = {"mean": impute_mean, "median": impute_median}
    if args.knn:
        methods["knn"] = lambda X: impute_knn(X, args.knn_neighbors)

    rows = []
    for (kind, param) in scenarios:
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            # 각 모달리티별 마스크 생성 (test 샘플 행에 한정)
            eval_masks = {}
            X_masked = {}
            for m in MODALITIES:
                full_mask = np.zeros(arr[m].shape, dtype=bool)
                # test 행만 가림 후보로
                sub_obs = np.zeros(arr[m].shape, dtype=bool)
                sub_obs[test_rows, :] = obs[m][test_rows, :]
                if kind == "mcar":
                    mk = make_mcar_mask(arr[m].shape, param, sub_obs, rng)
                else:
                    mk = make_block_mask(arr[m].shape, param, sub_obs, rng)
                eval_masks[m] = mk
                Xm = arr[m].copy()
                Xm[mk] = np.nan  # 가린 위치 = NaN
                X_masked[m] = Xm

            # --- 베이스라인 ---
            for meth_name, fn in methods.items():
                for m in MODALITIES:
                    Y_pred = fn(X_masked[m])
                    met = compute_metrics(arr[m], Y_pred, eval_masks[m])
                    met.update({"method": meth_name, "scenario": f"{kind}", "param": param,
                                "seed": seed, "modality": m})
                    rows.append(met)

            # --- MOCHI ---
            if args.use_mochi and args.ckpt:
                filled = {m: np.where(np.isnan(X_masked[m]), 0.0, X_masked[m]) for m in MODALITIES}
                preds = mochi_impute(filled, args.ckpt, device=args.device)
                for m in MODALITIES:
                    met = compute_metrics(arr[m], preds[m], eval_masks[m])
                    met.update({"method": "MOCHI", "scenario": f"{kind}", "param": param,
                                "seed": seed, "modality": m})
                    rows.append(met)

            print(f"  [{kind}={param}] seed {seed} 완료")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "eval_raw.tsv"
    df.to_csv(raw_path, sep="\t", index=False)

    # 집계: 시드 평균 ± 표준편차
    agg = (df.groupby(["scenario", "param", "modality", "method"])
             .agg(rmse_m=("rmse", "mean"), rmse_s=("rmse", "std"),
                  nrmse_m=("nrmse", "mean"),
                  feat_p_m=("feat_pearson", "mean"), feat_p_s=("feat_pearson", "std"),
                  samp_p_m=("samp_pearson", "mean"),
                  n=("n_cells", "mean"))
             .reset_index())
    agg_path = out / "eval_summary_clean.tsv"
    agg.to_csv(agg_path, sep="\t", index=False)

    print("\n===== 요약 (시드 평균) =====")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(agg.to_string(index=False))
    print(f"\n저장: {raw_path}\n저장: {agg_path}")
    return agg


def main():
    ap = argparse.ArgumentParser(description="멀티오믹스 결측 보정 엄밀 평가")
    ap.add_argument("--data_dir", default="/home/dyan/nmf/analysis/processed_data/original_only")
    ap.add_argument("--prefix", default="BRCA_PAM50")
    ap.add_argument("--out", default="/home/dyan/nmf/mochi_code/results/eval_clean")
    ap.add_argument("--scenario", choices=["mcar", "block", "both"], default="both")
    ap.add_argument("--rates", default="0.1,0.3,0.5", help="MCAR 결측률(콤마구분)")
    ap.add_argument("--block_frac", type=float, default=0.3, help="block-missing 대상 샘플 비율")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.0,
                    help="0이면 전체평가(빠른진단). >0이면 해당 비율을 test로 분리해 평가")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--knn", action="store_true", help="KNN 베이스라인 포함")
    ap.add_argument("--knn_neighbors", type=int, default=5)
    ap.add_argument("--use_mochi", action="store_true", help="MOCHI 체크포인트로 평가 포함")
    ap.add_argument("--ckpt", default="/home/dyan/nmf/mochi_code/results/tri_joint_v2/tri_best.ckpt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
