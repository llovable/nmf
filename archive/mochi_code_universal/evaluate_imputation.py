#!/usr/bin/env python3
"""
결측치 보간 품질 평가용 비교 도구.
- 기본 베이스라인(mean imputation)
- Tri-joint 모델 (옵션: checkpoint 제공 시)
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dataloader_universal import load_and_align_omics
from impute_universal import TriJointImputerUniversal


def _mask_random(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    관측값 중 일부를 무작위로 NaN 처리.
    반환: (masked_df, mask_bool) where mask_bool True=가림
    """
    arr = df.values.copy()
    obs_mask = ~np.isnan(arr)
    obs_idx = np.argwhere(obs_mask)
    n_mask = int(len(obs_idx) * rate)
    if n_mask <= 0:
        return df.copy(), np.zeros_like(arr, dtype=bool)
    choose = rng.choice(len(obs_idx), size=n_mask, replace=False)
    mask_idx = obs_idx[choose]
    masked = arr.copy()
    masked[mask_idx[:, 0], mask_idx[:, 1]] = np.nan
    mask_bool = np.zeros_like(arr, dtype=bool)
    mask_bool[mask_idx[:, 0], mask_idx[:, 1]] = True
    return pd.DataFrame(masked, index=df.index, columns=df.columns), mask_bool


def _mean_impute(df: pd.DataFrame) -> pd.DataFrame:
    # 특징별 평균으로 채움
    return df.apply(lambda col: col.fillna(col.mean()), axis=0)


def _flatten_masked(true_df: pd.DataFrame, pred_df: pd.DataFrame, mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = true_df.values[mask_bool]
    y_pred = pred_df.values[mask_bool]
    return y_true, y_pred


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "pearson": float("nan"), "n": 0}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"rmse": rmse, "mae": mae, "pearson": pearson, "n": int(y_true.size)}


def evaluate(
    rna: pd.DataFrame,
    methy: pd.DataFrame,
    protein: pd.DataFrame,
    missing_rate: float,
    seed: int,
    checkpoint: str | None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    rng = np.random.default_rng(seed)

    # pseudo-missing 생성
    rna_masked, rna_mask = _mask_random(rna, missing_rate, rng)
    methy_masked, methy_mask = _mask_random(methy, missing_rate, rng)
    protein_masked, protein_mask = _mask_random(protein, missing_rate, rng)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # baseline: mean
    rna_mean = _mean_impute(rna_masked)
    methy_mean = _mean_impute(methy_masked)
    protein_mean = _mean_impute(protein_masked)
    results["mean"] = {
        "rna": _metrics(*_flatten_masked(rna, rna_mean, rna_mask)),
        "methy": _metrics(*_flatten_masked(methy, methy_mean, methy_mask)),
        "protein": _metrics(*_flatten_masked(protein, protein_mean, protein_mask)),
    }

    # tri-joint model (optional)
    if checkpoint:
        imputer = TriJointImputerUniversal(checkpoint_path=checkpoint, device="auto", batch_size=64)
        imputed = imputer.impute(rna_masked, methy_masked, protein_masked)
        results["tri_joint"] = {
            "rna": _metrics(*_flatten_masked(rna, imputed["rna"], rna_mask)),
            "methy": _metrics(*_flatten_masked(methy, imputed["methy"], methy_mask)),
            "protein": _metrics(*_flatten_masked(protein, imputed["protein"], protein_mask)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="보간 품질 비교 도구")
    parser.add_argument("--rna", required=True, help="RNA TSV (features x samples)")
    parser.add_argument("--methy", required=True, help="Methylation TSV (features x samples)")
    parser.add_argument("--protein", required=True, help="Protein TSV (features x samples)")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="가릴 비율")
    parser.add_argument("--seed", type=int, default=123, help="랜덤 시드")
    parser.add_argument("--checkpoint", default=None, help="tri_best.ckpt 경로 (선택)")
    parser.add_argument("--sample_id_mode", type=str, default="none", choices=["none", "tcga12", "tcga15"])
    parser.add_argument("--output", default=None, help="결과 저장 경로(JSON)")

    args = parser.parse_args()

    tables = load_and_align_omics(
        rna_path=args.rna,
        methy_path=args.methy,
        protein_path=args.protein,
        sample_id_mode=args.sample_id_mode,
        sample_join="intersection",
    )

    results = evaluate(
        rna=tables.rna,
        methy=tables.methy,
        protein=tables.protein,
        missing_rate=args.missing_rate,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )

    # 출력
    print("\n=== Imputation Benchmark ===")
    for model_name, by_modality in results.items():
        print(f"\n[{model_name}]")
        for mod, m in by_modality.items():
            print(f"  {mod}: rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  pearson={m['pearson']:.4f}  n={m['n']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            import json
            json.dump(results, f, indent=2)
        print(f"\n✅ 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
