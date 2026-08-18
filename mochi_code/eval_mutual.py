#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
상호 보간의 이득을 경로 수준에서 확인한다.

같은 MOCHI 한 모형으로 타깃을 채우되 소스를 바꾼다.
  두 소스: 나머지 오믹스 둘 다 관측
  한 소스: 둘 중 하나만 관측 (나머지도 블록 결측 처리)
경로 활성 상관과 경로 분산 유지가 소스 수에 따라 어떻게 변하는지 본다.
Ridge는 같은 조건에서 1→1과 2→1로 맞춘 별도 모형을 쓴다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

from compare_gate import RIDGE_ALPHAS
from eval_biology import col_corr, load_gmt, pathway_scores
from eval_mcar_mnar import MODS, apply_zero
from eval_stress import predict_blocks
from eval_v5_noise import _get, _obs
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    clin = "/home/dyan/nmf/mochi_code/processed_data/clinical"
    ap.add_argument("--probemap", default=f"{clin}/gencode.probeMap")
    ap.add_argument("--gmt", default=f"{clin}/hallmark.gmt")
    ap.add_argument("--ckpt",
                    default="/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_mutual")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    test = TripleSplitDataset(args.data_dir, "test", stats=train.stats)
    d = Path(args.data_dir)
    rna_ids = pd.read_csv(d / "rna.train.tsv", sep="\t", index_col=0, usecols=[0]).index

    pm = pd.read_csv(args.probemap, sep="\t")
    pm["b"] = pm["id"].str.split(".").str[0]
    ens2sym = pm.drop_duplicates("b").set_index("b")["gene"]
    rna_sym = [ens2sym.get(i.split(".")[0], None) for i in rna_ids]
    gene_index = {g: i for i, g in enumerate(rna_sym) if g is not None}
    sets = load_gmt(args.gmt, set(gene_index))
    print(f"device={device} 경로={len(sets)}")

    ytrue = {m: _get(test, m) for m in MODS}
    obs = {m: _obs(test, m) for m in MODS}
    model = load_nmf_tf(args.ckpt, device)

    print("fitting Ridge (1→1, 2→1)...")
    ridge = {}
    for tgt in MODS:
        srcs = [s for s in MODS if s != tgt]
        X = np.concatenate([_get(train, s) for s in srcs], 1)
        ridge[(tgt, "both")] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, _get(train, tgt))
        for s in srcs:
            ridge[(tgt, s)] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(_get(train, s), _get(train, tgt))

    ps_true = pathway_scores(ytrue["rna"], gene_index, sets)
    rows = []
    tgt = "rna"
    srcs = [s for s in MODS if s != tgt]

    def score(name, source, Xhat):
        Xhat = np.asarray(Xhat, dtype=np.float64)
        ps = pathway_scores(Xhat, gene_index, sets)
        r = col_corr(ps_true, ps)
        sd = ps.std(0) / np.maximum(1e-8, ps_true.std(0))
        z = float(np.sqrt(np.mean((ytrue[tgt] - Xhat) ** 2)))
        rows.append({"method": name, "source": source, "z_rmse": z,
                     "pathway_r": float(np.mean(r)),
                     "pathway_sd_ratio": float(np.mean(sd))})

    masks_both = {m: (block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool))
                  for m in MODS}
    filled = apply_zero(ytrue, masks_both)
    score("MOCHI", "두 소스", predict_nmf_tf(model, filled, device, missing=tgt)[tgt])
    score("Ridge", "두 소스", ridge[(tgt, "both")].predict(
        np.concatenate([filled[s] for s in srcs], 1)))

    for keep in srcs:
        drop = [m for m in MODS if m != keep]
        masks = {m: (block_mask(obs[m]) if m in drop else np.zeros_like(obs[m], dtype=bool))
                 for m in MODS}
        f = apply_zero(ytrue, masks)
        score("MOCHI", f"{keep}만", predict_blocks(model, f, device, tuple(drop))[tgt])
        score("Ridge", f"{keep}만", ridge[(tgt, keep)].predict(f[keep]))

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "mutual.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print("\n=== RNA 블록 결측: 소스 수에 따른 경로 보존 ===")
    print(df.pivot_table(index="source", columns="method",
                         values=["z_rmse", "pathway_r", "pathway_sd_ratio"])
          .round(4).to_string())
    print(f"\nsaved {out / 'mutual.tsv'}")


if __name__ == "__main__":
    main()
