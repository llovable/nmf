#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""λ_W 스윕용: 예측 계수 Ŵ 가 참 NMF 계수 W_ref 에 얼마나 가까운지.

재구성 z-RMSE와 별개다. 논문이 묻는 것은 머리를 세게 당기면
과희소화·과분산이 W_ref 쪽으로 줄어드는가다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_nmf_tf import MODS, load_nmf_tf
from train_gate import TripleSplitDataset


@torch.no_grad()
def collect(model, ds, device, batch_size=64):
    model.eval()
    acc = {m: {"hat": [], "ref": []} for m in MODS}
    n = len(ds)
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        xs, present = {}, {}
        for m, key in (("rna", "rna_f"), ("protein", "prot_f"), ("methyl", "methy_f")):
            t = torch.from_numpy(np.asarray(getattr(ds, key)[sl], dtype=np.float32)).to(device)
            xs[m] = t
            present[m] = torch.ones(t.size(0), dtype=torch.bool, device=device)
        for tgt in MODS:
            _, W_hat = model.loo_parts(xs, present, tgt)
            W_ref = model.tokenizers[tgt].encode(xs[tgt])
            acc[tgt]["hat"].append(W_hat.cpu().numpy())
            acc[tgt]["ref"].append(W_ref.cpu().numpy())
    return {m: {k: np.concatenate(v, 0) for k, v in d.items()} for m, d in acc.items()}


def summarize(hat, ref, near_zero=1e-3):
    hat = hat.reshape(-1)
    ref = ref.reshape(-1)
    return {
        "zero_hat": float((hat < near_zero).mean()),
        "zero_ref": float((ref < near_zero).mean()),
        "mean_hat": float(hat.mean()),
        "mean_ref": float(ref.mean()),
        "sd_hat": float(hat.std()),
        "sd_ref": float(ref.std()),
        "sd_ratio": float(hat.std() / max(ref.std(), 1e-8)),
        "mse": float(np.mean((hat - ref) ** 2)),
        "r": float(np.corrcoef(hat, ref)[0, 1]) if hat.std() > 0 and ref.std() > 0 else 0.0,
        "min_hat": float(hat.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    ds = TripleSplitDataset(args.data_dir, args.split, stats=train.stats)
    model = load_nmf_tf(args.ckpt, device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"device={device} n={len(ds)}  w_head={ck.get('w_head_act', 'relu')}  "
          f"lambda_w={ck.get('lambda_w', '?')}  gan={ck.get('gan_to_mse', '?')}")

    bags = collect(model, ds, device)
    rows = []
    for m in MODS:
        row = {"modality": m, "w_head_act": ck.get("w_head_act", "relu"),
               "lambda_w": ck.get("lambda_w", np.nan),
               "gan_to_mse": ck.get("gan_to_mse", np.nan),
               "gamma_nonneg": ck.get("gamma_nonneg", False)}
        row.update(summarize(bags[m]["hat"], bags[m]["ref"]))
        rows.append(row)
        print(f"{m:8s}  zero_hat={row['zero_hat']:.3f} (ref {row['zero_ref']:.3f})  "
              f"sd_ratio={row['sd_ratio']:.3f}  r={row['r']:.3f}  mse={row['mse']:.4f}  "
              f"min_hat={row['min_hat']:.2e}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
