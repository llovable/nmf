#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""같은 BRCA split에서 공식 OmicsNMF · OmiTrans 1→1을 학습한다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from official_wrap import (
    PAIRS_1TO1, _arrays, train_omicsnmf_pair, train_omitrans_pair,
)
from train_gate import TripleSplitDataset


def _save_onmf(path, mdl, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "G": mdl.G.state_dict(),
        "shift": mdl.shift.shift,
        "meta": meta,
    }, path)


def _save_ot(path, mdl, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"G": mdl.G.state_dict(), "meta": meta}, path)


def eval_pairs(models, getter, ev):
    out = {}
    for tgt, src in PAIRS_1TO1:
        Xs, Yt = _arrays(ev, src, tgt)
        out[tgt] = getter(models[(tgt, src)], Xs, Yt)
    out["avg"] = float(np.mean([out["protein"], out["rna"], out["methyl"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--save_dir", default="/home/dyan/nmf/mochi_code/results/current/official")
    ap.add_argument("--which", default="both", choices=["omicsnmf", "omitrans", "both"])
    ap.add_argument("--epochs_onmf", type=int, default=80)
    ap.add_argument("--epochs_ot", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TripleSplitDataset(args.data_dir, "train")
    val_ds = TripleSplitDataset(args.data_dir, "val", stats=train_ds.stats)
    test_ds = TripleSplitDataset(args.data_dir, "test", stats=train_ds.stats)
    print(f"n train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    rows = []

    if args.which in ("omicsnmf", "both"):
        print("=== official OmicsNMF 1→1 (no test leak, our split) ===")
        onmf = {}
        for tgt, src in PAIRS_1TO1:
            print(f"OmicsNMF {src} → {tgt}")
            Xtr, Ytr = _arrays(train_ds, src, tgt)
            Xva, Yva = _arrays(val_ds, src, tgt)
            mdl, best = train_omicsnmf_pair(
                Xtr, Ytr, Xva, Yva, device,
                epochs=args.epochs_onmf, patience=args.patience)
            ckpt = save_dir / "omicsnmf" / f"{src}_to_{tgt}.ckpt"
            _save_onmf(ckpt, mdl, {"src": src, "tgt": tgt, "val": best})
            onmf[(tgt, src)] = mdl
            print(f"  saved {ckpt} val={best:.4f}")
        for split, ds in (("val", val_ds), ("test", test_ds)):
            met = eval_pairs(onmf, lambda m, x, y: m.rmse_z(x, y), ds)
            rows.append({"method": "OmicsNMF-official", "setting": "1to1", "split": split, **met})
            print(f"OmicsNMF-official {split}: avg={met['avg']:.4f} "
                  f"P={met['protein']:.4f} R={met['rna']:.4f} M={met['methyl']:.4f}")

    if args.which in ("omitrans", "both"):
        print("=== official OmiTrans 1→1 (FCG+FCD, λ_dist=100) ===")
        ot = {}
        for tgt, src in PAIRS_1TO1:
            print(f"OmiTrans {src} → {tgt}")
            Xtr, Ytr = _arrays(train_ds, src, tgt)
            Xva, Yva = _arrays(val_ds, src, tgt)
            mdl, best = train_omitrans_pair(
                Xtr, Ytr, Xva, Yva, device,
                epochs=args.epochs_ot, patience=args.patience)
            ckpt = save_dir / "omitrans" / f"{src}_to_{tgt}.ckpt"
            _save_ot(ckpt, mdl, {"src": src, "tgt": tgt, "val": best})
            ot[(tgt, src)] = mdl
            print(f"  saved {ckpt} val={best:.4f}")
        for split, ds in (("val", val_ds), ("test", test_ds)):
            met = eval_pairs(ot, lambda m, x, y: m.rmse(x, y), ds)
            rows.append({"method": "OmiTrans-official", "setting": "1to1", "split": split, **met})
            print(f"OmiTrans-official {split}: avg={met['avg']:.4f} "
                  f"P={met['protein']:.4f} R={met['rna']:.4f} M={met['methyl']:.4f}")

    df = pd.DataFrame(rows)
    out = save_dir / "metrics.tsv"
    df.to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
