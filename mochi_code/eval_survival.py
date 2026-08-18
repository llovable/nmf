#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRCA 631 스플릿에서 생존 평가.

Cox는 train에만 맞추고, C-index/KM은 test(및 val+test)에서만 본다.
블록 결측: 타깃을 지운 뒤 mean / TOBMI / Ridge / MOCHI-v5 / 공식 OmicsNMF·OmiTrans로 채운다.
특징은 train PCA(15)로 줄인다. 예전처럼 같은 샘플에 Cox를 맞추고 채점하지 않는다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors

from compare_gate import PAIRS_1TO1, PAIRS_2TO1, RIDGE_ALPHAS, load_v5
from eval_v5_noise import load_1to1, predict_2to1
from official_wrap import load_omicsnmf_ckpt, load_omitrans_ckpt
from train_gate import TripleSplitDataset

SRC_1TO1 = dict(PAIRS_1TO1)
SRC_2TO1 = dict(PAIRS_2TO1)
OFFICIAL_DIR = Path("/home/dyan/nmf/mochi_code/results/current/official")

MODS = ("protein", "rna", "methyl")
N_PC = 15


def n15(s):
    return str(s).split(".")[0][:15]


def load_survival(path, samples):
    df = pd.read_csv(path, sep="\t", compression="gzip")
    df["s15"] = df["sample"].map(n15)
    # 종양 01 우선
    df["_tumor"] = df["sample"].astype(str).str[13:15].eq("01")
    df = df.sort_values(["_tumor", "sample"], ascending=[False, True]).drop_duplicates("s15")
    want = pd.Index([n15(s) for s in samples])
    out = df.set_index("s15").reindex(want)
    out["OS.time"] = pd.to_numeric(out["OS.time"], errors="coerce")
    out["OS"] = pd.to_numeric(out["OS"], errors="coerce")
    return out


def _get(ds, name):
    return {"rna": ds.rna_f, "protein": ds.prot_f, "methyl": ds.methy_f}[name]


def concat_omics(rna, prot, methy):
    return np.concatenate([rna, prot, methy], axis=1).astype(np.float32)


def fit_ridges(train):
    r1, r2 = {}, {}
    for tgt, src in PAIRS_1TO1:
        r1[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(_get(train, src), _get(train, tgt))
    for tgt, srcs in PAIRS_2TO1:
        X = np.concatenate([_get(train, s) for s in srcs], 1)
        r2[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, _get(train, tgt))
    return r1, r2


def _hstack(tabs, names):
    return np.concatenate([tabs[n] for n in names], 1)


def tobmi_predict(Xtr, Ytr, Xev, k=10):
    k = min(k, len(Xtr))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xtr)
    idx = nn.kneighbors(Xev, return_distance=False)
    return Ytr[idx].mean(axis=1)


def load_official(official_dir, dim_r, dim_p, dim_m, device):
    dim = {"rna": dim_r, "protein": dim_p, "methyl": dim_m}
    onmf, ot = {}, {}
    for tgt, src in PAIRS_1TO1:
        onmf[tgt] = load_omicsnmf_ckpt(
            Path(official_dir) / "omicsnmf" / f"{src}_to_{tgt}.ckpt",
            dim[src], dim[tgt], device)
        ot[tgt] = load_omitrans_ckpt(
            Path(official_dir) / "omitrans" / f"{src}_to_{tgt}.ckpt",
            dim[src], dim[tgt], device)
    return onmf, ot


def impute_target(method, tgt, tabs, ctx):
    """tabs: dict of observed eval arrays. Returns a filled copy."""
    out = {k: v.copy() for k, v in tabs.items()}
    if method == "observed":
        return out
    if method == "mean":
        out[tgt] = np.zeros_like(out[tgt])
        return out
    if method == "TOBMI-1to1":
        src = SRC_1TO1[tgt]
        out[tgt] = tobmi_predict(ctx["train"][src], ctx["train"][tgt], tabs[src])
        return out
    if method == "TOBMI-2to1":
        srcs = SRC_2TO1[tgt]
        out[tgt] = tobmi_predict(
            _hstack(ctx["train"], srcs), ctx["train"][tgt], _hstack(tabs, srcs))
        return out
    if method == "Ridge-1to1":
        src = SRC_1TO1[tgt]
        out[tgt] = ctx["ridge1"][tgt].predict(tabs[src])
        return out
    if method == "Ridge-2to1":
        srcs = SRC_2TO1[tgt]
        out[tgt] = ctx["ridge2"][tgt].predict(_hstack(tabs, srcs))
        return out
    if method == "MOCHI-v5-1to1":
        pred = predict_2to1(ctx["mochi1"], tabs["rna"], tabs["protein"], tabs["methyl"], ctx["device"])
        out[tgt] = pred[tgt]
        return out
    if method == "MOCHI-v5-2to1":
        pred = predict_2to1(ctx["mochi2"], tabs["rna"], tabs["protein"], tabs["methyl"], ctx["device"])
        out[tgt] = pred[tgt]
        return out
    if method == "OmicsNMF-official":
        src = SRC_1TO1[tgt]
        out[tgt] = ctx["onmf"][tgt].predict_z(tabs[src])
        return out
    if method == "OmiTrans-official":
        src = SRC_1TO1[tgt]
        out[tgt] = ctx["ot"][tgt].predict(tabs[src])
        return out
    raise ValueError(method)


def cox_cindex(train_X, train_y, eval_X, eval_y, n_pc=N_PC):
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index

    ok_tr = train_y["OS.time"].notna() & train_y["OS"].notna() & (train_y["OS.time"] > 0)
    ok_ev = eval_y["OS.time"].notna() & eval_y["OS"].notna() & (eval_y["OS.time"] > 0)
    Xtr, ytr = train_X[ok_tr.values], train_y.loc[ok_tr]
    Xev, yev = eval_X[ok_ev.values], eval_y.loc[ok_ev]
    pca = PCA(n_components=min(n_pc, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zev = pca.transform(Xev)
    cols = [f"pc{i}" for i in range(Ztr.shape[1])]
    tr = pd.DataFrame(Ztr, columns=cols, index=ytr.index)
    tr["OS.time"] = ytr["OS.time"].values
    tr["OS"] = ytr["OS"].values
    ev = pd.DataFrame(Zev, columns=cols, index=yev.index)
    ev["OS.time"] = yev["OS.time"].values
    ev["OS"] = yev["OS"].values
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tr, duration_col="OS.time", event_col="OS")
    risk = cph.predict_partial_hazard(ev[cols]).values.ravel()
    cidx = float(concordance_index(ev["OS.time"], -risk, ev["OS"]))
    med = np.median(risk)
    high = risk > med
    lr = logrank_test(
        ev.loc[high, "OS.time"], ev.loc[~high, "OS.time"],
        ev.loc[high, "OS"], ev.loc[~high, "OS"],
    )
    return {
        "c_index": cidx,
        "logrank_p": float(lr.p_value),
        "n": int(ok_ev.sum()),
        "n_events": int(yev["OS"].sum()),
        "n_train": int(ok_tr.sum()),
        "n_train_events": int(ytr["OS"].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--surv", default="/home/dyan/nmf/mochi_code/processed_data/clinical/TCGA-BRCA.survival.tsv.gz")
    ap.add_argument("--v5_ckpt", default="/home/dyan/nmf/mochi_code/results/current/gate_tri_v5/tri_best.ckpt")
    ap.add_argument("--v5_1to1", default="/home/dyan/nmf/mochi_code/results/current/gate_compare/mochi_1to1.ckpt")
    ap.add_argument("--official_dir", default="/home/dyan/nmf/mochi_code/results/current/official")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_v5_survival")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    val = TripleSplitDataset(args.data_dir, "val", stats=train.stats)
    test = TripleSplitDataset(args.data_dir, "test", stats=train.stats)
    dim_r, dim_p, dim_m = train.rna_f.shape[1], train.prot_f.shape[1], train.methy_f.shape[1]
    print(f"device={device} n={len(train)}/{len(val)}/{len(test)}")

    split = pd.read_csv(Path(args.data_dir) / "sample_split.tsv", sep="\t")
    # TripleSplitDataset order follows intersection of tsv columns after T — use same index via csv load
    def ids_for(split_name):
        rna = pd.read_csv(Path(args.data_dir) / f"rna.{split_name}.tsv", sep="\t", index_col=0)
        prot = pd.read_csv(Path(args.data_dir) / f"protein.{split_name}.tsv", sep="\t", index_col=0)
        meth = pd.read_csv(Path(args.data_dir) / f"methy.{split_name}.tsv", sep="\t", index_col=0)
        idx = rna.columns.intersection(prot.columns).intersection(meth.columns)
        return list(idx)

    id_map = {s: ids_for(s) for s in ("train", "val", "test")}
    surv = {
        "train": load_survival(args.surv, id_map["train"]),
        "val": load_survival(args.surv, id_map["val"]),
        "test": load_survival(args.surv, id_map["test"]),
    }
    for k, v in surv.items():
        print(f"survival {k}: n={v['OS.time'].notna().sum()} events={int(v['OS'].fillna(0).sum())}")

    print("fitting Ridge 1→1/2→1 and loading models...")
    ridge1, ridge2 = fit_ridges(train)
    mochi2 = load_v5(args.v5_ckpt, dim_r, dim_p, dim_m, device)
    mochi1 = load_1to1(args.v5_1to1, dim_r, dim_p, dim_m, device)
    onmf, ot = load_official(args.official_dir, dim_r, dim_p, dim_m, device)

    mats = {
        "train": {m: _get(train, m) for m in MODS},
        "val": {m: _get(val, m) for m in MODS},
        "test": {m: _get(test, m) for m in MODS},
    }
    ctx = {
        "train": mats["train"],
        "ridge1": ridge1,
        "ridge2": ridge2,
        "mochi1": mochi1,
        "mochi2": mochi2,
        "onmf": onmf,
        "ot": ot,
        "device": device,
    }
    Xtr_obs = concat_omics(mats["train"]["rna"], mats["train"]["protein"], mats["train"]["methyl"])

    rows = []
    eval_sets = {
        "test": ("test",),
        "heldout": ("val", "test"),
    }
    methods = (
        "mean",
        "TOBMI-1to1", "TOBMI-2to1",
        "Ridge-1to1", "Ridge-2to1",
        "MOCHI-v5-1to1", "MOCHI-v5-2to1",
        "OmicsNMF-official", "OmiTrans-official",
    )
    setting = {
        "observed": "none", "mean": "none",
        "TOBMI-1to1": "1to1", "TOBMI-2to1": "2to1",
        "Ridge-1to1": "1to1", "Ridge-2to1": "2to1",
        "MOCHI-v5-1to1": "1to1", "MOCHI-v5-2to1": "2to1",
        "OmicsNMF-official": "1to1", "OmiTrans-official": "1to1",
    }
    targets = ("none",) + MODS

    for eval_name, parts in eval_sets.items():
        y_ev = pd.concat([surv[p] for p in parts], axis=0)
        tabs = {m: np.concatenate([mats[p][m] for p in parts], 0) for m in MODS}
        for tgt in targets:
            miss = "complete" if tgt == "none" else f"missing-{tgt}"
            use_methods = ("observed",) if tgt == "none" else methods
            for method in use_methods:
                filled = impute_target(method, tgt if tgt != "none" else "rna", tabs, ctx)
                Xev = concat_omics(filled["rna"], filled["protein"], filled["methyl"])
                met = cox_cindex(Xtr_obs, surv["train"], Xev, y_ev)
                rows.append({
                    "eval_set": eval_name, "missing": miss, "method": method,
                    "setting": setting[method], **met,
                })
                print(f"{eval_name:8s} {miss:18s} {method:20s} "
                      f"C={met['c_index']:.4f} p={met['logrank_p']:.3g} "
                      f"n={met['n']} ev={met['n_events']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "survival.tsv", sep="\t", index=False, float_format="%.6f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {out / 'survival.tsv'}")


if __name__ == "__main__":
    main()
