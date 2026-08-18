#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하위 과제로 본 대치 품질.

분류기는 train 분할의 실제 오믹스에만 맞춘다. 평가는 val+test에서
타깃 오믹스를 통째로 지우고 각 방법으로 채운 뒤 AUROC를 잰다.
실제 값(oracle)이 상한, 평균 대치가 하한이다.

레이블은 Xena TCGA BRCA clinicalMatrix에서 가져온다.
  ER    : ER_Status_nature2012 (면역조직화학, 오믹스와 독립)
  HER2  : lab_proc_her2_neu_immunohistochemistry_receptor_status (Equivocal 제외)
  Basal : PAM50Call_RNAseq == Basal (RNA 유래, 쉬운 대조군)
  LumAB : PAM50 LumA 대 LumB (같은 계열 안의 미세 구분, 가장 어렵다)
  Stage : 병기 III+ 대 I/II
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from compare_gate import RIDGE_ALPHAS
from eval_mcar_mnar import MODS, apply_zero
from eval_v5_noise import _get, _obs
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset

N_PC = 30
N_BOOT = 2000
TASKS = ("ER", "HER2", "Basal", "LumAB", "Stage")


def load_labels(path, samples):
    d = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    d.index = [str(i)[:15] for i in d.index]
    d = d[~d.index.duplicated()]
    m = d.reindex([str(s)[:15] for s in samples])

    pam = m["PAM50Call_RNAseq"]
    basal = pd.Series(np.nan, index=m.index)
    basal[pam.notna()] = (pam[pam.notna()] == "Basal").astype(float)
    lumab = pd.Series(np.nan, index=m.index)
    lumab[pam == "LumB"] = 1.0
    lumab[pam == "LumA"] = 0.0

    stage = m["pathologic_stage"].astype(str)
    late = stage.str.startswith(("Stage III", "Stage IV"))
    known = stage.str.startswith("Stage ")
    st = pd.Series(np.nan, index=m.index)
    st[known] = 0.0
    st[late] = 1.0

    return {
        "ER": m["ER_Status_nature2012"].map({"Positive": 1.0, "Negative": 0.0}).to_numpy(float),
        "HER2": m["lab_proc_her2_neu_immunohistochemistry_receptor_status"]
                 .map({"Positive": 1.0, "Negative": 0.0}).to_numpy(float),
        "Basal": basal.to_numpy(float),
        "LumAB": lumab.to_numpy(float),
        "Stage": st.to_numpy(float),
    }


def fit_clf(X, y, seed=42):
    n_pc = int(min(N_PC, X.shape[0] - 1, X.shape[1]))
    pipe = Pipeline([
        ("pca", PCA(n_components=n_pc, random_state=seed)),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    gs = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc", cv=3)
    gs.fit(X, y)
    return gs.best_estimator_


def boot_ci(y, s, seed=0, n=N_BOOT):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--clinical",
                    default="/home/dyan/nmf/mochi_code/processed_data/clinical/BRCA_clinicalMatrix")
    ap.add_argument("--mimir_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_downstream")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--runs", nargs="+", default=[
        "MOCHI=/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt",
        "MOCHI-nogan=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nogan/nmf_tf_best.ckpt",
        "MOCHI-notf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_mean/nmf_tf_best.ckpt",
        "MOCHI-nonmf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nonmf/nmf_tf_best.ckpt",
    ])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    val = TripleSplitDataset(args.data_dir, "val", stats=train.stats)
    test = TripleSplitDataset(args.data_dir, "test", stats=train.stats)

    split = pd.read_csv(Path(args.data_dir) / "sample_split.tsv", sep="\t")
    ids = {s: split.loc[split["split"] == s, "sample"].tolist() for s in ("train", "val", "test")}
    y_tr = load_labels(args.clinical, ids["train"])
    y_va = load_labels(args.clinical, ids["val"])
    y_te = load_labels(args.clinical, ids["test"])
    y_ev = {t: np.concatenate([y_va[t], y_te[t]]) for t in TASKS}

    tr_tabs = {m: _get(train, m) for m in MODS}
    ev_tabs = {m: np.concatenate([_get(val, m), _get(test, m)], 0) for m in MODS}
    ev_obs = {m: np.concatenate([_obs(val, m), _obs(test, m)], 0) for m in MODS}
    print(f"device={device} n_train={len(train)} n_eval={ev_tabs['rna'].shape[0]}")
    for t in TASKS:
        print(f"{t}: train n={int(np.isfinite(y_tr[t]).sum())} pos={int(np.nansum(y_tr[t]))} | "
              f"eval n={int(np.isfinite(y_ev[t]).sum())} pos={int(np.nansum(y_ev[t]))}")

    from sklearn.linear_model import RidgeCV
    print("fitting Ridge...")
    ridge2 = {}
    for tgt in MODS:
        srcs = [s for s in MODS if s != tgt]
        X = np.concatenate([tr_tabs[s] for s in srcs], 1)
        ridge2[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, tr_tabs[tgt])

    models = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            models[name] = load_nmf_tf(path, device)
        else:
            print(f"skip {name}: {path} 없음")

    mimir = mimir_mv = None
    if Path(args.mimir_dir).exists():
        from mimir_wrap import frames_from_dir, load_mimir, predict_block
        mimir, mimir_mv = load_mimir(args.mimir_dir, device)
        tr_frames, stats_df, _ = frames_from_dir(args.data_dir, "train")
        columns = {m: tr_frames[m].columns for m in MODS}
        ev_index = pd.Index(ids["val"] + ids["test"])

    rows = []
    for tgt in MODS:
        masks = {m: (block_mask(ev_obs[m]) if m == tgt else np.zeros_like(ev_obs[m], dtype=bool))
                 for m in MODS}
        filled = apply_zero(ev_tabs, masks)

        cand = {"oracle": ev_tabs[tgt], "mean": np.zeros_like(ev_tabs[tgt])}
        cand["Ridge-2to1"] = ridge2[tgt].predict(
            np.concatenate([filled[s] for s in MODS if s != tgt], 1))
        for name, mdl in models.items():
            cand[name] = predict_nmf_tf(mdl, filled, device, missing=tgt)[tgt]
        if mimir is not None:
            nan_tabs = {}
            for m in MODS:
                x = ev_tabs[m].copy()
                x[~ev_obs[m]] = np.nan
                x[masks[m]] = np.nan
                nan_tabs[m] = x
            present = {m: pd.DataFrame(nan_tabs[m], index=ev_index, columns=columns[m])
                       for m in MODS if m != tgt}
            cand["MIMIR"] = predict_block(mimir, mimir_mv, present, tgt,
                                          columns[tgt], ev_index, device)

        for task in TASKS:
            ok_tr = np.isfinite(y_tr[task])
            ok_ev = np.isfinite(y_ev[task])
            if len(np.unique(y_ev[task][ok_ev])) < 2 or ok_tr.sum() < 30:
                print(f"skip {task}: 레이블 부족")
                continue
            clf = fit_clf(tr_tabs[tgt][ok_tr], y_tr[task][ok_tr].astype(int))
            for name, Xhat in cand.items():
                s = clf.predict_proba(np.asarray(Xhat)[ok_ev])[:, 1]
                yy = y_ev[task][ok_ev].astype(int)
                auc = float(roc_auc_score(yy, s))
                lo, hi = boot_ci(yy, s)
                rows.append({"task": task, "imputed": tgt, "method": name,
                             "auc": auc, "ci_lo": lo, "ci_hi": hi,
                             "n_eval": int(ok_ev.sum()), "n_pos": int(yy.sum())})
            print(f"[{task}] imputed={tgt} done")

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "downstream_auc.tsv", sep="\t", index=False, float_format="%.4f")
    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        print(f"\n=== {task} AUROC (n={sub['n_eval'].iloc[0]}, pos={sub['n_pos'].iloc[0]}) ===")
        print(sub.pivot_table(index="method", columns="imputed", values="auc")
              .round(4).to_string())
    print(f"saved {out / 'downstream_auc.tsv'}")


if __name__ == "__main__":
    main()
