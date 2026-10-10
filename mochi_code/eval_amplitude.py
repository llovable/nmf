#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경로 진폭 수축이 이후 분석을 실제로 바꾸는지 본다.

RNA를 통째로 가린 뒤 채운 값으로 세 가지를 잰다.
  1) Hallmark 경로 점수의 환자 간 표준편차 비 (이미 보고한 진폭)
  2) 군집 안정성 — 참값 경로 점수의 k-means 분할을 대치값이 재현하는 정도(ARI),
     참값 중심점에 다시 붙였을 때의 정확도
  3) 차등 발현 검정력 — 참값에서 유의한 유전자·경로가 대치값에서도 남는지,
     효과크기(|t|, Cohen's d)가 얼마나 줄는지 (임상 라벨이 있을 때만)

진폭 비 자체는 이미 표 10·11에 있다. 이 스크립트는 그 수축이
군집·검정의 숫자로 나타나는지를 보기 위한 후속 실험이다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, silhouette_score

from compare_gate import RIDGE_ALPHAS
from eval_biology import col_corr, load_gmt, pathway_scores, welch_t
from eval_mcar_mnar import MODS, apply_zero
from eval_v5_noise import _get, _obs
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset

KS = (2, 3, 4, 5)
FDR = 0.05
TOP_D = 0.8


def bh_q(p):
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = np.clip(p[order], 0, 1)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return np.divide(a.mean(0) - b.mean(0), sp, out=np.zeros(X.shape[1]), where=sp > 1e-12)


def welch_p(X, y):
    a, b = X[y == 1], X[y == 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        res = ttest_ind(a, b, axis=0, equal_var=False)
    t = np.nan_to_num(np.asarray(res.statistic, dtype=np.float64))
    p = np.nan_to_num(np.asarray(res.pvalue, dtype=np.float64), nan=1.0)
    return t, p


def de_pack(X, y, t_true=None, d_true=None, sig_true=None):
    t, p = welch_p(X, y)
    q = bh_q(p)
    sig = q < FDR
    d = cohens_d(X, y)
    rec = {
        "n_sig": int(sig.sum()),
        "median_abs_t": float(np.median(np.abs(t))),
        "median_abs_d": float(np.median(np.abs(d))),
        "n_large_d": int((np.abs(d) >= TOP_D).sum()),
    }
    if sig_true is not None and sig_true.any():
        rec["recall"] = float(sig[sig_true].mean())
        rec["effect_ratio"] = float(
            np.median(np.abs(t[sig_true]) / np.maximum(1e-8, np.abs(t_true[sig_true])))
        )
        rec["d_ratio"] = float(
            np.median(np.abs(d[sig_true]) / np.maximum(1e-8, np.abs(d_true[sig_true])))
        )
    return rec, t, d, sig


def cluster_pack(true_ps, hat_ps, rng_seed=0):
    rec = {}
    aris = []
    accs = []
    for k in KS:
        if true_ps.shape[0] <= k:
            continue
        kt = KMeans(n_clusters=k, n_init=20, random_state=rng_seed).fit(true_ps)
        kh = KMeans(n_clusters=k, n_init=20, random_state=rng_seed).fit(hat_ps)
        ari = float(adjusted_rand_score(kt.labels_, kh.labels_))
        rec[f"ari_k{k}"] = ari
        aris.append(ari)
        pred = kt.predict(hat_ps)
        rec[f"centroid_acc_k{k}"] = float((pred == kt.labels_).mean())
        accs.append(rec[f"centroid_acc_k{k}"])
    rec["ari_mean"] = float(np.mean(aris)) if aris else np.nan
    rec["centroid_acc_mean"] = float(np.mean(accs)) if accs else np.nan
    return rec


def load_group(clinical, samples, col, mapping):
    if not clinical or not Path(clinical).exists():
        return None
    cm = pd.read_csv(clinical, sep="\t", index_col=0, low_memory=False)
    cm.index = [str(i)[:15] for i in cm.index]
    cm = cm[~cm.index.duplicated()]
    if col not in cm.columns:
        return None
    raw = cm.reindex([s[:15] for s in samples])[col]
    if mapping is None:
        y = raw.astype(object)
        ok = y.notna() & (y.astype(str) != "nan")
        if ok.sum() < 10 or y[ok].nunique() < 2:
            return None
        return y.to_numpy(), ok.to_numpy()
    y = raw.map(mapping).to_numpy(float)
    ok = np.isfinite(y)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    return y, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    clin = "/home/dyan/nmf/mochi_code/processed_data/clinical"
    ap.add_argument("--probemap", default=f"{clin}/gencode.probeMap")
    ap.add_argument("--gmt", default=f"{clin}/hallmark.gmt")
    ap.add_argument("--clinical", default=f"{clin}/BRCA_clinicalMatrix")
    ap.add_argument("--label_col", default="ER_Status_nature2012")
    ap.add_argument("--subtype_col", default="PAM50Call_RNAseq")
    ap.add_argument("--mimir_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/bio_amplitude")
    ap.add_argument("--gpu", type=int, default=-1, help="-1이면 CPU")
    ap.add_argument("--runs", nargs="+", default=[
        "MOCHI=/home/dyan/nmf/mochi_code/results/current/lr_brca_hybrid/nmf_tf_best.ckpt",
        "MOCHI-knockout=/home/dyan/nmf/mochi_code/results/current/lr_brca_hybrid/nmf_tf_best.ckpt=gamma0",
    ])
    args = ap.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    test = TripleSplitDataset(args.data_dir, "test", stats=train.stats)
    d = Path(args.data_dir)
    rna_ids = pd.read_csv(d / "rna.train.tsv", sep="\t", index_col=0, usecols=[0]).index

    pm = pd.read_csv(args.probemap, sep="\t")
    pm["b"] = pm["id"].str.split(".").str[0]
    ens2sym = pm.drop_duplicates("b").set_index("b")["gene"]
    rna_sym = pd.Index([ens2sym.get(i.split(".")[0], None) for i in rna_ids])
    gene_index = {g: i for i, g in enumerate(rna_sym) if g is not None}
    sets = load_gmt(args.gmt, set(gene_index))
    print(f"device={device} n_test={len(test)}  유전자={len(gene_index)}  경로={len(sets)}")

    ytrue = {m: _get(test, m) for m in MODS}
    obs = {m: _obs(test, m) for m in MODS}
    split = pd.read_csv(d / "sample_split.tsv", sep="\t")
    test_ids = split.loc[split["split"] == "test", "sample"].tolist()

    er = load_group(args.clinical, test_ids, args.label_col, {"Positive": 1.0, "Negative": 0.0})
    pam = load_group(args.clinical, test_ids, args.subtype_col, None)
    if er is None:
        print("임상 이진 라벨 없음: 차등 발현은 건너뛴다")
    else:
        print(f"{args.label_col} n={int(er[1].sum())} 양성 {int(np.nansum(er[0][er[1]]))}")
    if pam is None:
        print("아형 라벨 없음: PAM50 실루엣은 건너뛴다")
    else:
        print(f"{args.subtype_col} n={int(pam[1].sum())} 수준 {pd.Series(pam[0][pam[1]]).nunique()}")

    print("fitting Ridge...")
    srcs = [s for s in MODS if s != "rna"]
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(
        np.concatenate([_get(train, s) for s in srcs], 1), _get(train, "rna"))

    models = {}
    for spec in args.runs:
        parts = spec.split("=")
        name, path = parts[0], parts[1]
        if not Path(path).exists():
            print(f"skip missing ckpt {name} {path}")
            continue
        mdl = load_nmf_tf(path, device)
        if len(parts) > 2 and parts[2] == "gamma0":
            if not mdl.add_residual:
                print(f"{name}: add_residual=False 이므로 gamma0 녹아웃은 무연산")
            mdl.set_lowrank(False)
        models[name] = mdl

    mimir = mimir_mv = None
    if args.mimir_dir and (Path(args.mimir_dir) / "shared_best.pt").exists():
        from mimir_wrap import frames_from_dir, load_mimir, predict_block
        mimir, mimir_mv = load_mimir(args.mimir_dir, device)
        tr_frames, _, _ = frames_from_dir(args.data_dir, "train")
        columns = {m: tr_frames[m].columns for m in MODS}
        ev_index = pd.Index(test_ids)

    masks = {m: (block_mask(obs[m]) if m == "rna" else np.zeros_like(obs[m], dtype=bool))
             for m in MODS}
    filled = apply_zero(ytrue, masks)
    cand = {
        "mean": np.zeros_like(ytrue["rna"]),
        "Ridge-2to1": ridge.predict(np.concatenate([filled[s] for s in srcs], 1)),
    }
    for name, mdl in models.items():
        cand[name] = predict_nmf_tf(mdl, filled, device, missing="rna")["rna"]
    if mimir is not None:
        nan_tabs = {}
        for m in MODS:
            x = ytrue[m].copy()
            x[~obs[m]] = np.nan
            x[masks[m]] = np.nan
            nan_tabs[m] = x
        present = {m: pd.DataFrame(nan_tabs[m], index=ev_index, columns=columns[m])
                   for m in MODS if m != "rna"}
        cand["MIMIR"] = predict_block(mimir, mimir_mv, present, "rna",
                                      columns["rna"], ev_index, device)

    ps_true = pathway_scores(ytrue["rna"], gene_index, sets)
    rows = []
    t_true = d_true = sig_true = None
    pw_t_true = pw_d_true = pw_sig_true = None
    if er is not None:
        y, ok = er
        t_true, p_true = welch_p(ytrue["rna"][ok], y[ok])
        sig_true = bh_q(p_true) < FDR
        d_true = cohens_d(ytrue["rna"][ok], y[ok])
        pw_t_true, pw_p = welch_p(ps_true[ok], y[ok])
        pw_sig_true = bh_q(pw_p) < FDR
        pw_d_true = cohens_d(ps_true[ok], y[ok])
        print(f"참값 유의 유전자 {int(sig_true.sum())}/{len(sig_true)}  "
              f"유의 경로 {int(pw_sig_true.sum())}/{len(pw_sig_true)}")

    for name, Xhat in cand.items():
        Xhat = np.asarray(Xhat, dtype=np.float64)
        ps_hat = pathway_scores(Xhat, gene_index, sets)
        r_path = col_corr(ps_true, ps_hat)
        sd_path = ps_hat.std(0) / np.maximum(1e-8, ps_true.std(0))
        rec = {
            "method": name,
            "pathway_r": float(np.mean(r_path)),
            "pathway_sd_ratio": float(np.mean(sd_path)),
        }
        rec.update(cluster_pack(ps_true, ps_hat))

        if er is not None:
            y, ok = er
            gene, _, _, _ = de_pack(Xhat[ok], y[ok], t_true, d_true, sig_true)
            pw, _, _, _ = de_pack(ps_hat[ok], y[ok], pw_t_true, pw_d_true, pw_sig_true)
            rec.update({f"gene_{k}": v for k, v in gene.items()})
            rec.update({f"pathway_{k}": v for k, v in pw.items()})
            try:
                rec["er_silhouette"] = float(silhouette_score(ps_hat[ok], y[ok]))
            except ValueError:
                rec["er_silhouette"] = np.nan

        if pam is not None:
            lab, ok = pam
            codes = pd.Series(lab[ok]).astype("category").cat.codes.to_numpy()
            k = int(pd.Series(codes).nunique())
            if k >= 2 and ok.sum() > k:
                km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(ps_hat[ok])
                rec["subtype_ari"] = float(adjusted_rand_score(codes, km.labels_))
                try:
                    rec["subtype_silhouette"] = float(silhouette_score(ps_hat[ok], codes))
                except ValueError:
                    rec["subtype_silhouette"] = np.nan

        rows.append(rec)
        print(f"[{name}] sd={rec['pathway_sd_ratio']:.3f} ari={rec['ari_mean']:.3f}")

    if er is not None:
        y, ok = er
        rec = {"method": "oracle", "pathway_r": 1.0, "pathway_sd_ratio": 1.0}
        rec.update(cluster_pack(ps_true, ps_true))
        gene, _, _, _ = de_pack(ytrue["rna"][ok], y[ok], t_true, d_true, sig_true)
        pw, _, _, _ = de_pack(ps_true[ok], y[ok], pw_t_true, pw_d_true, pw_sig_true)
        rec.update({f"gene_{k}": v for k, v in gene.items()})
        rec.update({f"pathway_{k}": v for k, v in pw.items()})
        rec["er_silhouette"] = float(silhouette_score(ps_true[ok], y[ok]))
        if pam is not None:
            lab, pok = pam
            codes = pd.Series(lab[pok]).astype("category").cat.codes.to_numpy()
            k = int(pd.Series(codes).nunique())
            km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(ps_true[pok])
            rec["subtype_ari"] = float(adjusted_rand_score(codes, km.labels_))
            rec["subtype_silhouette"] = float(silhouette_score(ps_true[pok], codes))
        rows.append(rec)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "amplitude.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))
    print(f"\nsaved {out / 'amplitude.tsv'}")


if __name__ == "__main__":
    main()
