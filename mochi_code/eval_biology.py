#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대치 결과가 생물학적 의미를 보존하는지 본다.

블록 결측(타깃 오믹스 전체 제거)에서 채운 값을 놓고 네 가지를 잰다.
  1) Hallmark 경로 활성 보존 — 경로 점수의 환자 간 상관과 분산 유지
  2) 환자 간 유사도 구조 보존 — 표본×표본 상관행렬과 k-최근접 이웃 일치
  3) ER 양성/음성 차등 발현 보존 — 유전자별 t-통계량의 재현
  4) RNA-단백질 대응 보존 — 같은 유전자 쌍의 교차 오믹스 상관 유지
추가로 고정 NMF 성분이 어떤 경로에 대응하는지 과대표현으로 확인한다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, ttest_ind
from sklearn.linear_model import RidgeCV

from compare_gate import RIDGE_ALPHAS
from eval_mcar_mnar import MODS, apply_zero
from eval_v5_noise import _get, _obs
from missingness import block_mask
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset

# RPPA 총단백 항체 -> 유전자 심볼. 인산화 항체는 제외한다.
AB2GENE = {
    "ERALPHA": "ESR1", "PR": "PGR", "HER2": "ERBB2", "HER3": "ERBB3", "GATA3": "GATA3",
    "AR": "AR", "ECADHERIN": "CDH1", "NCADHERIN": "CDH2", "PCADHERIN": "CDH3",
    "EGFR": "EGFR", "CMYC": "MYC", "CYCLIND1": "CCND1", "CYCLINE1": "CCNE1",
    "CYCLINE2": "CCNE2", "CYCLINB1": "CCNB1", "BCL2": "BCL2", "BCLXL": "BCL2L1",
    "PTEN": "PTEN", "RB": "RB1", "CAVEOLIN1": "CAV1", "FASN": "FASN",
    "INPP4B": "INPP4B", "ASNS": "ASNS", "CLAUDIN7": "CLDN7", "ANNEXIN1": "ANXA1",
    "CD20": "MS4A1", "CD31": "PECAM1", "CD44": "CD44", "CD45": "PTPRC", "CD4": "CD4",
    "COLLAGENVI": "COL6A1", "FIBRONECTIN": "FN1", "IGFBP2": "IGFBP2", "MITF": "MITF",
    "SCD1": "SCD", "SFRP1": "SFRP1", "SNAIL": "SNAI1", "TFRC": "TFRC", "XBP1": "XBP1",
    "YAP": "YAP1", "EPPK1": "EPPK1", "DIRAS3": "DIRAS3", "RAB25": "RAB25",
    "MSH2": "MSH2", "MSH6": "MSH6", "MLH1": "MLH1", "PDL1": "CD274", "CA9": "CA9",
    "GATA6": "GATA6", "PAX8": "PAX8", "TTF1": "NKX2-1", "NAPSINA": "NAPSA",
    "VHL": "VHL", "SYK": "SYK", "LCK": "LCK", "JAK2": "JAK2", "NOTCH1": "NOTCH1",
    "Notch3": "NOTCH3", "EZH2": "EZH2", "FOXM1": "FOXM1", "ETS1": "ETS1",
    "IRF1": "IRF1", "STATHMIN": "STMN1", "Twist": "TWIST1", "ZEB1": "ZEB1",
    "PAI1": "SERPINE1", "GRB7": "GRB7", "LDHA": "LDHA", "LDHB": "LDHB",
    "PKM2": "PKM", "G6PD": "G6PD", "PHGDH": "PHGDH", "TFAM": "TFAM",
    "SOD1": "SOD1", "SOD2": "SOD2", "NQO1": "NQO1", "DNMT1": "DNMT1",
    "BRD4": "BRD4", "ARID1A": "ARID1A", "ATRX": "ATRX", "SETD2": "SETD2",
    "PARP1": "PARP1", "RAD50": "RAD50", "RAD51": "RAD51", "MRE11": "MRE11",
    "ATM": "ATM", "ATR": "ATR", "CHK1": "CHEK1", "CHK2": "CHEK2", "Wee1": "WEE1",
    "PCNA": "PCNA", "RRM1": "RRM1", "RRM2": "RRM2", "CDK1": "CDK1", "PLK1": "PLK1",
    "Aurora-A": "AURKA", "Aurora-B": "AURKB", "CKIT": "KIT", "CMET": "MET",
    "AXL": "AXL", "DDR1": "DDR1", "EphA2": "EPHA2", "PDGFRB": "PDGFRB",
    "VEGFR2": "KDR", "IDO": "IDO1", "MMP2": "MMP2", "MMP14": "MMP14",
    "S100A4": "S100A4", "HSP27": "HSPB1", "HSP70": "HSPA1A", "MIF": "MIF",
    "P16INK4A": "CDKN2A", "P21": "CDKN1A", "P27": "CDKN1B", "P53": "TP53",
    "P63": "TP63", "PDCD1": "PDCD1", "CTLA4": "CTLA4", "Granzyme-B": "GZMB",
    "IL-6": "IL6", "Jagged1": "JAG1", "HES1": "HES1", "LAD1": "LAD1",
    "MYH11": "MYH11", "SLC1A5": "SLC1A5", "UGT1A": "UGT1A1", "B7-H3": "CD276",
    "B7-H4": "VTCN1", "CD26": "DPP4", "CD38": "CD38", "CD86": "CD86",
    "MCT4": "SLC16A3", "PYGB": "PYGB", "PYGL": "PYGL", "PYGM": "PYGM",
}
N_KNN = 10
TOP_GENES = 100


def load_gmt(path, universe):
    sets = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        genes = {g for g in parts[2:] if g in universe}
        if len(genes) >= 10:
            sets[parts[0]] = genes
    return sets


def pathway_scores(X, gene_index, sets):
    """경로별 평균 z 점수. X는 이미 표준화된 행렬이다."""
    out = np.zeros((X.shape[0], len(sets)), dtype=np.float64)
    for j, (_, genes) in enumerate(sets.items()):
        cols = [gene_index[g] for g in genes if g in gene_index]
        out[:, j] = X[:, cols].mean(1)
    return out


def col_corr(a, b):
    """열별 Pearson 상관."""
    az = a - a.mean(0)
    bz = b - b.mean(0)
    num = (az * bz).sum(0)
    den = np.sqrt((az ** 2).sum(0) * (bz ** 2).sum(0))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def sample_similarity(X):
    C = np.corrcoef(X)
    return np.nan_to_num(C)


def knn_overlap(Ct, Cp, k=N_KNN):
    n = Ct.shape[0]
    k = min(k, n - 1)
    scores = []
    for i in range(n):
        t = np.argsort(-Ct[i]);  t = [j for j in t if j != i][:k]
        p = np.argsort(-Cp[i]);  p = [j for j in p if j != i][:k]
        scores.append(len(set(t) & set(p)) / k)
    return float(np.mean(scores))


def welch_t(X, y):
    a, b = X[y == 1], X[y == 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ttest_ind(a, b, axis=0, equal_var=False).statistic
    return np.nan_to_num(np.asarray(t, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    clin = "/home/dyan/nmf/mochi_code/processed_data/clinical"
    ap.add_argument("--probemap", default=f"{clin}/gencode.probeMap")
    ap.add_argument("--gmt", default=f"{clin}/hallmark.gmt")
    ap.add_argument("--clinical", default=f"{clin}/BRCA_clinicalMatrix")
    ap.add_argument("--label_col", default="ER_Status_nature2012")
    ap.add_argument("--methylmap", default=f"{clin}/methyl450.probeMap")
    ap.add_argument("--mimir_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_biology")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--runs", nargs="+", default=[
        "MOCHI=/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt",
        # 같은 가중치에서 저랭크 잔차만 끈 녹아웃. 그림 1 Panel B가 이 행을 쓴다.
        # 기본 실행에 넣어야 진폭(pathway_sd_ratio)과 함께 순서(pathway_r)에도
        # 무슨 일이 일어나는지 매번 같이 나온다 — 둘 중 하나만 보고하면 안 된다.
        "MOCHI-knockout=/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt=gamma0",
        "MOCHI-nogan=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nogan/nmf_tf_best.ckpt",
        "MOCHI-notf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_mean/nmf_tf_best.ckpt",
        "MOCHI-nonmf=/home/dyan/nmf/mochi_code/results/current/gate_ablate_nonmf/nmf_tf_best.ckpt",
    ])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = TripleSplitDataset(args.data_dir, "train")
    test = TripleSplitDataset(args.data_dir, "test", stats=train.stats)
    d = Path(args.data_dir)
    feat = {
        "rna": pd.read_csv(d / "rna.train.tsv", sep="\t", index_col=0, usecols=[0]).index,
        "protein": pd.read_csv(d / "protein.train.tsv", sep="\t", index_col=0, usecols=[0]).index,
        "methyl": pd.read_csv(d / "methy.train.tsv", sep="\t", index_col=0, usecols=[0]).index,
    }

    pm = pd.read_csv(args.probemap, sep="\t")
    pm["b"] = pm["id"].str.split(".").str[0]
    ens2sym = pm.drop_duplicates("b").set_index("b")["gene"]
    rna_sym = pd.Index([ens2sym.get(i.split(".")[0], None) for i in feat["rna"]])
    # probeMap에 심볼이 비어 있거나 NaN이면 pandas가 float로 넣는다.
    # 그대로 gene_index에 들어가면 성분 상위 유전자 정렬이 str/float 혼합으로 죽는다.
    gene_index = {g: i for i, g in enumerate(rna_sym) if isinstance(g, str) and g}
    sets = load_gmt(args.gmt, set(gene_index))
    print(f"device={device} n_test={len(test)}  매핑 유전자={len(gene_index)}  경로={len(sets)}")

    prot_pairs = []
    for j, ab in enumerate(feat["protein"]):
        g = AB2GENE.get(ab)
        if g is not None and g in gene_index:
            prot_pairs.append((j, gene_index[g], ab, g))
    print(f"RNA-단백질 짝지은 유전자 쌍={len(prot_pairs)}")

    ytrue = {m: _get(test, m) for m in MODS}
    obs = {m: _obs(test, m) for m in MODS}

    split = pd.read_csv(d / "sample_split.tsv", sep="\t")
    test_ids = split.loc[split["split"] == "test", "sample"].tolist()
    er, er_ok = None, np.zeros(len(test_ids), dtype=bool)
    if args.clinical and Path(args.clinical).exists():
        cm = pd.read_csv(args.clinical, sep="\t", index_col=0, low_memory=False)
        cm.index = [str(i)[:15] for i in cm.index]
        cm = cm[~cm.index.duplicated()]
        if args.label_col in cm.columns:
            er = cm.reindex([s[:15] for s in test_ids])[args.label_col] \
                   .map({"Positive": 1.0, "Negative": 0.0}).to_numpy(float)
            er_ok = np.isfinite(er)
            print(f"{args.label_col} 검사 집합 n={int(er_ok.sum())} (양성 {int(np.nansum(er))})")
    if er is None:
        print("임상 라벨 없음: 차등 발현 보존은 건너뛴다")

    print("fitting Ridge...")
    ridge2 = {}
    for tgt in MODS:
        srcs = [s for s in MODS if s != tgt]
        X = np.concatenate([_get(train, s) for s in srcs], 1)
        ridge2[tgt] = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, _get(train, tgt))

    models = {}
    for spec in args.runs:
        parts = spec.split("=")
        name, path = parts[0], parts[1]
        if not Path(path).exists():
            continue
        mdl = load_nmf_tf(path, device)
        # 세 번째 칸이 gamma0이면 저랭크 잔차를 끈다.
        # gamma.zero_()는 gamma_nonneg에서 softplus(0)=0.693이 되어 녹아웃이 아니다.
        if len(parts) > 2 and parts[2] == "gamma0":
            if not mdl.add_residual:
                print(f"{name}: add_residual=False 이므로 gamma0 녹아웃은 무연산")
            mdl.set_lowrank(False)
        models[name] = mdl

    mimir = mimir_mv = None
    if args.mimir_dir and (Path(args.mimir_dir) / "shared_best.pt").exists():
        from mimir_wrap import frames_from_dir, load_mimir, predict_block
        mimir, mimir_mv = load_mimir(args.mimir_dir, device)
        tr_frames, stats_df, _ = frames_from_dir(args.data_dir, "train")
        columns = {m: tr_frames[m].columns for m in MODS}
        ev_index = pd.Index(test_ids)

    def imputations(tgt):
        masks = {m: (block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool))
                 for m in MODS}
        filled = apply_zero(ytrue, masks)
        cand = {"mean": np.zeros_like(ytrue[tgt])}
        cand["Ridge-2to1"] = ridge2[tgt].predict(
            np.concatenate([filled[s] for s in MODS if s != tgt], 1))
        for name, mdl in models.items():
            cand[name] = predict_nmf_tf(mdl, filled, device, missing=tgt)[tgt]
        if mimir is not None:
            nan_tabs = {}
            for m in MODS:
                x = ytrue[m].copy()
                x[~obs[m]] = np.nan
                x[masks[m]] = np.nan
                nan_tabs[m] = x
            present = {m: pd.DataFrame(nan_tabs[m], index=ev_index, columns=columns[m])
                       for m in MODS if m != tgt}
            cand["MIMIR"] = predict_block(mimir, mimir_mv, present, tgt,
                                          columns[tgt], ev_index, device)
        return cand

    rows = []

    # ---- RNA 블록: 경로 활성, 차등 발현, 환자 유사도 ----
    rna_imp = imputations("rna")
    ps_true = pathway_scores(ytrue["rna"], gene_index, sets)
    Ct_rna = sample_similarity(ytrue["rna"])
    t_true = welch_t(ytrue["rna"][er_ok], er[er_ok]) if er_ok.sum() > 10 else None
    if t_true is not None and len(np.unique(er[er_ok])) < 2:
        t_true = None
    top_true = set(np.argsort(-np.abs(t_true))[:TOP_GENES]) if t_true is not None else set()

    for name, Xhat in rna_imp.items():
        Xhat = np.asarray(Xhat, dtype=np.float64)
        ps_hat = pathway_scores(Xhat, gene_index, sets)
        r_path = col_corr(ps_true, ps_hat)
        sd_path = ps_hat.std(0) / np.maximum(1e-8, ps_true.std(0))
        Cp = sample_similarity(Xhat)
        iu = np.triu_indices_from(Ct_rna, k=1)
        rec = {
            "modality": "rna", "method": name,
            "pathway_r": float(np.mean(r_path)),
            "pathway_sd_ratio": float(np.mean(sd_path)),
            "pathway_r_min": float(np.min(r_path)),
            "patient_sim_r": float(np.corrcoef(Ct_rna[iu], Cp[iu])[0, 1]),
            "knn_overlap": knn_overlap(Ct_rna, Cp),
        }
        if t_true is not None:
            t_hat = welch_t(Xhat[er_ok], er[er_ok])
            rec["de_t_r"] = float(np.corrcoef(t_true, t_hat)[0, 1])
            top_hat = set(np.argsort(-np.abs(t_hat))[:TOP_GENES])
            rec["de_top100_overlap"] = len(top_true & top_hat) / TOP_GENES
        rows.append(rec)
        print(f"[rna] {name} done")

    # ---- 단백질 블록: RNA-단백질 대응, 환자 유사도 ----
    prot_imp = imputations("protein")
    Ct_p = sample_similarity(ytrue["protein"])
    pj = [p[0] for p in prot_pairs]
    gj = [p[1] for p in prot_pairs]
    r_pair_true = col_corr(ytrue["rna"][:, gj], ytrue["protein"][:, pj])
    for name, Xhat in prot_imp.items():
        Xhat = np.asarray(Xhat, dtype=np.float64)
        r_pair_hat = col_corr(ytrue["rna"][:, gj], Xhat[:, pj])
        Cp = sample_similarity(Xhat)
        iu = np.triu_indices_from(Ct_p, k=1)
        rows.append({
            "modality": "protein", "method": name,
            "rna_prot_r_true": float(np.mean(r_pair_true)),
            "rna_prot_r_imp": float(np.mean(r_pair_hat)),
            "rna_prot_recovery": float(np.mean(r_pair_hat) / max(1e-8, np.mean(r_pair_true))),
            "rna_prot_profile_r": float(np.corrcoef(r_pair_true, r_pair_hat)[0, 1]),
            "patient_sim_r": float(np.corrcoef(Ct_p[iu], Cp[iu])[0, 1]),
            "knn_overlap": knn_overlap(Ct_p, Cp),
        })
        print(f"[protein] {name} done")

    # ---- 메틸화 블록: 환자 유사도 + 메틸화-발현 결합 ----
    methy_imp = imputations("methyl")
    Ct_m = sample_similarity(ytrue["methyl"])
    cg_pairs = []
    if args.methylmap and Path(args.methylmap).exists():
        mp = pd.read_csv(args.methylmap, sep="\t", usecols=[0, 1])
        mp.columns = ["probe", "gene"]
        gmap = mp.drop_duplicates("probe").set_index("probe")["gene"]
        for j, cg in enumerate(feat["methyl"]):
            g = gmap.get(cg)
            if not isinstance(g, str):
                continue
            for sym in g.split(","):
                if sym in gene_index:
                    cg_pairs.append((j, gene_index[sym]))
                    break
        print(f"메틸화-발현 짝지은 프로브 쌍={len(cg_pairs)}")
    mj = [p[0] for p in cg_pairs]
    gj2 = [p[1] for p in cg_pairs]
    r_me_true = col_corr(ytrue["methyl"][:, mj], ytrue["rna"][:, gj2]) if cg_pairs else None

    for name, Xhat in methy_imp.items():
        Xhat = np.asarray(Xhat, dtype=np.float64)
        Cp = sample_similarity(Xhat)
        iu = np.triu_indices_from(Ct_m, k=1)
        rec = {
            "modality": "methyl", "method": name,
            "patient_sim_r": float(np.corrcoef(Ct_m[iu], Cp[iu])[0, 1]),
            "knn_overlap": knn_overlap(Ct_m, Cp),
        }
        if r_me_true is not None:
            r_hat = col_corr(Xhat[:, mj], ytrue["rna"][:, gj2])
            rec["me_r_true"] = float(np.mean(r_me_true))
            rec["me_r_imp"] = float(np.mean(r_hat))
            rec["me_profile_r"] = float(np.corrcoef(r_me_true, r_hat)[0, 1])
            rec["me_sign_agree"] = float(np.mean(np.sign(r_me_true) == np.sign(r_hat)))
            neg = r_me_true < -0.2
            if neg.sum() > 5:
                rec["me_neg_recovery"] = float(np.mean(r_hat[neg]) / np.mean(r_me_true[neg]))
        rows.append(rec)
        print(f"[methyl] {name} done")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "biology.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"saved {out / 'biology.tsv'}")

    # ---- 고정 NMF 성분의 경로 과대표현 ----
    # 표 10의 숫자는 위에서 이미 썼다. 성분 과대표현이 죽어도 본문 지표는 남긴다.
    comp_rows = []
    mochi = models.get("MOCHI")
    if mochi is not None:
        H = mochi.tokenizers["rna"].H.detach().cpu().numpy()
        universe = set(gene_index)
        M = len(universe)
        for c in range(H.shape[0]):
            order = np.argsort(-H[c])
            top = [rna_sym[i] for i in order if isinstance(rna_sym[i], str)
                   and rna_sym[i] in universe][:TOP_GENES]
            top = set(top)
            best = None
            for pw, genes in sets.items():
                k = len(top & genes)
                if k < 3:
                    continue
                p = hypergeom.sf(k - 1, M, len(genes), len(top))
                if best is None or p < best[1]:
                    best = (pw, float(p), k)
            names = sorted(str(x) for x in top)
            comp_rows.append({
                "component": c,
                "top_genes": ",".join(names[:8]),
                "pathway": best[0] if best else "",
                "p": best[1] if best else np.nan,
                "overlap": best[2] if best else 0,
            })

    if comp_rows:
        cdf = pd.DataFrame(comp_rows).sort_values("p")
        cdf.to_csv(out / "nmf_components.tsv", sep="\t", index=False, float_format="%.3g")

    pd.set_option("display.width", 200)
    for mod in ("rna", "protein", "methyl"):
        sub = df[df["modality"] == mod].dropna(axis=1, how="all")
        print(f"\n=== {mod} 블록 결측 ===")
        print(sub.drop(columns=["modality"]).to_string(index=False))
    if comp_rows:
        print("\n=== NMF 성분 상위 경로 (상위 10개 성분) ===")
        print(cdf.head(10)[["component", "pathway", "overlap", "p"]].to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
