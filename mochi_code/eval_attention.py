#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
그림 1 재료: 타깃 오믹스를 채울 때 모형이 어떤 NMF 성분을 보는가.

블록 결측 상황에서 융합 질의 q가 토큰들에 주는 어텐션 가중치를 뽑는다.
성분 토큰에는 hallmark 과대표현으로 붙인 경로 이름을 달아, 어떤 생물학
축을 읽어 타깃을 복원하는지 보인다. 히트맵 두 장과 표를 저장한다.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom

from eval_biology import load_gmt
from eval_mcar_mnar import MODS, apply_zero
from eval_v5_noise import _get, _obs
from missingness import block_mask
from models_nmf_tf import load_nmf_tf
from train_gate import TripleSplitDataset

TOP_GENES = 100
SHORT = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": "IFN-a",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": "IFN-g",
    "HALLMARK_ALLOGRAFT_REJECTION": "Immune",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY": "Estrogen",
    "HALLMARK_ESTROGEN_RESPONSE_LATE": "Estrogen late",
    "HALLMARK_G2M_CHECKPOINT": "G2M",
    "HALLMARK_E2F_TARGETS": "E2F",
    "HALLMARK_MYC_TARGETS_V1": "MYC",
    "HALLMARK_ADIPOGENESIS": "Adipogenesis",
    "HALLMARK_APICAL_JUNCTION": "Apical junction",
    "HALLMARK_KRAS_SIGNALING_UP": "KRAS up",
    "HALLMARK_KRAS_SIGNALING_DN": "KRAS dn",
    "HALLMARK_MYOGENESIS": "Myogenesis",
    "HALLMARK_COAGULATION": "Coagulation",
    "HALLMARK_COMPLEMENT": "Complement",
    "HALLMARK_INFLAMMATORY_RESPONSE": "Inflammation",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": "TNFa/NFkB",
    "HALLMARK_HYPOXIA": "Hypoxia",
    "HALLMARK_GLYCOLYSIS": "Glycolysis",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
    "HALLMARK_FATTY_ACID_METABOLISM": "Fatty acid",
    "HALLMARK_XENOBIOTIC_METABOLISM": "Xenobiotic",
    "HALLMARK_BILE_ACID_METABOLISM": "Bile acid",
    "HALLMARK_P53_PATHWAY": "p53",
    "HALLMARK_APOPTOSIS": "Apoptosis",
    "HALLMARK_UV_RESPONSE_DN": "UV dn",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "IL6/JAK/STAT3",
    "HALLMARK_IL2_STAT5_SIGNALING": "IL2/STAT5",
    "HALLMARK_TGF_BETA_SIGNALING": "TGF-b",
    "HALLMARK_ANGIOGENESIS": "Angiogenesis",
    "HALLMARK_APICAL_SURFACE": "Apical surface",
    "HALLMARK_PROTEIN_SECRETION": "Secretion",
    "HALLMARK_MTORC1_SIGNALING": "mTORC1",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "UPR",
}


def annotate_components(model, mod, symbols, sets, universe):
    """성분마다 가장 유의한 hallmark 경로를 붙인다."""
    H = model.tokenizers[mod].H.detach().cpu().numpy()
    M = len(universe)
    labels = []
    for c in range(H.shape[0]):
        order = np.argsort(-H[c])
        top = []
        for i in order:
            g = symbols[i] if symbols is not None else None
            if g in universe:
                top.append(g)
            if len(top) >= TOP_GENES:
                break
        top = set(top)
        best = ("", 1.0, 0)
        for pw, genes in sets.items():
            k = len(top & genes)
            if k < 3:
                continue
            p = hypergeom.sf(k - 1, M, len(genes), len(top))
            if p < best[1]:
                best = (pw, float(p), k)
        name = SHORT.get(best[0], best[0].replace("HALLMARK_", "").title()[:16])
        labels.append(f"{mod[:1].upper()}{c:02d} {name}" if name else f"{mod[:1].upper()}{c:02d}")
    return labels


@torch.no_grad()
def attention_for_target(model, tabs, device, target, batch_size=64):
    """타깃 블록 결측일 때 질의 q의 토큰별 어텐션 가중치 평균."""
    fuse = model.fuse
    present_mods = [m for m in MODS if m != target]
    n = tabs[present_mods[0]].shape[0]
    acc = []
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
        present[target] = torch.zeros(b, dtype=torch.bool, device=device)
        hs = model.encode_h(xs)
        Ws = model.encode_W(xs)
        keep = dict(present)
        tokens, pad = fuse._stack(hs, Ws, keep)
        memory = fuse.encoder(tokens, src_key_padding_mask=pad)
        q = fuse.query.expand(tokens.size(0), -1, -1)
        _, w = fuse.attn(q, memory, memory, key_padding_mask=pad,
                         need_weights=True, average_attn_weights=True)
        acc.append(w.squeeze(1).cpu().numpy())
    return np.concatenate(acc, 0)


def token_labels(model, comp_labels):
    """_stack 순서와 같은 토큰 이름표를 만든다."""
    names, kinds, mods = [], [], []
    for m in model.fuse.mods:
        names.append(f"{m} 은닉")
        kinds.append("hidden")
        mods.append(m)
        if model.fuse.use_nmf_tokens:
            for c in range(model.fuse.k):
                names.append(comp_labels[m][c])
                kinds.append("component")
                mods.append(m)
    return names, kinds, mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    clin = "/home/dyan/nmf/mochi_code/processed_data/clinical"
    ap.add_argument("--probemap", default=f"{clin}/gencode.probeMap")
    ap.add_argument("--gmt", default=f"{clin}/hallmark.gmt")
    ap.add_argument("--ckpt",
                    default="/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h/nmf_tf_best.ckpt")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_attention")
    ap.add_argument("--cohort", default="BRCA")
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
    universe = {g for g in rna_sym if g is not None}
    sets = load_gmt(args.gmt, universe)
    model = load_nmf_tf(args.ckpt, device)
    print(f"device={device} n_test={len(test)} 경로={len(sets)} k={model.fuse.k}")

    comp_labels = {"rna": annotate_components(model, "rna", rna_sym, sets, universe)}
    for m in ("protein", "methyl"):
        comp_labels[m] = [f"{m[:1].upper()}{c:02d}" for c in range(model.fuse.k)]
    names, kinds, mods = token_labels(model, comp_labels)

    ytrue = {m: _get(test, m) for m in MODS}
    obs = {m: _obs(test, m) for m in MODS}

    rows = []
    per_target = {}
    for tgt in MODS:
        masks = {m: (block_mask(obs[m]) if m == tgt else np.zeros_like(obs[m], dtype=bool))
                 for m in MODS}
        filled = apply_zero(ytrue, masks)
        W = attention_for_target(model, filled, device, tgt)
        w = W.mean(0)
        per_target[tgt] = w
        for j, nm in enumerate(names):
            rows.append({"target": tgt, "token": nm, "kind": kinds[j],
                         "source": mods[j], "weight": float(w[j])})
        print(f"[{tgt}] 어텐션 추출 완료, 토큰 {W.shape[1]}개")

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "attention.tsv", sep="\t", index=False, float_format="%.6f")

    # 관측 소스의 RNA 성분만 모아 히트맵
    keep = df[(df["kind"] == "component") & (df["source"] == "rna")]
    mat = keep.pivot_table(index="token", columns="target", values="weight")
    mat = mat[[c for c in ("protein", "methyl", "rna") if c in mat.columns]]
    order = mat[["protein", "methyl"]].mean(1).sort_values(ascending=False).index
    mat = mat.loc[order]

    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(mat) + 1.4))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([{"protein": "단백질 복원", "methyl": "메틸화 복원",
                         "rna": "RNA 복원"}.get(c, c) for c in mat.columns])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(f"{args.cohort}: RNA NMF 성분에 대한 어텐션", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label="평균 어텐션 가중치")
    fig.tight_layout()
    fig.savefig(out / "fig1_component_attention.png", dpi=200)
    plt.close(fig)

    # 소스 종류별 총 어텐션 배분
    agg = (df.groupby(["target", "source", "kind"])["weight"].sum().reset_index())
    agg["label"] = agg["source"] + " " + agg["kind"].map({"hidden": "은닉", "component": "성분"})
    piv = agg.pivot_table(index="label", columns="target", values="weight").fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    piv.plot(kind="barh", ax=ax)
    ax.set_xlabel("어텐션 가중치 합")
    ax.set_ylabel("")
    ax.set_title(f"{args.cohort}: 복원 타깃별 어텐션 배분", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1b_attention_budget.png", dpi=200)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n=== 단백질 복원 시 상위 RNA 성분 ===")
    print(mat.sort_values("protein", ascending=False).head(8).round(4).to_string())
    print("\n=== 어텐션 배분 ===")
    print(piv.round(4).to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
