#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게이트 실험용 BRCA 데이터 준비.

- PAM50 필터 없음
- 원발 종양(01)만, 15자리 바코드로 RNA ∩ 메틸화 ∩ 단백질 교집합
- train/val/test 분할 후, 분산 top-k는 train에서만 계산 (누수 방지)
"""

import argparse
import gzip
import heapq
from pathlib import Path

import numpy as np
import pandas as pd


def norm15(s: str) -> str:
    return str(s).split(".")[0][:15]


def is_tumor_01(s: str) -> bool:
    s = str(s).split(".")[0]
    return len(s) >= 15 and s[13:15] == "01"


def read_header(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
    return cols[0], cols[1:]


def pick_columns(sample_ids, keep_15):
    """15자리 -> 원본 컬럼명. 중복이면 첫 번째."""
    mapping = {}
    for sid in sample_ids:
        key = norm15(sid)
        if key in keep_15 and key not in mapping and is_tumor_01(sid):
            mapping[key] = sid
    return mapping


def split_ids(ids, seed=42, ratios=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(ids))
    perm = rng.permutation(len(ids))
    ids = ids[perm]
    n = len(ids)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    train = ids[:n_tr].tolist()
    val = ids[n_tr:n_tr + n_va].tolist()
    test = ids[n_tr + n_va:].tolist()
    return train, val, test


def load_subset(path, col_map, ordered_keys):
    usecols = [None]
    # pandas: first column is index
    names = None
    header_feat, header_ids = read_header(path)
    wanted = set(col_map[k] for k in ordered_keys)
    usecols = [header_feat] + [c for c in header_ids if c in wanted]
    df = pd.read_csv(path, sep="\t", index_col=0, usecols=usecols)
    # 15자리 순서로 정렬
    rename = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename)
    df = df[ordered_keys]
    return df.astype(np.float32)


def topk_by_train_var(df, train_ids, k):
    if k is None or k >= df.shape[0]:
        return df
    var = df[train_ids].var(axis=1, skipna=True)
    keep = var.sort_values(ascending=False).head(k).index
    return df.loc[keep]


def stream_methy_topk(path, col_map, ordered_keys, train_keys, k):
    """메틸화는 파일이 커서 한 줄씩 읽으며 train 분산 top-k만 유지."""
    header_feat, header_ids = read_header(path)
    idx = {c: i + 1 for i, c in enumerate(header_ids)}  # +1: feature col
    col_pos = [idx[col_map[k]] for k in ordered_keys]
    train_pos = [j for j, key in enumerate(ordered_keys) if key in set(train_keys)]

    heap = []  # min-heap (var, feat, values)
    opener = gzip.open if path.endswith(".gz") else open
    n_seen = 0
    with opener(path, "rt") as f:
        next(f)
        for line in f:
            n_seen += 1
            parts = line.rstrip("\n").split("\t")
            feat = parts[0]
            vals = np.empty(len(col_pos), dtype=np.float32)
            for j, p in enumerate(col_pos):
                tok = parts[p]
                vals[j] = np.nan if tok in ("", "NA", "NaN") else float(tok)
            tr = vals[train_pos]
            if np.isfinite(tr).sum() < 10:
                continue
            v = float(np.nanvar(tr))
            item = (v, feat, vals)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif v > heap[0][0]:
                heapq.heapreplace(heap, item)
            if n_seen % 50000 == 0:
                print(f"  methy scanned {n_seen} probes, heap={len(heap)}")

    heap.sort(key=lambda x: -x[0])
    feats = [h[1] for h in heap]
    mat = np.stack([h[2] for h in heap], axis=0)
    df = pd.DataFrame(mat, index=feats, columns=ordered_keys)
    print(f"  methy kept {df.shape[0]} / {n_seen} probes")
    return df


def save_split_matrices(df, train, val, test, out_dir, prefix):
    out_dir = Path(out_dir)
    df[train].to_csv(out_dir / f"{prefix}.train.tsv", sep="\t")
    df[val].to_csv(out_dir / f"{prefix}.val.tsv", sep="\t")
    df[test].to_csv(out_dir / f"{prefix}.test.tsv", sep="\t")
    print(f"  saved {prefix}: {df.shape[0]} features x {df.shape[1]} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna", default="/home/dyan/TCGA-BRCA.star_tpm.tsv.gz")
    ap.add_argument("--methy", default="/home/dyan/TCGA-BRCA.methylation450.tsv.gz")
    ap.add_argument("--protein", default="/home/dyan/TCGA-BRCA.protein.tsv.gz")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--rna_k", type=int, default=2000)
    ap.add_argument("--methy_k", type=int, default=5000)
    ap.add_argument("--protein_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _, rna_ids = read_header(args.rna)
    _, methy_ids = read_header(args.methy)
    _, prot_ids = read_header(args.protein)

    rna15 = {norm15(s) for s in rna_ids if is_tumor_01(s)}
    methy15 = {norm15(s) for s in methy_ids if is_tumor_01(s)}
    prot15 = {norm15(s) for s in prot_ids if is_tumor_01(s)}
    common = sorted(rna15 & methy15 & prot15)
    print(f"tumor 01 triple intersection (15-char): {len(common)}")

    rna_map = pick_columns(rna_ids, set(common))
    methy_map = pick_columns(methy_ids, set(common))
    prot_map = pick_columns(prot_ids, set(common))
    common = [k for k in common if k in rna_map and k in methy_map and k in prot_map]
    print(f"after unique column pick: {len(common)}")

    train, val, test = split_ids(common, seed=args.seed)
    split_df = pd.DataFrame(
        [{"sample": s, "split": sp} for sp, ids in [("train", train), ("val", val), ("test", test)] for s in ids]
    )
    split_df.to_csv(out / "sample_split.tsv", sep="\t", index=False)
    print(f"split train/val/test = {len(train)}/{len(val)}/{len(test)}")

    print("loading RNA...")
    rna = load_subset(args.rna, rna_map, common)
    rna = topk_by_train_var(rna, train, args.rna_k)
    save_split_matrices(rna, train, val, test, out, "rna")

    print("loading protein...")
    prot = load_subset(args.protein, prot_map, common)
    prot = topk_by_train_var(prot, train, args.protein_k)
    save_split_matrices(prot, train, val, test, out, "protein")

    print("streaming methylation top-k on train variance...")
    methy = stream_methy_topk(args.methy, methy_map, common, train, args.methy_k)
    save_split_matrices(methy, train, val, test, out, "methy")

    meta = pd.DataFrame([
        {"modality": "rna", "n_features": rna.shape[0], "n_samples": rna.shape[1], "na_rate": float(rna.isna().mean().mean())},
        {"modality": "protein", "n_features": prot.shape[0], "n_samples": prot.shape[1], "na_rate": float(prot.isna().mean().mean())},
        {"modality": "methy", "n_features": methy.shape[0], "n_samples": methy.shape[1], "na_rate": float(methy.isna().mean().mean())},
    ])
    meta.to_csv(out / "feature_summary.tsv", sep="\t", index=False)
    print(meta.to_string(index=False))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
