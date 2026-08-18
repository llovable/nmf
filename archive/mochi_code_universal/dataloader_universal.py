#!/usr/bin/env python3
"""
범용 멀티오믹스 데이터 로더 (샘플 수/특징 수 가변 지원)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _normalize_sample_ids(ids: List[str], mode: str) -> List[str]:
    if mode == "tcga12":
        return [str(s)[:12] for s in ids]
    if mode == "tcga15":
        return [str(s)[:15] for s in ids]
    return [str(s) for s in ids]


def _dedupe_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.columns.duplicated().any():
        dup_cnt = int(df.columns.duplicated().sum())
        print(f"⚠️ {label} 중복 샘플 ID {dup_cnt}개 발견 → 첫 번째만 유지")
        df = df.loc[:, ~df.columns.duplicated()]
    return df


@dataclass
class OmicsTables:
    rna: pd.DataFrame
    methy: pd.DataFrame
    protein: pd.DataFrame
    sample_ids: List[str]
    active_slots: List[str] = None  # which slots have real data

    def __post_init__(self):
        if self.active_slots is None:
            self.active_slots = ["rna", "methy", "protein"]


def _make_dummy(sample_ids: List[str], label: str = "dummy") -> pd.DataFrame:
    """1-feature dummy DataFrame (all NaN) for missing modality slot."""
    return pd.DataFrame(
        np.nan,
        index=[f"{label}_feat_0"],
        columns=sample_ids,
    )


def load_and_align_omics(
    rna_path: str = None,
    methy_path: str = None,
    protein_path: str = None,
    sample_id_mode: str = "none",
    sample_join: str = "intersection",
) -> OmicsTables:
    """
    TSV 로드 후 샘플 ID 정규화/정렬.
    최소 1개 modality path가 필요하며, 없는 modality는 더미로 채움.
    """
    provided = {}
    for slot, path in [("rna", rna_path), ("methy", methy_path), ("protein", protein_path)]:
        if path and str(path).strip():
            provided[slot] = _read_tsv(path)

    if not provided:
        raise ValueError("최소 1개 이상의 omics 파일이 필요합니다.")

    active_slots = list(provided.keys())

    for slot, df in provided.items():
        df.columns = _normalize_sample_ids(list(df.columns), sample_id_mode)
        provided[slot] = _dedupe_columns(df, slot.upper())

    sample_sets = [set(df.columns) for df in provided.values()]
    if sample_join == "union":
        sample_ids = sorted(list(set().union(*sample_sets)))
    else:
        sample_ids = sorted(list(set.intersection(*sample_sets)))

    if len(sample_ids) == 0:
        raise ValueError("공통/합집합 샘플이 없습니다. 샘플 ID를 확인해주세요.")

    for slot in provided:
        provided[slot] = provided[slot].reindex(columns=sample_ids)

    rna = provided.get("rna", _make_dummy(sample_ids, "rna"))
    methy = provided.get("methy", _make_dummy(sample_ids, "methy"))
    protein = provided.get("protein", _make_dummy(sample_ids, "protein"))

    n_real = len(active_slots)
    n_dummy = 3 - n_real
    print(f"✅ 샘플 정렬 완료: {len(sample_ids)}개 (join={sample_join}), "
          f"modalities: {n_real}개 실제 + {n_dummy}개 더미")

    return OmicsTables(rna=rna, methy=methy, protein=protein,
                       sample_ids=sample_ids, active_slots=active_slots)


class TriModalDatasetFlexible(Dataset):
    """
    RNA/Protein/Methylation 데이터를 (samples x features)로 변환하여 반환.
    - 결측치는 NaN으로 들어오며, 텐서로 변환 시 0으로 채움.
    - 마스크는 1=NA, 0=관측
    """
    def __init__(self, rna: pd.DataFrame, methy: pd.DataFrame, protein: pd.DataFrame):
        self.sample_ids = list(rna.columns)

        self.rna = rna.T.astype(np.float32)      # (samples x features)
        self.methy = methy.T.astype(np.float32)
        self.protein = protein.T.astype(np.float32)

        self.m_rna = self.rna.isna().astype(np.float32)
        self.m_methy = self.methy.isna().astype(np.float32)
        self.m_prot = self.protein.isna().astype(np.float32)

        self.rna_f = self.rna.fillna(0.0)
        self.methy_f = self.methy.fillna(0.0)
        self.protein_f = self.protein.fillna(0.0)

    def __len__(self) -> int:
        return self.rna.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "x_rna": torch.from_numpy(self.rna_f.iloc[i].to_numpy()),
            "x_methy": torch.from_numpy(self.methy_f.iloc[i].to_numpy()),
            "x_prot": torch.from_numpy(self.protein_f.iloc[i].to_numpy()),
            "m_rna": torch.from_numpy(self.m_rna.iloc[i].to_numpy()),
            "m_methy": torch.from_numpy(self.m_methy.iloc[i].to_numpy()),
            "m_prot": torch.from_numpy(self.m_prot.iloc[i].to_numpy()),
        }


def get_triple_dataloaders_flexible(
    rna: pd.DataFrame,
    methy: pd.DataFrame,
    protein: pd.DataFrame,
    batch_size: int = 32,
    split: Tuple[float, float] = (0.8, 0.2),
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    ds = TriModalDatasetFlexible(rna=rna, methy=methy, protein=protein)
    n = len(ds)
    n_tr = max(1, int(n * split[0]))
    n_va = max(1, n - n_tr)

    gen = torch.Generator().manual_seed(seed)
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_va], generator=gen)

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    valid_loader = DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False)

    data_info = {
        "src_dim_rna": rna.shape[0],
        "src_dim_prot": protein.shape[0],
        "src_dim_methy": methy.shape[0],
        "n_train": n_tr,
        "n_valid": n_va,
        "n_total": n,
    }
    return train_loader, valid_loader, data_info
