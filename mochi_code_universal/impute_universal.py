#!/usr/bin/env python3
"""
범용 Tri-joint Imputation 스크립트 (특징/샘플 자동 정렬)
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from dataloader_universal import load_and_align_omics
from models import Generator


class TriJointImputerUniversal:
    def __init__(self, checkpoint_path: str, device: str = "auto", batch_size: int = 64):
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
        self.batch_size = batch_size
        self.ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.meta = self.ckpt.get("meta", {})
        self._load_models()

    def _load_models(self):
        dims = self.meta.get("dims", {})
        dim_r = dims["rna"]
        dim_p = dims["protein"]
        dim_m = dims["methy"]

        self.Gp = Generator(input_size=dim_r + dim_m, output_size=dim_p,
                            use_attn=True, n_heads=4, d_head=64, target_type="protein")
        self.Gr = Generator(input_size=dim_p + dim_m, output_size=dim_r,
                            use_attn=True, n_heads=4, d_head=64, target_type="rna")
        self.Gm = Generator(input_size=dim_r + dim_p, output_size=dim_m,
                            use_attn=True, n_heads=4, d_head=64, target_type="methyl")

        self.Gp.load_state_dict(self.ckpt["Gp"])
        self.Gr.load_state_dict(self.ckpt["Gr"])
        self.Gm.load_state_dict(self.ckpt["Gm"])

        self.Gp.to(self.device).eval()
        self.Gr.to(self.device).eval()
        self.Gm.to(self.device).eval()

    def _prepare_tensor(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        missing_mask = torch.tensor(df.isna().values, dtype=torch.bool, device=self.device).T
        filled = df.fillna(0.0)
        tensor = torch.tensor(filled.values, dtype=torch.float32, device=self.device).T
        return tensor, missing_mask

    def _tensor_to_df(self, tensor: torch.Tensor, ref_df: pd.DataFrame) -> pd.DataFrame:
        arr = tensor.detach().cpu().numpy()
        if arr.shape == (len(ref_df.columns), len(ref_df.index)):
            arr = arr.T
        return pd.DataFrame(arr, index=ref_df.index, columns=ref_df.columns)

    def _align_features(self, df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.astype(str)
        overlap = len(set(df.index) & set(features))
        ratio = overlap / max(len(features), 1)
        if ratio < 0.5:
            print(f"⚠️ {label} 특징 겹침 비율 낮음: {overlap}/{len(features)} ({ratio:.1%})")
        aligned = df.reindex(features)
        return aligned

    def _align_samples_union(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        all_samples = set()
        for df in tables.values():
            all_samples |= set(df.columns)
        sample_ids = sorted(list(all_samples))
        return {k: v.reindex(columns=sample_ids) for k, v in tables.items()}

    def impute(self, rna: pd.DataFrame, methy: pd.DataFrame, protein: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        features = self.meta.get("features", {})
        rna = self._align_features(rna, features["rna"], "RNA")
        methy = self._align_features(methy, features["methy"], "Methylation")
        protein = self._align_features(protein, features["protein"], "Protein")

        tables = self._align_samples_union({"rna": rna, "methy": methy, "protein": protein})
        rna, methy, protein = tables["rna"], tables["methy"], tables["protein"]

        rna_tensor, rna_mask = self._prepare_tensor(rna)
        prot_tensor, prot_mask = self._prepare_tensor(protein)
        methy_tensor, methy_mask = self._prepare_tensor(methy)

        X_Gp = torch.cat([rna_tensor, methy_tensor], dim=1)
        X_Gr = torch.cat([prot_tensor, methy_tensor], dim=1)
        X_Gm = torch.cat([rna_tensor, prot_tensor], dim=1)

        def _batch_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            if self.batch_size is None:
                return model(x)
            outputs = []
            for i in range(0, x.shape[0], self.batch_size):
                outputs.append(model(x[i:i + self.batch_size]))
            return torch.cat(outputs, dim=0)

        protein_pred = _batch_forward(self.Gp, X_Gp)
        rna_pred = _batch_forward(self.Gr, X_Gr)
        methy_pred = _batch_forward(self.Gm, X_Gm)

        protein_imp = torch.where(prot_mask, protein_pred, prot_tensor)
        rna_imp = torch.where(rna_mask, rna_pred, rna_tensor)
        methy_imp = torch.where(methy_mask, methy_pred, methy_tensor)

        return {
            "rna": self._tensor_to_df(rna_imp, rna),
            "protein": self._tensor_to_df(protein_imp, protein),
            "methy": self._tensor_to_df(methy_imp, methy),
        }


def main():
    parser = argparse.ArgumentParser(description="범용 멀티오믹스 Imputation (1~3 modality)")
    parser.add_argument("--checkpoint", required=True, help="tri_best.ckpt 경로")
    parser.add_argument("--rna", default=None, help="Omics slot 1 TSV (features x samples)")
    parser.add_argument("--methy", default=None, help="Omics slot 2 TSV (features x samples)")
    parser.add_argument("--protein", default=None, help="Omics slot 3 TSV (features x samples)")
    parser.add_argument("--output_dir", required=True, help="결과 저장 디렉토리")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_id_mode", type=str, default="none", choices=["none", "tcga12", "tcga15"])

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tables = load_and_align_omics(
        rna_path=args.rna,
        methy_path=args.methy,
        protein_path=args.protein,
        sample_id_mode=args.sample_id_mode,
        sample_join="union",
    )

    imputer = TriJointImputerUniversal(args.checkpoint, device=args.device, batch_size=args.batch_size)
    results = imputer.impute(tables.rna, tables.methy, tables.protein)

    for slot in tables.active_slots:
        results[slot].to_csv(os.path.join(args.output_dir, f"{slot}_imputed.tsv"), sep="\t")

    print(f"✅ imputation 완료 ({len(tables.active_slots)}개 modality): {args.output_dir}")


if __name__ == "__main__":
    main()
