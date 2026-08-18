#!/usr/bin/env python3
"""
향상된 멀티오믹스 데이터 전처리 스크립트
제공된 코드를 기반으로 NMF+TGAN과 완벽 호환되는 데이터셋 생성
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------
# Helpers
# ----------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _pick_id_column(df: pd.DataFrame, prefer_cols: tuple = ("Sample", "sample", "SAMPLE", "barcode", "ID", "id")) -> str:
    for c in prefer_cols:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower() not in ["pam50", "subtype"]:
            return c
    return df.columns[0]


def normalize_tcga_id(s: str, keep_suffix: bool = True) -> str:
    s = str(s).split(".")[0]
    return s[:15] if keep_suffix else s[:12]


def align_columns_like(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(a.columns) & set(b.columns))
    return a[common].copy(), b[common].copy()



# ----------------------
# Subtype filtering
# ----------------------
class SubtypeFilterConfig:
    def __init__(self, subtype_table_path: str, label_col: str = "PAM50",
                 valid_labels: tuple = ("Basal", "Her2", "LumA", "LumB"),
                 id_col: Optional[str] = None, keep_suffix: bool = True,
                 apply_id_normalization: bool = False):
        self.subtype_table_path = subtype_table_path
        self.label_col = label_col
        self.valid_labels = valid_labels
        self.id_col = id_col
        self.keep_suffix = keep_suffix
        self.apply_id_normalization = apply_id_normalization


def load_and_filter_subtypes(cfg: SubtypeFilterConfig) -> pd.DataFrame:
    raw = pd.read_csv(cfg.subtype_table_path, sep="\t")
    df = pd.DataFrame(raw)
    
    # ID 컬럼 찾기 (sample 컬럼 우선)
    id_col = cfg.id_col or _pick_id_column(df)
    print(f"선택된 ID 컬럼: {id_col}")
    
    if cfg.label_col not in df.columns:
        raise ValueError(f"label_col '{cfg.label_col}' not found.")

    # 유효한 서브타입만 필터링
    df = df[df[cfg.label_col].isin(cfg.valid_labels)].copy()
    print(f"유효한 서브타입 샘플 수: {len(df)}")
    
    # ID 설정
    if cfg.apply_id_normalization:
        ids = df[id_col].apply(lambda x: normalize_tcga_id(x, keep_suffix=cfg.keep_suffix))
    else:
        ids = df[id_col].astype(str)
    
    # 중복 제거 및 인덱스 설정
    df["ID"] = ids
    df = df.drop_duplicates(subset=["ID"]).set_index("ID")
    
    print(f"최종 필터링된 샘플 수: {len(df)}")
    print(f"서브타입 분포:\n{df[cfg.label_col].value_counts()}")
    
    return df[[cfg.label_col]]


# ----------------------
# Data loading
# ----------------------
class DataPaths:
    def __init__(self, rna: str, methy: str, protein: str):
        self.rna = rna
        self.methy = methy
        self.protein = protein


class FeatureFilterConfig:
    def __init__(self, top_var_rna: Optional[int] = None, top_var_methy: Optional[int] = 10000,
                 top_var_protein: Optional[int] = None):
        self.top_var_rna = top_var_rna
        self.top_var_methy = top_var_methy
        self.top_var_protein = top_var_protein


def _variance_filter(df: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
    if k is None or k >= df.shape[0]:
        return df
    var = df.var(axis=1, skipna=True)
    keep = var.sort_values(ascending=False).head(k).index
    return df.loc[keep].copy()


def load_omics(paths: DataPaths, apply_id_normalization: bool = False, keep_suffix: bool = True,
               feature_filter: Optional[FeatureFilterConfig] = None) -> Dict[str, pd.DataFrame]:
    def _read(tsv: str) -> pd.DataFrame:
        df = pd.read_csv(tsv, sep="\t", index_col=0)
        if apply_id_normalization:
            df.columns = [normalize_tcga_id(c, keep_suffix=keep_suffix) for c in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        return df

    rna = _read(paths.rna)
    methy = _read(paths.methy)
    protein = _read(paths.protein)

    feature_filter = feature_filter or FeatureFilterConfig()
    rna = _variance_filter(rna, feature_filter.top_var_rna)
    methy = _variance_filter(methy, feature_filter.top_var_methy)
    protein = _variance_filter(protein, feature_filter.top_var_protein)

    return {"rna": rna, "methy": methy, "protein": protein}


# ----------------------
# Contamination (RNA-only) - 결측치 마스킹 전용
# ----------------------
class MissingnessConfig:
    def __init__(self, rate: float = 0.1, random_state: int = 0):
        """
        결측치 마스킹 설정
        
        Args:
            rate: 결측치 비율 (0.0 ~ 1.0)
            random_state: 랜덤 시드
        """
        self.rate = rate
        self.random_state = random_state


def inject_missing_random(df: pd.DataFrame, cfg: MissingnessConfig) -> pd.DataFrame:
    """
    무작위 결측치 마스킹 적용
    
    Args:
        df: 입력 데이터프레임
        cfg: 결측치 설정
        
    Returns:
        결측치가 마스킹된 데이터프레임
    """
    rng = np.random.default_rng(cfg.random_state)
    X = df.copy()
    
    # 전체 요소 중 지정된 비율만큼 무작위로 선택하여 마스킹
    total_elements = X.size
    num_missing = int(total_elements * cfg.rate)
    
    # 무작위 위치 선택 (중복 없이)
    flat_indices = rng.choice(total_elements, num_missing, replace=False)
    row_indices, col_indices = np.unravel_index(flat_indices, X.shape)
    
    # 선택된 위치를 NaN으로 마스킹
    for i in range(len(row_indices)):
        X.iloc[row_indices[i], col_indices[i]] = np.nan
    
    print(f"  결측치 마스킹 완료: {num_missing}/{total_elements} ({cfg.rate:.1%})")
    return X


def contaminate_rna(rna_df: pd.DataFrame, missing_cfg: Optional[MissingnessConfig] = None) -> pd.DataFrame:
    """
    RNA 데이터에 결측치 마스킹만 적용
    
    Args:
        rna_df: RNA 데이터프레임
        missing_cfg: 결측치 설정
        
    Returns:
        결측치가 마스킹된 RNA 데이터프레임
    """
    X = rna_df.copy()
    
    if missing_cfg is not None:
        X = inject_missing_random(X, missing_cfg)
    
    return X


# ----------------------
# Dataset variants
# ----------------------
class PrepConfig:
    def __init__(self, out_dir: str, out_prefix: str = "BRCA_PAM50",
                 save_omicsnmf_pairs: bool = True, first_col_name: str = "features"):
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.save_omicsnmf_pairs = save_omicsnmf_pairs
        self.first_col_name = first_col_name


def compute_common_ids(omics: Dict[str, pd.DataFrame], how: str = "union") -> List[str]:
    sets = [set(df.columns) for df in omics.values()]
    ids = set.intersection(*sets) if how == "intersection" else set.union(*sets)
    return sorted(ids)


def subset_to_ids(omics: Dict[str, pd.DataFrame], ids: List[str]) -> Dict[str, pd.DataFrame]:
    return {k: df[[i for i in ids if i in df.columns]].copy() for k, df in omics.items()}


def drop_samples_with_any_na(omics: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    all_ids = compute_common_ids(omics, how="intersection")
    keep_ids = []
    for sid in all_ids:
        if all(not df[sid].isna().any() for df in omics.values()):
            keep_ids.append(sid)
    return subset_to_ids(omics, keep_ids), keep_ids


def save_variant(omics: Dict[str, pd.DataFrame], cfg: PrepConfig, tag: str):
    _ensure_dir(cfg.out_dir)
    
    # 각 오믹스 데이터 저장
    for om, df in omics.items():
        if df.shape[0] > 0 and df.shape[1] > 0:
            out_tsv = os.path.join(cfg.out_dir, f"{cfg.out_prefix}.{om}.{tag}.tsv")
            df.to_csv(out_tsv, sep="\t")
            print(f"  저장됨: {om}.{tag}.tsv ({df.shape})")
        else:
            print(f"  ⚠️ {om}.{tag} 데이터프레임이 비어있어 저장하지 않음: {df.shape}")
    
    # OmicsNMF 쌍 저장
    if cfg.save_omicsnmf_pairs:
        pairs = [("rna", "methy"), ("rna", "protein"), ("methy", "rna"), ("protein", "rna"), ("methy", "protein"), ("protein", "methy")]
        for s, t in pairs:
            if s in omics and t in omics:
                try:
                    s_df, t_df = align_columns_like(omics[s], omics[t])
                    if s_df.shape[1] == 0:
                        print(f"  ⚠️ {s}_to_{t} 공통 샘플이 없어 저장하지 않음")
                        continue
                    
                    out_s = os.path.join(cfg.out_dir, f"{cfg.out_prefix}.{s}_to_{t}.{tag}.source.tsv")
                    out_t = os.path.join(cfg.out_dir, f"{cfg.out_prefix}.{s}_to_{t}.{tag}.target.tsv")
                    
                    # 안전한 저장 (TSV 형식)
                    if s_df.shape[0] > 0 and s_df.shape[1] > 0:
                        s_df.to_csv(out_s, sep='\t')
                        print(f"  저장됨: {s}_to_{t}.{tag}.source.tsv ({s_df.shape})")
                    
                    if t_df.shape[0] > 0 and t_df.shape[1] > 0:
                        t_df.to_csv(out_t, sep='\t')
                        print(f"  저장됨: {s}_to_{t}.{tag}.target.tsv ({t_df.shape})")
                        
                except Exception as e:
                    print(f"  ❌ {s}_to_{t} 저장 중 오류: {e}")
                    continue


def prepare_all_variants(data_paths: DataPaths,
                         subtype_cfg: SubtypeFilterConfig,
                         prep_cfg: PrepConfig,
                         feature_filter: Optional[FeatureFilterConfig] = None,
                         missing_cfg: Optional[MissingnessConfig] = MissingnessConfig()) -> Dict[str, Dict[str, pd.DataFrame]]:
    print("🚀 멀티오믹스 데이터 전처리 시작")
    print("=" * 60)
    
    # 1. 오믹스 데이터 로드
    print("📊 오믹스 데이터 로딩 중...")
    omics = load_omics(data_paths,
                       apply_id_normalization=subtype_cfg.apply_id_normalization,
                       keep_suffix=subtype_cfg.keep_suffix,
                       feature_filter=feature_filter)
    
    for omics_type, df in omics.items():
        print(f"  {omics_type.upper()}: {df.shape}")
    
    # 2. 서브타입 필터링
    print("\n🏷️ PAM50 서브타입 필터링 중...")
    sub_df = load_and_filter_subtypes(subtype_cfg)
    
    # PAM50 샘플 ID와 RNA 컬럼명 매칭
    pam50_sample_ids = set(sub_df.index)
    rna_sample_ids = set(omics["rna"].columns)
    
    print(f"  PAM50 샘플 ID 수: {len(pam50_sample_ids)}")
    print(f"  RNA 샘플 ID 수: {len(rna_sample_ids)}")
    
    # 공통 샘플 찾기
    common_ids = pam50_sample_ids & rna_sample_ids
    print(f"  공통 샘플 수: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("❌ 경고: PAM50과 RNA 간에 공통 샘플이 없습니다!")
        print("PAM50 샘플 ID 예시:", list(pam50_sample_ids)[:5])
        print("RNA 샘플 ID 예시:", list(rna_sample_ids)[:5])
        return {}
    
    keep_ids = sorted(list(common_ids))
    print(f"  최종 사용할 샘플: {len(keep_ids)}개")
    print(f"  서브타입 분포:")
    print(sub_df.loc[keep_ids, 'PAM50'].value_counts().sort_index())
    
    omics = subset_to_ids(omics, keep_ids)
    
    # 3. 데이터셋 변형 생성
    variants: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    # Original: 원본 데이터
    print("\n📋 Original 데이터셋 생성 중...")
    variants["original"] = {k: df.copy() for k, df in omics.items()}
    save_variant(variants["original"], prep_cfg, tag="original")
    
    # Noisy: RNA에 오염 추가
    print("\n🔊 Noisy 데이터셋 생성 중...")
    rna_noisy = contaminate_rna(omics["rna"], missing_cfg=missing_cfg)
    variants["noisy"] = {"rna": rna_noisy, "methy": omics["methy"].copy(), "protein": omics["protein"].copy()}
    save_variant(variants["noisy"], prep_cfg, tag="noisy")
    
    # Complete: 결측치 완전 제거 (선택적)
    print("\n🧹 Complete 데이터셋 생성 중...")
    complete_omics, keep_ids_complete = drop_samples_with_any_na(variants["original"])
    
    if len(keep_ids_complete) > 0:
        variants["complete"] = complete_omics
        save_variant(variants["complete"], prep_cfg, tag="complete")
        print(f"  Complete 데이터셋 생성 완료: {len(keep_ids_complete)}개 샘플")
    else:
        print("  ⚠️ Complete 데이터셋 생성 불가: 모든 샘플에 결측치가 존재")
        print("  Complete 데이터셋을 건너뜁니다.")
        # variants에서 complete 제거하지 않음 (나중에 처리)
    
    # 4. 요약 정보 생성
    print("\n📊 데이터셋 요약 정보 생성 중...")
    meta_rows = []
    
    # 실제로 생성된 데이터셋만 처리
    for tag, d in variants.items():
        if tag == "complete" and len(d) == 0:
            print(f"  ⚠️ {tag} 데이터셋은 생성되지 않았습니다.")
            continue
            
        for k, df in d.items():
            # 데이터프레임 크기 확인 및 안전한 NA 비율 계산
            if df.shape[0] > 0 and df.shape[1] > 0:
                na_rate = float(df.isna().sum().sum()) / (df.shape[0] * df.shape[1])
            else:
                na_rate = 0.0
                print(f"⚠️ 경고: {tag}.{k} 데이터프레임이 비어있습니다. 크기: {df.shape}")
            
            meta_rows.append({
                "variant": tag, 
                "modality": k, 
                "n_features": df.shape[0], 
                "n_samples": df.shape[1], 
                "NA_rate": na_rate
            })
    
    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(os.path.join(prep_cfg.out_dir, f"{prep_cfg.out_prefix}.summary.tsv"), sep="\t", index=False)
        print(f"  요약 정보 저장 완료: {len(meta_rows)}개 데이터셋")
    else:
        print("  ❌ 저장할 데이터셋이 없습니다.")
    
    print("\n🎉 모든 데이터셋 생성 완료!")
    print(f"저장 위치: {prep_cfg.out_dir}")
    
    return variants


# ------------- CLI -------------
def _as_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


def main():
    parser = argparse.ArgumentParser(description="향상된 BRCA 멀티오믹스 데이터 전처리 (NMF+TGAN 호환)")
    parser.add_argument("--rna", required=True, help="RNA 데이터 파일 경로")
    parser.add_argument("--methy", required=True, help="Methylation 데이터 파일 경로")
    parser.add_argument("--protein", required=True, help="Protein 데이터 파일 경로")
    parser.add_argument("--subtype", required=True, help="PAM50 서브타입 파일 경로")
    parser.add_argument("--out_dir", required=True, help="출력 디렉토리")
    parser.add_argument("--prefix", default="BRCA_PAM50", help="출력 파일 접두사")
    parser.add_argument("--keep_suffix", default="true", help="TCGA ID 접미사 유지 여부")
    parser.add_argument("--apply_id_normalization", default="false", help="TCGA ID 정규화 적용 여부")
    parser.add_argument("--var_methy", type=int, default=10000, help="Methylation 특징 수 (분산 기준)")
    parser.add_argument("--save_pairs", default="true", help="source-target 쌍 저장 여부")
    parser.add_argument("--first_col_name", default="features", help="첫 번째 열 이름")
    
    # 결측치 마스킹 설정
    parser.add_argument("--miss_rate", type=float, default=0.10, help="결측치 마스킹 비율 (0.0 ~ 1.0)")

    args = parser.parse_args()

    # 설정 객체 생성
    dpaths = DataPaths(rna=args.rna, methy=args.methy, protein=args.protein)
    subtype_cfg = SubtypeFilterConfig(
        subtype_table_path=args.subtype,
        label_col="PAM50",
        valid_labels=("Basal", "Her2", "LumA", "LumB"),
        id_col=None,
        keep_suffix=_as_bool(args.keep_suffix),
        apply_id_normalization=_as_bool(args.apply_id_normalization),
    )
    prep_cfg = PrepConfig(
        out_dir=args.out_dir, 
        out_prefix=args.prefix, 
        save_omicsnmf_pairs=_as_bool(args.save_pairs),
        first_col_name=args.first_col_name
    )
    feat_cfg = FeatureFilterConfig(top_var_methy=args.var_methy)

    # 결측치 마스킹 설정
    missing_cfg = MissingnessConfig(rate=args.miss_rate, random_state=0)

    # 데이터셋 생성
    variants = prepare_all_variants(
        data_paths=dpaths,
        subtype_cfg=subtype_cfg,
        prep_cfg=prep_cfg,
        feature_filter=feat_cfg,
        missing_cfg=missing_cfg,
    )

    print(f"\n✅ 모든 데이터셋이 {args.out_dir}에 저장되었습니다!")


if __name__ == "__main__":
    main()


