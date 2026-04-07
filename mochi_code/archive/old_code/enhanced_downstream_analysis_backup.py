#!/usr/bin/env python3
"""
향상된 다운스트림 분석 스크립트
- Imputation 성능 평가
- 생물학적 의미 분석
- 임상적 관련성 분석
- 데이터 품질 심화 분석
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
# 바꾼 후
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test  # ← 여기로!

# multivariate_logrank_test 안전 임포트
try:
    from lifelines.statistics import multivariate_logrank_test
    MULTIVARIATE_LOGRANK_AVAILABLE = True
except ImportError:
    multivariate_logrank_test = None
    MULTIVARIATE_LOGRANK_AVAILABLE = False
    print("⚠️ multivariate_logrank_test를 사용할 수 없습니다. 전체 그룹 비교는 생략됩니다.")
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 및 시각화 스타일
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# lifelines 패키지 사용 가능 여부 확인
try:
    import lifelines
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("⚠️ lifelines 패키지가 설치되지 않았습니다. 생존분석 기능이 제한됩니다.")

# 로컬 모듈 임포트
try:
    from utils import calculate_imputation_metrics
except ImportError:
    print("⚠️ utils 모듈을 찾을 수 없습니다. 일부 기능이 제한됩니다.")
    def calculate_imputation_metrics(*args, **kwargs):
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan}

# 생존분석 관련 추가 임포트
# StratifiedKFold는 사용되지 않으므로 제거

# TCGA 샘플 ID 인식 및 안정화 헬퍼
import re

_TCGA_RE = re.compile(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', re.I)

def _tcga_hits(index_like):
    s = pd.Index(index_like).astype(str)
    return int(s.str.contains(_TCGA_RE).sum())

def _ensure_samples_by_rows(df: pd.DataFrame) -> pd.DataFrame:
    """행=샘플, 열=특징 형태 보장"""
    idx_hits = _tcga_hits(df.index)
    col_hits = _tcga_hits(df.columns)
    if col_hits > idx_hits:
        df = df.T
    df.index = df.index.astype(str)
    return df

# 감독형 PCA + Cox 튜닝 헬퍼
from sklearn.model_selection import KFold

def _safe_import_lifelines():
    """lifelines 안전 임포트"""
    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test
        from lifelines.utils import concordance_index
        return KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index
    except ImportError:
        raise RuntimeError("lifelines가 필요합니다. `pip install lifelines` 후 다시 실행하세요.")

def supervised_pca(X: pd.DataFrame, y: pd.DataFrame, max_k=50, var_thr=0.90):
    """표준화→PCA→각 PC에 대해 단변량 Cox p-value 평가→상위 PC만 선택 (성능 향상 중심)"""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    
    # PC 개수를 더 공격적으로 설정 (표본 수 기반)
    k0 = min(max_k, max(5, min(Xs.shape[1]-1, Xs.shape[0]//3)))  # 표본/4 → 3으로 변경
    pca = PCA(n_components=k0, random_state=42)
    Z = pca.fit_transform(Xs)
    Zdf = pd.DataFrame(Z, index=X.index, columns=[f"PC{i+1}" for i in range(Z.shape[1])])

    # 각 PC 단변량 Cox p-value
    KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
    
    pvals = []
    for c in Zdf.columns:
        df1 = Zdf[[c]].join(y[["OS.time","OS"]])
        cph1 = CoxPHFitter(penalizer=0.05)  # 0.0 → 0.05로 변경
        try:
            cph1.fit(df1, duration_col="OS.time", event_col="OS")
            pvals.append((c, float(cph1.summary.loc[c, "p"])))
        except Exception:
            pvals.append((c, 1.0))
    pvals.sort(key=lambda x: x[1])  # p 작은 PC 먼저

    # 더 공격적인 PC 선택 (분산 85% + 표본/4 제한)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k_var = int(np.argmax(cum >= 0.85) + 1)  # 80% → 85%로 증가
    k_cap = max(5, min(len(pvals), X.shape[0]//4))  # 표본/5 → 4로 변경, 최소 3 → 5
    k_final = max(5, min(k_var, k_cap, 15))  # 최대 8 → 15개 PC로 증가
    top_cols = [c for c, _ in pvals[:k_final]]

    return Zdf[top_cols], sc, pca, top_cols

def tune_cox(df: pd.DataFrame, grid_pen=(0.05,0.1,0.15,0.2,0.25), grid_l1=(0.3,0.5,0.7), n_splits=5):
    """KFold로 C-index 최대화하는 페널티 튜닝 (성능 향상 중심)"""
    KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
    
    n_max = max(2, min(n_splits, len(df)-1, (len(df)//6 if len(df)>=30 else 3)))
    kf = KFold(n_splits=n_max, shuffle=True, random_state=42)
    best_c, best_hyp = -1.0, (0.1, 0.5)  # 기본값을 더 공격적으로
    
    for pen in grid_pen:
        for l1 in grid_l1:
            cs=[]
            for tr, va in kf.split(df):
                tr_df, va_df = df.iloc[tr], df.iloc[va]
                cph = CoxPHFitter(penalizer=pen, l1_ratio=l1)
                try:
                    cph.fit(tr_df, duration_col="OS.time", event_col="OS")
                    risk = cph.predict_partial_hazard(va_df).values.ravel()
                    c = concordance_index(va_df["OS.time"], risk, va_df["OS"])
                    cs.append(c)
                except Exception:
                    cs.append(0.5)
            m = float(np.mean(cs))
            if m > best_c:
                best_c, best_hyp = m, (pen, l1)
    
    return best_hyp

class EnhancedDownstreamAnalysis:
    """
    향상된 다운스트림 분석 클래스
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.output_dir = None
        
        # 키 별칭 매핑
        self._PLOT_KEY_ALIASES = {
            "figure_size": "figsize", "figsize": "figsize",
            "title_font_size": "title_fontsize", "title_fontsize": "title_fontsize",
            "label_font_size": "label_fontsize", "label_fontsize": "label_fontsize",
            "legend_font_size": "legend_fontsize", "legend_fontsize": "legend_fontsize",
        }
        
        # 그래프 설정 (사용자가 커스터마이징 가능) - 키 통일
        self.plot_config = {
            'figsize': (12, 8),           # figure_size → figsize로 통일
            'dpi': 300,
            'linewidth': 4,
            'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'],
            'title_fontsize': 16,         # title_font_size → title_fontsize
            'label_fontsize': 14,         # font_size → label_fontsize
            'legend_fontsize': 12,        # legend_font_size → legend_fontsize
            'grid_alpha': 0.3,
            'color_palette': 'Set2',
            'style': 'seaborn-v0_8',
            'save_format': 'png',
            'facecolor': 'white',
            'bbox_inches': 'tight'
        }
        
        # 사용자 설정이 있으면 업데이트
        if 'plot_config' in config:
            self._apply_plot_config(config['plot_config'])
        
        # matplotlib 설정 적용
        self._apply_plot_config({})
    
    def _apply_plot_config(self, cfg: dict):
        """그래프 설정 전역 적용 + 키 정규화"""
        if not cfg:
            cfg = {}
        
        # 1) 키 정규화
        norm = {}
        for k, v in cfg.items():
            norm[self._PLOT_KEY_ALIASES.get(k, k)] = v
        
        # 2) 병합
        self.plot_config.update(norm)
        
        # 3) rcParams/스타일 적용
        style = self.plot_config.get("style", "seaborn-v0_8")
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8")
        
        dpi = self.plot_config.get("dpi", 300)
        plt.rcParams["figure.dpi"] = dpi
        plt.rcParams["savefig.dpi"] = dpi
        
        # 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_data(self, data_paths):
        """데이터 로드"""
        print("📊 데이터 로딩 중...")
        
        # Imputed 데이터
        self.train_imputed = pd.read_csv(data_paths['train_imputed'], sep='\t', index_col=0)
        self.valid_imputed = pd.read_csv(data_paths['valid_imputed'], sep='\t', index_col=0)
        
        # 원본 데이터
        self.original_data = pd.read_csv(data_paths['original'], sep='\t', index_col=0)
        
        # 마스크
        self.mask_data = pd.read_csv(data_paths['mask'], sep='\t', index_col=0)
        
        # 서브타입 정보 (있는 경우)
        self.subtype_data = None
        if 'subtype' in data_paths and os.path.exists(data_paths['subtype']):
            self.subtype_data = pd.read_csv(data_paths['subtype'], sep='\t', index_col=0)
        
        # 생존 데이터 로드
        self.survival_data = None
        if 'survival' in data_paths and os.path.exists(data_paths['survival']):
            self.survival_data = pd.read_csv(data_paths['survival'], sep='\t', index_col=0)
            print(f"✅ 생존 데이터 로드: {self.survival_data.shape}")
        else:
            print("⚠️ 생존 데이터가 없습니다.")
        
        print(f"✅ 데이터 로드 완료")
        print(f"훈련 데이터: {self.train_imputed.shape}")
        print(f"검증 데이터: {self.valid_imputed.shape}")
        print(f"원본 데이터: {self.original_data.shape}")
        
        # 공통 샘플 찾기
        self.common_samples = self.train_imputed.index.intersection(
            self.valid_imputed.index
        ).intersection(self.original_data.index)
        
        # 데이터 정렬
        self.align_data()
        
    def align_data(self):
        """데이터 정렬"""
        self.train_aligned = self.train_imputed.loc[self.common_samples]
        self.valid_aligned = self.valid_imputed.loc[self.common_samples]
        self.original_aligned = self.original_data.loc[self.common_samples]
        self.mask_aligned = self.mask_data.loc[self.common_samples]
        
        if self.subtype_data is not None:
            self.subtype_aligned = self.subtype_data.loc[
                self.subtype_data.index.intersection(self.common_samples)
            ]
        
        # 생존 데이터 정렬
        if self.survival_data is not None:
            # TCGA 샘플 ID를 표준화 (01A, 11B 등 접미사 제거)
            survival_samples = [s.split('-01A')[0] if '-01A' in s else s.split('-11B')[0] if '-11B' in s else s 
                              for s in self.survival_data.index]
            self.survival_data.index = survival_samples
            
            # ⚠️ 생존 시간 단위 확인: TCGA는 보통 일(day) 단위
            # 월 단위로 사용하려면: self.survival_data['OS.time'] = self.survival_data['OS.time'] / 30.44
            print(f"생존 시간 범위: {self.survival_data['OS.time'].min():.1f} ~ {self.survival_data['OS.time'].max():.1f}")
            if self.survival_data['OS.time'].max() > 1000:  # 1000일 이상이면 일 단위일 가능성
                print("⚠️ 생존 시간이 1000 이상입니다. 일 단위일 수 있습니다.")
                print("월 단위로 변환하려면: OS.time / 30.44")
            
            # 공통 샘플과 매칭
            self.survival_aligned = self.survival_data.loc[
                self.survival_data.index.intersection(self.common_samples)
            ]
            print(f"✅ 생존 데이터 정렬 완료: {self.survival_aligned.shape}")
        else:
            self.survival_aligned = None
        
        # 메타데이터 정렬 (새로 추가)
        if hasattr(self, 'metadata') and self.metadata is not None:
            self.restrict_to_imputed_samples(min_samples_per_subtype=2, strict_filtering=False)
    
    def basic_imputation_analysis(self):
        """1. 기본 Imputation 성능 분석"""
        print("\n🔍 1단계: 기본 Imputation 성능 분석")
        
        # 마스크를 boolean으로 변환
        mask_bool = self.mask_aligned.astype(bool)
        
        # 훈련 데이터 성능
        train_metrics = calculate_imputation_metrics(
            self.original_aligned.values, 
            self.train_aligned.values, 
            self.mask_aligned.values
        )
        
        # 검증 데이터 성능
        valid_metrics = calculate_imputation_metrics(
            self.original_aligned.values, 
            self.valid_aligned.values, 
            self.mask_aligned.values
        )
        
        # 결측치 비율
        missing_ratio = mask_bool.sum().sum() / mask_bool.size
        
        self.results['basic_imputation'] = {
            'train_performance': train_metrics,
            'valid_performance': valid_metrics,
            'missing_ratio': float(missing_ratio),
            'n_samples': len(self.common_samples),
            'n_features': self.train_aligned.shape[1]
        }
        
        print(f"훈련 데이터 RMSE: {train_metrics['rmse']:.4f}")
        print(f"검증 데이터 RMSE: {valid_metrics['rmse']:.4f}")
        print(f"결측치 비율: {missing_ratio:.2%}")
        
        return train_metrics, valid_metrics
    
    def load_metadata(self, subtype_path: str, survival_path: str, metadata_tsv: str = None):
        """PAM50+생존 메타데이터 로드"""
        import re
        
        def patient_id(x):
            if pd.isna(x): 
                return x
            m = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(x))
            return m.group(1) if m else str(x)[:12]

        if metadata_tsv and os.path.exists(metadata_tsv):
            meta = pd.read_csv(metadata_tsv, sep="\t", index_col=0)
            print(f"✅ 기존 메타데이터 로드: {meta.shape}")
        else:
            print("🔧 메타데이터 생성 중...")
            
            # PAM50 데이터
            pam = pd.read_csv(subtype_path, sep="\t")
            sid = [c for c in pam.columns if c.lower() in ["sample", "id", "barcode"]]
            sid = sid[0] if sid else pam.columns[0]
            sub = [c for c in pam.columns if c.lower() in ["pam50", "subtype"]]
            sub = sub[0] if sub else pam.columns[-1]
            
            pam = pam[[sid, sub]].rename(columns={sid: "sample", sub: "PAM50"})
            pam["patient"] = pam["sample"].apply(patient_id)

            # 생존 데이터
            surv = pd.read_csv(survival_path, sep="\t", index_col=0).reset_index().rename(columns={"index": "sample"})
            surv["patient"] = surv["sample"].apply(patient_id)
            
            rename_map = {}
            for c in surv.columns:
                lc = c.lower()
                if lc in ["os_time", "os.time", "overall_survival_time", "time"]: 
                    rename_map[c] = "OS.time"
                if lc in ["os", "status", "event"]: 
                    rename_map[c] = "OS"
            
            surv = surv.rename(columns=rename_map)
            if surv["OS.time"].max() > 1000: 
                surv["OS.time"] = surv["OS.time"] / 30.44
            surv["OS"] = surv["OS"].map(lambda x: 1 if str(x) in ["1", "True", "TRUE", "DECEASED"] else 0)

            # 병합
            meta = pd.merge(pam[["patient", "PAM50"]], surv[["patient", "OS.time", "OS"]], on="patient", how="inner")
            meta = meta.drop_duplicates(subset=["patient"]).set_index("patient")

        self.metadata = meta
        print(f"✅ 메타데이터 준비 완료: {self.metadata.shape}")
        return self.metadata

    def restrict_to_imputed_samples(self, min_samples_per_subtype=2, strict_filtering=False):
        """imputation 결과의 샘플 ID를 patient-level로 매핑 후 metadata 필터링 (PAM50 4개 하위유형 보존)"""
        import re
        
        def to_patient_index(idx):
            return idx.str.slice(0, 12) if isinstance(idx, pd.Index) else pd.Index([str(x)[:12] for x in idx])

        print("🔧 샘플 ID를 patient-level로 변환 중... (PAM50 4개 하위유형 보존)")
        
        # 공통 샘플 (원래 로직)
        self.common_samples = self.train_imputed.index.intersection(self.valid_imputed.index).intersection(self.original_data.index)

        # patient-level 인덱스 생성
        train_pat = to_patient_index(self.train_imputed.index)
        valid_pat = to_patient_index(self.valid_imputed.index)
        orig_pat = to_patient_index(self.original_data.index)
        common_pat = train_pat.intersection(valid_pat).intersection(orig_pat)

        # 메타데이터 교집합으로 최종 제한
        keep_pat = common_pat.intersection(self.metadata.index)
        print(f"   공통 patient 수: {len(keep_pat)}")

        # PAM50 하위유형 보존을 위한 샘플 선택 전략
        if hasattr(self, 'metadata') and 'PAM50' in self.metadata.columns:
            # 현재 선택된 샘플들의 PAM50 분포 확인
            current_pam50 = self.metadata.loc[keep_pat]['PAM50'].value_counts()
            print(f"🔍 현재 선택된 PAM50 분포: {dict(current_pam50)}")
            
            # PAM50 4개 하위유형 확인
            expected_subtypes = ['LumA', 'LumB', 'Her2', 'Basal']
            missing_subtypes = [st for st in expected_subtypes if st not in current_pam50.index]
            
            if missing_subtypes:
                print(f"⚠️ 누락된 하위유형: {missing_subtypes}")
                print("🔧 누락된 하위유형을 포함하는 샘플 추가 시도...")
                
                # 누락된 하위유형을 포함하는 샘플 찾기
                for missing_st in missing_subtypes:
                    missing_samples = self.metadata[self.metadata['PAM50'] == missing_st]
                    if len(missing_samples) > 0:
                        # 가장 가까운 샘플 추가 (patient-level로)
                        for idx in missing_samples.index:
                            if idx in common_pat:  # imputation된 샘플 중에서만
                                keep_pat = keep_pat.union([idx])
                                print(f"✅ {missing_st} 하위유형 샘플 추가: {idx}")
                                break

        # 원래 행 인덱스로 되돌릴 매핑
        self.train_aligned = self.train_imputed[train_pat.isin(keep_pat)].copy()
        self.valid_aligned = self.valid_imputed[valid_pat.isin(keep_pat)].copy()
        self.original_aligned = self.original_data[orig_pat.isin(keep_pat)].copy()
        
        if hasattr(self, 'mask_data'):
            self.mask_aligned = self.mask_data.loc[self.train_aligned.index.intersection(self.mask_data.index)].copy()

        # patient index를 메타데이터와 완전 일치시키기 위해 patient-level 인덱스 추가
        self.train_aligned["patient"] = to_patient_index(self.train_aligned.index)
        self.valid_aligned["patient"] = to_patient_index(self.valid_aligned.index)
        self.original_aligned["patient"] = to_patient_index(self.original_aligned.index)

        # 하나의 patient에 여러 샘플이 있으면 첫 샘플만 대표로 사용 (생존은 환자단위라서)
        self.train_aligned = self.train_aligned.drop_duplicates("patient").set_index("patient").drop(columns=["patient"])
        self.valid_aligned = self.valid_aligned.drop_duplicates("patient").set_index("patient").drop(columns=["patient"])
        self.original_aligned = self.original_aligned.drop_duplicates("patient").set_index("patient").drop(columns=["patient"])

        # 서브타입/생존 정렬
        self.survival_aligned = self.metadata.loc[self.metadata.index.intersection(self.train_aligned.index)][["OS.time", "OS"]]
        self.pam50_aligned = self.metadata.loc[self.survival_aligned.index][["PAM50"]]
        
        # 최종 PAM50 분포 확인
        final_pam50 = self.pam50_aligned['PAM50'].value_counts()
        print(f"🎯 최종 PAM50 분포: {dict(final_pam50)}")
        
        print("✅ 메타데이터-표현형 정렬 완료:")
        print(f"   Train: {self.train_aligned.shape}")
        print(f"   Valid: {self.valid_aligned.shape}")
        print(f"   Original: {self.original_aligned.shape}")
        print(f"   Survival: {self.survival_aligned.shape}")
        print(f"   PAM50: {self.pam50_aligned.shape}")

    def _plot_pam50_survival_analysis(self, output_dir):
        """PAM50 서브타입별 생존분석"""
        try:
            if not hasattr(self, 'pam50_aligned') or self.pam50_aligned is None:
                print("⚠️ PAM50 메타데이터가 없어 생략")
                return
                
            df = self.survival_aligned.join(self.pam50_aligned)
            print(f"🔍 PAM50 서브타입별 생존분석: {df.shape}")
            
            # PAM50 그룹별 Kaplan-Meier
            KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
            import matplotlib.pyplot as plt
            
            # 그래프 설정 적용 (키 통일)
            plt.style.use(self.plot_config['style'])
            kmf = KaplanMeierFitter()
            plt.figure(figsize=self.plot_config['figsize'])
            
            # PAM50 4개 하위유형 모두 포함하도록 보장
            expected_subtypes = ['LumA', 'LumB', 'Her2', 'Basal']
            available_subtypes = df["PAM50"].unique()
            
            print(f"🔍 사용 가능한 PAM50 하위유형: {list(available_subtypes)}")
            print(f"🎯 기대하는 PAM50 하위유형: {expected_subtypes}")
            
            for g, sub in df.groupby("PAM50"):
                # 최소 샘플 수 요구사항 완화 (2개 이상)
                if len(sub) < 2:
                    print(f"⚠️ {g}: 샘플 수 부족 ({len(sub)}개) - 생략")
                    continue
                kmf.fit(sub["OS.time"], event_observed=sub["OS"], label=f"{g} (n={len(sub)})")
                kmf.plot(ci_show=False)
                
            plt.title("PAM50 Subtype별 생존곡선", fontsize=self.plot_config['title_fontsize'])
            plt.xlabel("개월", fontsize=self.plot_config['label_fontsize'])
            plt.ylabel("생존확률", fontsize=self.plot_config['label_fontsize'])
            plt.legend(fontsize=self.plot_config['legend_fontsize'])
            plt.grid(True, alpha=self.plot_config['grid_alpha'])
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"km_by_PAM50.{self.plot_config['save_format']}")
            plt.savefig(output_path, dpi=self.plot_config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"✅ PAM50 서브타입 생존곡선 저장: {output_path}")
            
            # 로그랭크 검정 (이미 임포트됨)
            
            if len(df["PAM50"].unique()) > 1:
                groups = df["PAM50"].unique()
                if len(groups) >= 2:
                    group1 = df[df["PAM50"] == groups[0]]
                    group2 = df[df["PAM50"] == groups[1]]
                    
                    result = logrank_test(group1["OS.time"], group2["OS.time"], 
                                        group1["OS"], group2["OS"])
                    print(f"📊 로그랭크 검정 결과: p-value = {result.p_value:.4f}")
                    
        except Exception as e:
            print(f"⚠️ 서브타입 생존 분석 오류: {e}")
            import traceback
            traceback.print_exc()

    def build_multiomics_matrix(self, paths: dict, n_comp_per_omics=50, zscore=True):
        """멀티오믹스 통합 행렬 생성"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        def load_mat(p):
            return pd.read_csv(p, sep="\t", index_col=0)

        print("🔧 멀티오믹스 통합 행렬 생성 중...")
        
        blocks_train, blocks_valid, blocks_orig = [], [], []
        
        for omics, dd in paths.items():
            print(f"   {omics} 모달리티 처리 중...")
            
            Xt = load_mat(dd["train"])
            Xv = load_mat(dd["valid"])
            Xo = load_mat(dd["orig"])
            
            # 공통 교집합
            inter = Xt.index.intersection(Xv.index).intersection(Xo.index)
            Xt, Xv, Xo = Xt.loc[inter], Xv.loc[inter], Xo.loc[inter]
            
            print(f"     공통 샘플 수: {len(inter)}")

            if zscore:
                sc = StandardScaler(with_mean=True, with_std=True)
                Xt = pd.DataFrame(sc.fit_transform(Xt), index=Xt.index, columns=Xt.columns)
                Xv = pd.DataFrame(sc.transform(Xv), index=Xv.index, columns=Xv.columns)
                Xo = pd.DataFrame(sc.transform(Xo), index=Xo.index, columns=Xo.columns)
                print(f"     Z-score 표준화 완료")

            if n_comp_per_omics:
                k = min(n_comp_per_omics, Xt.shape[1]-1) if Xt.shape[1] > 1 else 1
                pca = PCA(n_components=k, random_state=42)
                
                Xt = pd.DataFrame(pca.fit_transform(Xt), index=Xt.index, 
                                columns=[f"{omics}_PC{i+1}" for i in range(k)])
                Xv = pd.DataFrame(pca.transform(Xv), index=Xv.index, 
                                columns=[f"{omics}_PC{i+1}" for i in range(k)])
                Xo = pd.DataFrame(pca.transform(Xo), index=Xo.index, 
                                columns=[f"{omics}_PC{i+1}" for i in range(k)])

                print(f"     PCA 차원축소: {Xt.shape[1]} → {k}")

            blocks_train.append(Xt)
            blocks_valid.append(Xv)
            blocks_orig.append(Xo)

        # 통합
        self.X_multi_train = pd.concat(blocks_train, axis=1)
        self.X_multi_valid = pd.concat(blocks_valid, axis=1)
        self.X_multi_orig = pd.concat(blocks_orig, axis=1)
        
        # 메타데이터/생존과 교집합
        if hasattr(self, 'survival_aligned'):
            keep = self.survival_aligned.index.intersection(self.X_multi_train.index).intersection(self.X_multi_valid.index)
            self.X_multi_train = self.X_multi_train.loc[keep]
            self.X_multi_valid = self.X_multi_valid.loc[keep]
            self.X_multi_orig = self.X_multi_orig.loc[keep]
        
        print(f"✅ 멀티오믹스 통합 행렬 준비 완료: {self.X_multi_train.shape}")
        return self.X_multi_train

    def cox_eval(self, X: pd.DataFrame, label="multiomics"):
        """Cox 비례위험 모델 평가"""
        try:
            from lifelines import CoxPHFitter
            
            if not hasattr(self, 'survival_aligned'):
                print("⚠️ 생존 데이터가 없어 Cox 분석을 건너뜁니다")
                return None, None
                
            df = X.join(self.survival_aligned)  # columns: features..., OS.time, OS
            print(f"🔍 Cox 분석: {df.shape}, 라벨: {label}")
            
            # 페널티 조금 주는 걸 권장 (고차원)
            cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)  # ridge/lasso 혼합
            
            # 전체 적합 후 Concordance index
            cph.fit(df, duration_col="OS.time", event_col="OS")
            cindex = cph.concordance_index_
            
            if not hasattr(self, 'results'):
                self.results = {}
            self.results.setdefault("cox", {})[label] = {"cindex": float(cindex)}
            
            print(f"💊 Cox {label} C-index: {cindex:.3f}")
            
            # 위험점수 계산
            risk_scores = cph.predict_partial_hazard(df)
            df_with_risk = df.copy()
            df_with_risk['risk_score'] = risk_scores
            
            # 중앙값 기준으로 High/Low risk 분리
            median_risk = df_with_risk['risk_score'].median()
            df_with_risk['risk_group'] = df_with_risk['risk_score'].apply(
                lambda x: 'High' if x > median_risk else 'Low'
            )
            
            # High vs Low risk 생존분석
            KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
            import matplotlib.pyplot as plt
            
            # 그래프 설정 적용
            plt.style.use(self.plot_config['style'])
            plt.figure(figsize=self.plot_config['figsize'])
            kmf = KaplanMeierFitter()
            
            for risk_group in ['Low', 'High']:
                group_data = df_with_risk[df_with_risk['risk_group'] == risk_group]
                if len(group_data) > 0:
                    kmf.fit(group_data["OS.time"], event_observed=group_data["OS"], 
                           label=f"{risk_group} Risk (n={len(group_data)})")
                    kmf.plot(ci_show=False)
            
            plt.title(f"Cox Risk Score 기반 생존분석 - {label}", fontsize=self.plot_config['title_fontsize'])
            plt.xlabel("개월", fontsize=self.plot_config['label_fontsize'])
            plt.ylabel("생존확률", fontsize=self.plot_config['label_fontsize'])
            plt.legend(fontsize=self.plot_config['legend_fontsize'])
            plt.grid(True, alpha=self.plot_config['grid_alpha'])
            plt.tight_layout()
            
            # 저장
            output_dir = getattr(self, 'output_dir', './')
            output_path = os.path.join(output_dir, f"cox_risk_km_{label}.{self.plot_config['save_format']}")
            plt.savefig(output_path, dpi=self.plot_config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"✅ Cox risk score 생존곡선 저장: {output_path}")
            
            return cph, cindex
            
        except Exception as e:
            print(f"⚠️ Cox 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    # ===== 생존분석 관련 함수들 추가 =====
    
    def to_patient_index(self, idx):
        """인덱스를 그대로 반환 (가공 없이)"""
        return idx  # 원본 ID 그대로 사용

    def load_metadata_simple(self, meta_tsv: Path):
        """간단한 메타데이터 로드 + 전처리"""
        meta = pd.read_csv(meta_tsv, sep="\t", index_col=0)
        
        # OS.time 단위(일 → 개월) 자동보정
        if meta["OS.time"].max() > 1000:
            meta["OS.time"] = meta["OS.time"] / 30.44
            print("✅ OS.time을 일 → 개월 단위로 변환")
        
        # OS 이벤트 0/1 표준화(문자형 케이스 대비)
        def _to_event(x):
            s = str(x).strip().upper()
            if s in {"1","TRUE","DECEASED","DEAD","EVENT","YES"}: 
                return 1
            try:
                return 1 if float(s) > 0 else 0
            except:
                return 0
        
        meta["OS"] = meta["OS"].map(_to_event).astype(int)
        # meta.index = self.to_patient_index(meta.index)  # ← 이 줄 제거! 원본 인덱스 그대로 사용
        
        # PAM50 컬럼이 있으면 포함, 없으면 생존정보만
        if "PAM50" in meta.columns:
            return meta[["OS.time","OS","PAM50"]]
        else:
            return meta[["OS.time","OS"]]

    def load_dataset_matrix(self, dataset_rate: str, modality: str, impute_base: Path):
        """데이터셋 매트릭스 로드 + 자동 전치"""
        # 파일명 규칙: {modality}_full_{suffix}.tsv (origin은 suffix="origin")
        suffix = "origin" if dataset_rate == "origin" else dataset_rate
        base_path = impute_base / (f"imputation_missorigin" if dataset_rate=="origin" else f"imputation_miss{dataset_rate}")
        
        f = base_path / f"{modality}_full_{suffix}.tsv"
        if not f.exists():
            print(f"⚠️ 파일 없음: {f}")
            return None
            
        X = pd.read_csv(f, sep="\t", index_col=0)
        # X.index = self.to_patient_index(X.index)  # ← 이 줄 제거! 원본 인덱스 그대로 사용
        
        # TCGA ID 기반 안정적 전치
        X = _ensure_samples_by_rows(X)   # ★ 새 로직: TCGA ID 기반 전치
        print(f"   ✅ {modality} 데이터 로드 완료: {X.shape}")
        
        return X

    def cox_on_matrix_with_survival_data(self, X: pd.DataFrame, surv: pd.DataFrame, n_comp: int = 50, penalizer: float = 0.1):
        """매트릭스에 대해 Cox 분석 수행 + PAM50 보정변수 + PAM50 하위유형별 생존분석"""
        try:
            # 상단에서 이미 임포트했으므로 그대로 사용
            # multivariate_logrank_test는 상단에서 확인됨
            
            # 공통 전처리 사용
            result = self._prepare_data_for_cox(X, surv[["OS.time","OS","PAM50"]], n_comp)
            if result[0] is None:
                return None
            
            Xdf, y, keep = result
            
            # 1. Cox 분석 (PAM50을 보정변수로 포함)
            # PAM50을 one-hot encoding으로 변환
            pam50_dummies = pd.get_dummies(y['PAM50'], prefix='PAM50')
            print(f"     🔧 PAM50 one-hot encoding: {list(pam50_dummies.columns)}")
            
            # Cox 분석용 데이터에 PAM50 더미변수 포함
            df_for_cox = Xdf.join(y[["OS.time","OS"]]).join(pam50_dummies)
            print(f"     📊 Cox 분석 데이터 shape: {df_for_cox.shape}")
            
            # ★ 페널티 튜닝
            pen, l1 = tune_cox(df_for_cox)
            cph = CoxPHFitter(penalizer=pen, l1_ratio=l1)
            cph.fit(df_for_cox, duration_col="OS.time", event_col="OS")
            cindex = cph.concordance_index_
            
            print(f"     ✅ Cox 분석 완료: C-index = {cindex:.4f}")
            print(f"     📊 Cox 모델 요약 건너뛰기 (시간 절약)")
            summary = None

            # 2. PAM50 하위유형별 생존분석 (Kaplan-Meier)
            pam50_groups = y['PAM50'].unique()
            print(f"     🔍 PAM50 하위유형: {list(pam50_groups)}")
            
            # Log-rank test (모든 하위유형 비교)
            logrank_result = None
            print(f"     🔍 MULTIVARIATE_LOGRANK_AVAILABLE: {MULTIVARIATE_LOGRANK_AVAILABLE}")
            print(f"     🔍 multivariate_logrank_test: {multivariate_logrank_test}")
            
            if MULTIVARIATE_LOGRANK_AVAILABLE and multivariate_logrank_test is not None:
                try:
                    print(f"     🔍 Log-rank test 시도 중...")
                    logrank_result = multivariate_logrank_test(y['OS.time'], y['PAM50'], y['OS'])
                    print(f"     📊 Log-rank test p-value: {logrank_result.p_value:.6f}")
                except Exception as e:
                    print(f"     ⚠️ Log-rank test 실패: {e}")
                    import traceback
                    print(f"     🔍 Log-rank test 에러 상세: {traceback.format_exc()}")
                    logrank_result = None
            else:
                print("     ⚠️ multivariate_logrank_test를 사용할 수 없어 전체 그룹 비교는 생략됩니다.")
            
            # 생존분석 모델 결과 저장
            print(f"     🔍 생존분석 모델 결과 저장 중...")
            survival_model_results = {
                'survival_times': y['OS.time'].values,
                'events': y['OS'].values,
                'pam50_subtypes': y['PAM50'].values,
                'c_index': float(cindex),
                'n_samples': len(keep),
                'logrank_p_value': float(logrank_result.p_value) if logrank_result and hasattr(logrank_result, 'p_value') else None,
                'cox_model': cph,
                'pam50_data': {},
                'cox_summary': summary if summary is not None else str(cph.summary) if hasattr(cph, 'summary') else None
            }
            print(f"     ✅ survival_model_results 생성 완료")
            
            # 각 PAM50 하위유형별 생존 데이터 저장 (Kaplan-Meier용)
            for subtype in pam50_groups:
                subtype_mask = y['PAM50'] == subtype
                if subtype_mask.sum() >= 5:  # 최소 5개 샘플
                    # Kaplan-Meier 적합
                    kmf = KaplanMeierFitter()
                    kmf.fit(y.loc[subtype_mask, 'OS.time'], y.loc[subtype_mask, 'OS'])
                    
                    survival_model_results['pam50_data'][subtype] = {
                        'times': y.loc[subtype_mask, 'OS.time'].values,
                        'events': y.loc[subtype_mask, 'OS'].values,
                        'n_samples': int(subtype_mask.sum()),
                        'median_survival': float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None,
                        'km_fitter': kmf
                    }
                    print(f"     📊 {subtype}: {subtype_mask.sum()}개 샘플, 중앙생존: {survival_model_results['pam50_data'][subtype]['median_survival']:.1f}개월")
            
            print(f"     🎉 모든 분석 완료! 결과 반환 중...")
            return survival_model_results
            
        except Exception as e:
            print(f"     ❌ Cox 분석 에러: {str(e)}")
            import traceback
            print(f"     🔍 에러 상세: {traceback.format_exc()}")
            return None

    def _prepare_data_for_cox(self, X: pd.DataFrame, surv: pd.DataFrame, n_comp: int = 50):
        """Cox 분석을 위한 데이터 전처리 (공통 로직)"""
        # 생존과 교집합
        keep = X.index.intersection(surv.index)
        if len(keep) < 20:
            print(f"     ⚠️ 공통 샘플 수 부족: {len(keep)} < 20")
            return None, None, None  # 표본 너무 적으면 스킵
        
        print(f"     🔍 공통 샘플 수: {len(keep)}")
        
        X = X.loc[keep]
        y = surv.loc[keep]
        
        # 데이터 타입 확인
        print(f"     📊 X shape: {X.shape}, y shape: {y.shape}")
        print(f"     📊 X index 예시: {list(X.index[:3])}")
        print(f"     📊 y index 예시: {list(y.index[:3])}")

        # 감독형 PCA로 유효 PC만 선택 (성능 향상 중심)
        Xspc, sc, pca, cols = supervised_pca(X, y, max_k=n_comp, var_thr=0.85)
        print(f"     🚀 Performance-Optimized PCA 선택 PC: {len(cols)}개 (예: {cols[:5]})")
        return Xspc, y, keep

    def cox_on_matrix(self, X: pd.DataFrame, surv: pd.DataFrame, n_comp: int = 50, penalizer: float = 0.1):
        """매트릭스에 대해 Cox 분석 수행 (기본 버전)"""
        try:
            # 공통 전처리
            result = self._prepare_data_for_cox(X, surv[["OS.time","OS"]], n_comp)
            if result[0] is None:
                return None, None, 0
            
            Xdf, y, keep = result
            df = Xdf.join(y)
            
            # ★ 페널티 튜닝
            pen, l1 = tune_cox(df)
            cph = CoxPHFitter(penalizer=pen, l1_ratio=l1)
            cph.fit(df, duration_col="OS.time", event_col="OS")
            cindex = float(cph.concordance_index_)

            # 위험도 → median split
            risk = cph.predict_partial_hazard(df).astype(float)
            thr = float(risk.median())
            g0 = y.loc[risk<=thr]
            g1 = y.loc[risk>thr]
            pval = logrank_test(g0["OS.time"], g1["OS.time"], g0["OS"], g1["OS"]).p_value if len(g0)>2 and len(g1)>2 else np.nan

            # KM용 원자료(나중 예쁜 그림에 사용)
            km_blob = {
                "low":  {"t": g0["OS.time"].values, "e": g0["OS"].values, "n": int(len(g0))},
                "high": {"t": g1["OS.time"].values, "e": g1["OS"].values, "n": int(len(g1))}
            }
            return cindex, float(pval) if not np.isnan(pval) else None, len(keep), km_blob
            
        except Exception as e:
            print(f"     ❌ Cox 분석 에러: {str(e)}")
            import traceback
            print(f"     🔍 에러 상세: {traceback.format_exc()}")
            return None, None, 0

    def run_simple_survival_comparison(self, impute_base: str, metadata_tsv: str, output_base: str, n_comp_per_omics: int = 50):
        """간단한 생존분석 비교 실행 - 4개 데이터셋 × 3개 모달리티 + 앙상블"""
        print("🎯 BRCA 생존분석 비교 시작!")
        
        impute_base = Path(impute_base)
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 로드
        print("\n📊 메타데이터 로드 중...")
        survival = self.load_metadata_simple(Path(metadata_tsv))
        print(f"✅ 생존 데이터: {survival.shape}")
        
        dataset_list = ["origin", "10p", "30p", "50p"]
        modalities = ["rna", "protein", "methyl"]
        
        rows = []

        for ds in dataset_list:
            print(f"\n{'='*50}")
            print(f" [{ds.upper()}] 데이터셋 분석 시작")
            print(f"{'='*50}")
            
            # 1) 모달리티별 단독 결과
            risks_for_ensemble = {}
            
            for m in modalities:
                print(f"\n🔍 {m.upper()} 모달리티 분석 중...")
                
                X = self.load_dataset_matrix(ds, m, impute_base)
                if X is None:
                    print(f"   ⚠️ {m} 모달리티 데이터 로드 실패")
                    continue
                    
                print(f"   데이터 크기: {X.shape}")
                
                try:
                    cidx, pval, n, _ = self.cox_on_matrix(X, survival, n_comp_per_omics)
                    
                    if cidx is not None:
                        rows.append({
                            "dataset": ds, 
                            "modality": m, 
                            "n": n, 
                            "C_index": cidx, 
                            "logrank_p": pval
                        })
                        print(f"   ✅ C-index: {cidx:.4f}, logrank p: {pval:.4f}, 샘플: {n}")
                        
                        # 앙상블용 위험도 계산
                        keep = X.index.intersection(survival.index)
                        if n >= 20:
                            # 위험도 다시 계산(동일 로직 재사용)
                            sc = StandardScaler(with_mean=True, with_std=True)
                            Xs = sc.fit_transform(X.loc[keep])
                            k = min(n_comp_per_omics, max(5, min(Xs.shape[0]//10, Xs.shape[1]-1))) if Xs.shape[1] > 10 else min(Xs.shape[1], 10)
                            
                            if k > 0 and Xs.shape[1] > k:
                                pca = PCA(n_components=k, random_state=42)
                                Xs = pca.fit_transform(Xs)
                                cols = [f"PC{i+1}" for i in range(Xs.shape[1])]
                            else:
                                cols = list(X.columns if isinstance(X, pd.DataFrame) else range(Xs.shape[1]))
                            
                            Xdf = pd.DataFrame(Xs, index=keep, columns=cols)
                            df = Xdf.join(survival.loc[keep][["OS.time","OS"]])
                            cph = CoxPHFitter(penalizer=0.1)
                            cph.fit(df, duration_col="OS.time", event_col="OS")
                            risks_for_ensemble[m] = cph.predict_partial_hazard(df).rename(m).astype(float)
                    else:
                        print(f"   ❌ {m} 모달리티 분석 실패")
                except Exception as e:
                    print(f"   ❌ {m} 모달리티 분석 실패: {str(e)}")
                    import traceback
                    print(f"   🔍 에러 상세: {traceback.format_exc()}")

            # 2) 앙상블 분석
            print(f"\n {ds} 데이터셋 앙상블 분석 중...")
            
            if len(risks_for_ensemble) >= 2:
                ens = pd.concat(risks_for_ensemble, axis=1).dropna()
                if len(ens) >= 20:
                    # 각 모달 위험도 z-score → 평균
                    ens_z = (ens - ens.mean())/ens.std(ddof=0)
                    ens_mean = ens_z.mean(axis=1).to_frame("ens")
                    
                    # 단일 공변량 Cox로 C-index/로그랭크 계산
                    y = survival.loc[ens_mean.index][["OS.time","OS"]]
                    df = ens_mean.join(y)
                    cph = CoxPHFitter(penalizer=0.0)
                    cph.fit(df, duration_col="OS.time", event_col="OS")
                    cidx = cph.concordance_index_
                    
                    thr = ens_mean["ens"].median()
                    g0 = y.loc[ens_mean["ens"]<=thr]
                    g1 = y.loc[ens_mean["ens"]>thr]
                    pval = logrank_test(g0["OS.time"], g1["OS.time"], g0["OS"], g1["OS"]).p_value
                    
                    rows.append({
                        "dataset": ds, 
                        "modality": "ensemble(mean_z)", 
                        "n": len(ens_mean), 
                        "C_index": float(cidx), 
                        "logrank_p": float(pval)
                    })
                    print(f"   ✅ 앙상블 C-index: {cidx:.4f}, logrank p: {pval:.4f}, 샘플: {len(ens_mean)}")
                else:
                    print(f"   ⚠️ 앙상블 샘플 수 부족: {len(ens)}")
            else:
                print(f"   ⚠️ 앙상블 가능한 모달리티 부족: {len(risks_for_ensemble)}")

        # 3) 결과 저장/출력
        print(f"\n{'='*60}")
        print("🎉 모든 분석 완료!")
        print(f"{'='*60}")
        
        out_tsv = output_base / "summary_survival.tsv"
        res = pd.DataFrame(rows)
        
        if not res.empty:
            # 보기 좋게 정렬: dataset -> modality -> C_index desc
            order_mod = {k:i for i,k in enumerate(["rna","protein","methyl","ensemble(mean_z)"])}
            res["mod_order"] = res["modality"].map(lambda x: order_mod.get(x, 99))
            res = res.sort_values(["dataset","mod_order","C_index"], ascending=[True, True, False]).drop(columns="mod_order")
            
            # 결과 저장
            res.to_csv(out_tsv, sep="\t", index=False)
            
            print("\n=== 결과 요약 ===")
            print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
            print(f"\n✅ 결과 저장 완료: {out_tsv}")
            print("\n💡 결과 해석:")
            print("• C_index: 높을수록 생존 예측 성능 좋음")
            print("• logrank_p: 낮을수록 High/Low risk 구분 잘됨")
            print("• ensemble(mean_z): 3개 모달리티 통합 성능")
            
            # 최고 성능 데이터셋/모달리티 찾기
            best_cindex = res.loc[res['C_index'].idxmax()]
            print(f"\n🏆 최고 C-index: {best_cindex['dataset']} - {best_cindex['modality']} ({best_cindex['C_index']:.4f})")
            
            return res
        else:
            print("⚠️ 유효한 결과가 없습니다(샘플 수 미달 등).")
            return None
    
    def run_survival_analysis_with_plots(self, impute_base: str, metadata_tsv: str, output_base: str, n_comp_per_omics: int = 50, plot_config: dict = None, min_samples_per_subtype: int = 2, strict_filtering: bool = False):
        """생존분석 실행 + 모달리티 통합 + 그래프 생성"""
        print("🎯 BRCA 생존분석 + 모달리티 통합 + 그래프 생성 시작!")
        
        impute_base = Path(impute_base)
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 로드
        print("\n📊 메타데이터 로드 중...")
        survival = self.load_metadata_simple(Path(metadata_tsv))
        print(f"✅ 생존 데이터: {survival.shape}")
        
        dataset_list = ["origin", "10p", "30p", "50p"]
        modalities = ["rna", "protein", "methyl"]
        
        # 생존분석 모델 결과 저장 (4개 데이터셋만)
        survival_model_results = {}

        for ds in dataset_list:
            print(f"\n{'='*50}")
            print(f" [{ds.upper()}] 데이터셋 모달리티 통합 생존분석 시작")
            print(f"{'='*50}")
            
            # 1. 각 모달리티 데이터 로드 및 통합
            integrated_data = None
            for m in modalities:
                print(f"\n🔍 {m.upper()} 모달리티 데이터 로드 중...")
                
                X = self.load_dataset_matrix(ds, m, impute_base)
                if X is None:
                    print(f"   ⚠️ {m} 모달리티 데이터 로드 실패")
                    continue
                    
                print(f"   데이터 크기: {X.shape}")
                
                # 첫 번째 모달리티면 그대로, 아니면 열 방향으로 합치기
                if integrated_data is None:
                    integrated_data = X
                else:
                    # 공통 샘플만 사용
                    common_samples = integrated_data.index.intersection(X.index)
                    if len(common_samples) < 20:
                        print(f"   ⚠️ 공통 샘플 수 부족: {len(common_samples)} < 20")
                        continue
                    
                    integrated_data = integrated_data.loc[common_samples]
                    X = X.loc[common_samples]
                    
                    # 열 방향으로 합치기 (features 추가)
                    integrated_data = pd.concat([integrated_data, X], axis=1)
                    print(f"   ✅ 통합 완료: {integrated_data.shape}")
            
            if integrated_data is None:
                print(f"   ❌ {ds} 데이터셋 통합 실패")
                continue
            
            print(f"\n🎯 {ds} 데이터셋 통합 데이터: {integrated_data.shape}")
            
            # 2. 통합된 데이터로 생존분석 수행
            result = self.cox_on_matrix_with_survival_data(integrated_data, survival, n_comp_per_omics)
            
            if result is not None:
                survival_model_results[ds] = result
                print(f"   ✅ 생존분석 완료:")
                print(f"      📊 C-index: {result['c_index']:.4f}")
                print(f"      📊 Log-rank p-value: {result['logrank_p_value']:.3e}" if result['logrank_p_value'] else "      📊 Log-rank p-value: N/A")
                print(f"      📊 샘플 수: {result['n_samples']}")
                print(f"      📊 PAM50 하위유형: {list(result['pam50_data'].keys())}")
                
                # 각 하위유형별 중앙생존 시간 출력
                for subtype, data in result['pam50_data'].items():
                    if data['median_survival']:
                        print(f"         • {subtype}: {data['n_samples']}개, 중앙생존 {data['median_survival']:.1f}개월")
                    else:
                        print(f"         • {subtype}: {data['n_samples']}개, 중앙생존 N/A")
                
                # 3. Kaplan-Meier 그래프 생성 및 저장 (전역 설정 적용)
                if plot_config:
                    self.update_plot_config(**plot_config)   # 전역에 반영
                self._create_survival_plots(result, ds, output_base)  # kwargs 없이
                
            else:
                print(f"   ❌ {ds} 데이터셋 생존분석 실패")

        # 결과 저장 (안전한 pickle 저장)
        import pickle
        
        def _strip_for_pickle(res):
            """pickle 저장 전 모델 객체 제거"""
            clean = {}
            for ds, r in res.items():
                rr = {k: v for k, v in r.items() if k != "cox_model"}
                if "pam50_data" in rr:
                    pdict = {}
                    for sub, d in rr["pam50_data"].items():
                        pdict[sub] = {kk: vv for kk, vv in d.items() if kk != "km_fitter"}
                    rr["pam50_data"] = pdict
                clean[ds] = rr
            return clean
        
        results_file = output_base / "survival_model_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(_strip_for_pickle(survival_model_results), f)
        
        # 요약 결과도 TSV로 저장 (4개 데이터셋만)
        summary_rows = []
        for ds in dataset_list:
            if ds in survival_model_results:
                result = survival_model_results[ds]
                summary_rows.append({
                    'dataset': ds,
                    'c_index': result['c_index'],
                    'logrank_p_value': result['logrank_p_value'],
                    'n_samples': result['n_samples'],
                    'pam50_subtypes': ', '.join(list(result['pam50_data'].keys())),
                    'median_survival_basal': result['pam50_data'].get('Basal', {}).get('median_survival', 'N/A'),
                    'median_survival_her2': result['pam50_data'].get('Her2', {}).get('median_survival', 'N/A'),
                    'median_survival_luma': result['pam50_data'].get('LumA', {}).get('median_survival', 'N/A'),
                    'median_survival_lumb': result['pam50_data'].get('LumB', {}).get('median_survival', 'N/A'),
                    'median_survival_normal': result['pam50_data'].get('Normal', {}).get('median_survival', 'N/A')
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = output_base / "survival_analysis_summary.tsv"
            summary_df.to_csv(summary_file, sep="\t", index=False)
            print(f"📊 요약 결과 저장: {summary_file}")
        
        print(f"\n🎉 생존분석 모델 결과 저장 완료: {results_file}")
        print(f"📊 저장된 데이터셋: {list(survival_model_results.keys())}")
        
        # 전체 결과 요약 출력 (4개 데이터셋만)
        print(f"\n{'='*60}")
        print("🎯 생존분석 결과 요약 (모달리티 통합)")
        print(f"{'='*60}")
        for ds in dataset_list:
            if ds in survival_model_results:
                result = survival_model_results[ds]
                print(f"\n📊 [{ds.upper()}] 데이터셋:")
                print(f"   🔍 통합 모달리티: C-index={result['c_index']:.4f}, Log-rank p={result['logrank_p_value']:.3e}" if result['logrank_p_value'] else f"   🔍 통합 모달리티: C-index={result['c_index']:.4f}, Log-rank p=N/A")
                print(f"   📊 샘플 수: {result['n_samples']}")
                print(f"   🏷️ PAM50 하위유형: {', '.join(list(result['pam50_data'].keys()))}")
        
        return survival_model_results
    
    def update_plot_config(self, **kwargs):
        """그래프 설정 업데이트"""
        self._apply_plot_config(kwargs)
        print("✅ 그래프 설정 업데이트 완료!")
        print(f"현재 설정: {self.plot_config}")
    
    def get_plot_config(self):
        """현재 그래프 설정 반환"""
        return self.plot_config.copy()
    
    def _create_survival_plots(self, result: dict, dataset_name: str, output_base: Path, **kwargs):
        """생존분석 결과로부터 PAM50 하위유형별 Kaplan-Meier 그래프 생성 (전역 설정 사용)"""
        try:
            import matplotlib.pyplot as plt
            
            # 전역 설정 사용 (kwargs가 있으면 덮어쓰기 허용)
            cfg = self.plot_config.copy()
            cfg.update({k: v for k, v in kwargs.items() if k in cfg})
            
            print(f"   🎨 전역 설정 적용 중: {cfg}")
            
            # 커스텀 스타일 적용
            plt.style.use(cfg.get('style', 'seaborn-v0_8'))
            
            # 그래프 크기 및 스타일 설정
            fig, ax = plt.subplots(1, 1, figsize=cfg.get('figsize', (14, 10)))
            fig.suptitle(f'📈 PAM50 Subtype Survival Analysis - {dataset_name.upper()} Dataset', 
                        fontsize=cfg.get('title_fontsize', 20), fontweight='bold', y=0.95)
            
            # PAM50 하위유형별 생존 곡선 (전역 설정 사용)
            pam50_groups = list(result['pam50_data'].keys())
            colors = cfg.get('colors', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'])
            linewidth = float(cfg.get('linewidth', 4.0))
            
            # lifelines 지연 임포트
            KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
            
            for i, subtype in enumerate(pam50_groups):
                color = colors[i % len(colors)]
                
                data = result['pam50_data'][subtype]
                kmf_subtype = KaplanMeierFitter()
                kmf_subtype.fit(data['times'], data['events'])
                
                # 생존 곡선 (전역 설정 사용)
                ax.plot(kmf_subtype.survival_function_.index, kmf_subtype.survival_function_.values,
                       linewidth=linewidth, color=color, 
                       label=f'{subtype} (n={data["n_samples"]})', alpha=0.9)
                
                # 중앙 생존 시간 표시 (점선으로) - 안전한 선두께 계산
                if data['median_survival']:
                    mid_lw = max(0.5 * linewidth, 1.0)  # // 대신 * 0.5 사용
                    ax.axvline(data['median_survival'], color=color, linestyle='--', 
                              alpha=0.8, linewidth=mid_lw)
            
            # 축 라벨 및 제목 (전역 설정 사용)
            ax.set_xlabel('Survival Time (months)', fontsize=cfg.get('label_fontsize', 16), fontweight='bold')
            ax.set_ylabel('Survival Probability', fontsize=cfg.get('label_fontsize', 16), fontweight='bold')
            ax.set_title(f'🏷️ PAM50 Subtype Survival Curves\nC-index: {result["c_index"]:.4f}', 
                        fontsize=cfg.get('title_fontsize', 18), fontweight='bold', pad=30)
            
            # 범례 (전역 설정 사용)
            ax.legend(frameon=True, fancybox=True, shadow=True, 
                     loc='upper right', fontsize=cfg.get('legend_fontsize', 14), 
                     bbox_to_anchor=(1.15, 1.0))
            
            # 그리드 및 축 설정 (전역 설정 사용)
            ax.grid(True, alpha=cfg.get('grid_alpha', 0.4), linestyle='-', linewidth=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, None)
            
            # 축 눈금 및 폰트 설정
            ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            # Log-rank p-value 표시 (커스텀 스타일)
            if result['logrank_p_value']:
                ax.text(0.05, 0.95, f'Log-rank test\np = {result["logrank_p_value"]:.3e}', 
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.8", facecolor="white", 
                                alpha=0.95, edgecolor='gray', linewidth=2))
            
            # 배경색 및 전체 스타일링
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor(cfg.get('facecolor', 'white'))
            
            plt.tight_layout()
            
            # 저장 (전역 설정 적용)
            plot_file = output_base / f"survival_analysis_{dataset_name}.{cfg.get('save_format', 'png')}"
            plt.savefig(plot_file, 
                       dpi=cfg.get('dpi', 300), 
                       bbox_inches=cfg.get('bbox_inches', 'tight'),
                       facecolor=cfg.get('facecolor', 'white'), 
                       edgecolor='none')
            plt.close()
            
            print(f"   🎨 Global Config PAM50 Survival Plot Saved: {plot_file}")
            
        except Exception as e:
            print(f"   ⚠️ Plot Generation Failed: {e}")
            import traceback
            print(f"   🔍 Error Details: {traceback.format_exc()}")
    
    def _create_survival_plots_custom(self, result: dict, dataset_name: str, output_base: Path, **kwargs):
        """생존분석 결과로부터 PAM50 하위유형별 Kaplan-Meier 그래프 생성 (완전 커스터마이징)"""
        try:
            import matplotlib.pyplot as plt
            
            # matplotlib 스타일 적용
            plt.style.use(kwargs.get('style', 'seaborn-v0_8'))
            
            # 그래프 생성
            fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (14, 10)))
            
            # 제목 설정
            fig.suptitle(f'📈 PAM50 Subtype Survival Analysis - {dataset_name.upper()} Dataset', 
                        fontsize=kwargs.get('title_fontsize', 20), fontweight='bold', y=0.95)
            
            # PAM50 하위유형별 생존 곡선
            pam50_groups = list(result['pam50_data'].keys())
            colors = kwargs.get('colors', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'])
            
            # lifelines 지연 임포트
            KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
            
            for i, subtype in enumerate(pam50_groups):
                color = colors[i % len(colors)]
                data = result['pam50_data'][subtype]
                kmf_subtype = KaplanMeierFitter()
                kmf_subtype.fit(data['times'], data['events'])
                
                # 생존 곡선 (사용자 설정 적용)
                ax.plot(kmf_subtype.survival_function_.index, kmf_subtype.survival_function_.values,
                       linewidth=kwargs.get('linewidth', 4), color=color, 
                       label=f'{subtype} (n={data["n_samples"]})', alpha=0.9)
                
                # 중앙 생존 시간
                if data['median_survival']:
                    ax.axvline(data['median_survival'], color=color, linestyle='--', 
                              alpha=0.8, linewidth=kwargs.get('linewidth', 4)//2)
            
            # 축 설정
            ax.set_xlabel('Survival Time (months)', fontsize=kwargs.get('label_fontsize', 16), fontweight='bold')
            ax.set_ylabel('Survival Probability', fontsize=kwargs.get('label_fontsize', 16), fontweight='bold')
            ax.set_title(f'🏷️ PAM50 Subtype Survival Curves\nC-index: {result["c_index"]:.4f}', 
                        fontsize=kwargs.get('title_fontsize', 18), fontweight='bold', pad=30)
            
            # 범례
            ax.legend(frameon=True, fancybox=True, shadow=True, 
                     loc='upper right', fontsize=kwargs.get('legend_fontsize', 14), 
                     bbox_to_anchor=(1.15, 1.0))
            
            # 그리드 및 스타일
            ax.grid(True, alpha=kwargs.get('grid_alpha', 0.4), linestyle='-', linewidth=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, None)
            
            # 축 스타일
            ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            # Log-rank p-value
            if result['logrank_p_value']:
                ax.text(0.05, 0.95, f'Log-rank test\np = {result["logrank_p_value"]:.3e}', 
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.8", facecolor="white", 
                                alpha=0.95, edgecolor='gray', linewidth=2))
            
            # 배경색
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor(kwargs.get('facecolor', 'white'))
            
            plt.tight_layout()
            
            # 저장
            plot_file = output_base / f"survival_analysis_{dataset_name}.{kwargs.get('save_format', 'png')}"
            plt.savefig(plot_file, 
                       dpi=kwargs.get('dpi', 300), 
                       bbox_inches=kwargs.get('bbox_inches', 'tight'),
                       facecolor=kwargs.get('facecolor', 'white'), 
                       edgecolor=kwargs.get('edgecolor', 'none'))
            plt.close()
            
            print(f"   🎨 Custom PAM50 Survival Plot Saved: {plot_file}")
            
        except Exception as e:
            print(f"   ⚠️ Custom Plot Generation Failed: {e}")
            import traceback
            print(f"   🔍 Error Details: {traceback.format_exc()}")
    
    def create_pam50_survival_plots_custom(self, results_dict: dict, output_dir: Path, **matplotlib_kwargs):
        """PAM50 하위유형별 생존 곡선 생성 (matplotlib 스타일 완전 커스터마이징)"""
        
        # 기본값 설정
        defaults = {
            'figsize': (14, 10),
            'dpi': 300,
            'linewidth': 4,
            'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'],
            'title_fontsize': 20,
            'label_fontsize': 16,
            'legend_fontsize': 14,
            'grid_alpha': 0.4,
            'save_format': 'png',
            'style': 'seaborn-v0_8',
            'facecolor': 'white',
            'edgecolor': 'none',
            'bbox_inches': 'tight'
        }
        
        # 사용자 입력으로 기본값 덮어쓰기
        defaults.update(matplotlib_kwargs)
        
        for dataset_name, result in results_dict.items():
            self._create_survival_plots_custom(
                result, dataset_name, output_dir, **defaults
            )
    
    def plot_summary_pretty(self, res_df: pd.DataFrame, outdir: Path, 
                           heatmap_cmap="YlGnBu", heatmap_vmin=0.45, heatmap_vmax=0.85,
                           bar_colors=None, figure_size_heatmap=(8,4.5), figure_size_bar=(9,4.8),
                           title_fontsize=14, annotation_fontsize=9, dpi=300):
        """성능 요약: C-index 히트맵 + 막대 (사용자 커스터마이징 가능)"""
        outdir.mkdir(parents=True, exist_ok=True)
        
        # 기본 색상 설정
        if bar_colors is None:
            bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        
        # 히트맵용 피벗
        pv = res_df.pivot(index="dataset", columns="modality", values="C_index").reindex(index=["origin","10p","30p","50p"])
        plt.figure(figsize=figure_size_heatmap)
        sns.heatmap(pv, annot=True, fmt=".3f", cmap=heatmap_cmap, vmin=heatmap_vmin, vmax=heatmap_vmax,
                    cbar_kws={"label":"C-index"}, linewidths=.5, linecolor="#EEEEEE")
        plt.title("BRCA Survival — C-index by Dataset × Modality", fontsize=title_fontsize, pad=12)
        plt.xlabel(""); plt.ylabel("")
        plt.tight_layout()
        plt.savefig(outdir/"cindex_heatmap.png", dpi=dpi)
        plt.close()

        # 막대 (모달리티별 비교)
        order = ["rna","protein","methyl","ensemble(mean_z)"]
        gg = res_df.copy()
        gg["modality"] = pd.Categorical(gg["modality"], order)
        plt.figure(figsize=figure_size_bar)
        ax = sns.barplot(data=gg, x="dataset", y="C_index", hue="modality", hue_order=order, 
                        edgecolor="white", palette=bar_colors)
        for p in ax.patches:
            h = p.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2, h+0.005), 
                           ha="center", va="bottom", fontsize=annotation_fontsize)
        ax.set_ylim(0.45, 0.90)
        ax.grid(axis='y', alpha=0.25)
        plt.title("C-index Comparison", fontsize=title_fontsize, pad=12)
        plt.tight_layout()
        plt.savefig(outdir/"cindex_bar.png", dpi=dpi)
        plt.close()
    
    def plot_km_panels_by_dataset(self, km_store: dict, outdir: Path, survival: pd.DataFrame, results_df: pd.DataFrame,
                                 colors=None, figure_size=(14, 10), title_fontsize=18, subtitle_fontsize=13,
                                 linewidth=2, grid_alpha=0.3, dpi=300, save_format="png"):
        """진짜 KM: 데이터셋별 2×2 패널 (RNA / Protein / Methyl / Ensemble) - 사용자 커스터마이징 가능"""
        
        # 기본 색상 설정
        if colors is None:
            colors = {"rna":"#4585D3", "protein":"#ECB54F", "methyl":"#F16767", "ensemble(mean_z)":"#6AA57A"}
        
        for ds, mods in km_store.items():
            # 2x2 패널
            fig, axes = plt.subplots(2, 2, figsize=figure_size)
            axes = axes.ravel()
            title = f"KM Curves by Risk (High vs Low) — {ds.upper()}"
            fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=0.98)

            show_list = []
            for m in ["rna","protein","methyl","ensemble(mean_z)"]:
                if m in mods: show_list.append(m)
            for i, m in enumerate(show_list):
                ax = axes[i]
                blob = mods[m]
                kmf = KaplanMeierFitter()
                
                # 색상 선택
                color = colors.get(m, f"#{np.random.randint(0, 0xFFFFFF):06x}")
                
                # Low
                kmf.fit(blob["low"]["t"], event_observed=blob["low"]["e"], label=f"Low (n={blob['low']['n']})")
                kmf.plot(ax=ax, ci_show=False, linewidth=linewidth, color=color)
                # High
                kmf.fit(blob["high"]["t"], event_observed=blob["high"]["e"], label=f"High (n={blob['high']['n']})")
                kmf.plot(ax=ax, ci_show=False, linewidth=linewidth, color=color)

                # p-value 텍스트
                row = results_df[(results_df.dataset==ds) & (results_df.modality==m)]
                ptxt = f"p={row['logrank_p'].values[0]:.2e}" if not row.empty and pd.notna(row['logrank_p'].values[0]) else "p=N/A"
                ax.text(0.03, 0.95, ptxt, transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9), 
                        fontsize=10, fontweight="bold")
                ax.set_title(m.upper(), fontsize=subtitle_fontsize, pad=6)
                ax.set_xlabel("Months"); ax.set_ylabel("Survival")
                ax.grid(alpha=grid_alpha, linestyle="--")
                ax.set_ylim(0,1.05)

            for j in range(len(show_list), 4):  # 빈칸 처리
                fig.delaxes(axes[j])
            plt.tight_layout()
            plt.savefig(outdir/f"km_panels_{ds}.{save_format}", dpi=dpi, bbox_inches="tight")
            plt.close()
    
    def data_quality_analysis(self):
        """2. 데이터 품질 심화 분석"""
        print("\n📈 2단계: 데이터 품질 심화 분석")
        
        # 전체 상관관계
        train_corr = np.corrcoef(
            self.original_aligned.values.flatten(), 
            self.train_aligned.values.flatten()
        )[0, 1]
        valid_corr = np.corrcoef(
            self.original_aligned.values.flatten(), 
            self.valid_aligned.values.flatten()
        )[0, 1]
        
        # 분포 비교 (결측치가 없는 위치에서)
        non_missing_mask = ~self.mask_aligned.astype(bool)
        
        if non_missing_mask.to_numpy().sum() > 0:
            # 관측된 값들의 분포 비교
            original_obs = self.original_aligned.to_numpy()[non_missing_mask.to_numpy()]
            train_obs = self.train_aligned.to_numpy()[non_missing_mask.to_numpy()]
            valid_obs = self.valid_aligned.to_numpy()[non_missing_mask.to_numpy()]
            
            # 분포 통계
            distribution_stats = {
                'original': {
                    'mean': float(np.mean(original_obs)),
                    'std': float(np.std(original_obs)),
                    'min': float(np.min(original_obs)),
                    'max': float(np.max(original_obs))
                },
                'train_imputed': {
                    'mean': float(np.mean(train_obs)),
                    'std': float(np.std(train_obs)),
                    'min': float(np.min(train_obs)),
                    'max': float(np.max(train_obs))
                },
                'valid_imputed': {
                    'mean': float(np.mean(valid_obs)),
                    'std': float(np.std(valid_obs)),
                    'min': float(np.min(valid_obs)),
                    'max': float(np.max(valid_obs))
                }
            }
        else:
            distribution_stats = None
        
        self.results['data_quality'] = {
            'overall_correlation': {
                'train': float(train_corr) if not np.isnan(train_corr) else None,
                'valid': float(valid_corr) if not np.isnan(valid_corr) else None
            },
            'distribution_comparison': distribution_stats
        }
        
        print(f"전체 상관계수 (훈련): {train_corr:.4f}")
        print(f"전체 상관계수 (검증): {valid_corr:.4f}")
        
        return train_corr, valid_corr
    
    def feature_level_analysis(self):
        """3. 특징별 성능 분석"""
        print("\n🧬 3단계: 특징별 성능 분석")
        
        feature_performance = []
        n_features = self.original_aligned.shape[1]
        
        for i in range(n_features):
            col_mask = self.mask_aligned.iloc[:, i].astype(bool)
            if col_mask.sum() > 0:  # 결측치가 있는 특징만
                col_metrics = calculate_imputation_metrics(
                    self.original_aligned.iloc[:, i].values.reshape(-1, 1),
                    self.train_aligned.iloc[:, i].values.reshape(-1, 1),
                    self.mask_aligned.iloc[:, i].values.reshape(-1, 1)
                )
                
                feature_performance.append({
                    'feature_id': int(i),
                    'feature_name': str(self.original_aligned.columns[i]),   # ✅ 실제 특징명
                    'n_missing': int(col_mask.sum()),
                    'missing_ratio': float(col_mask.sum() / len(col_mask)),
                    'rmse': col_metrics['rmse'],
                    'mae': col_metrics['mae'],
                    'pearson': col_metrics['pearson']
                })
        
        # 성능 순위별 정렬
        feature_performance.sort(key=lambda x: x['rmse'] if x['rmse'] is not None else float('inf'))
        
        # 상위/하위 성능 특징
        top_performers = feature_performance[:10]  # 상위 10개
        bottom_performers = feature_performance[-10:]  # 하위 10개
        
        self.results['feature_analysis'] = {
            'total_features_with_missing': len(feature_performance),
            'feature_performance': feature_performance,
            'top_performers': top_performers,
            'bottom_performers': bottom_performers,
            'performance_summary': {
                'mean_rmse': float(np.mean([f['rmse'] for f in feature_performance if f['rmse'] is not None])),
                'mean_mae': float(np.mean([f['mae'] for f in feature_performance if f['mae'] is not None])),
                'mean_pearson': float(np.mean([f['pearson'] for f in feature_performance if f['pearson'] is not None]))
            }
        }
        
        print(f"결측치가 있는 특징 수: {len(feature_performance)}")
        print(f"평균 RMSE: {self.results['feature_analysis']['performance_summary']['mean_rmse']:.4f}")
        print(f"평균 Pearson: {self.results['feature_analysis']['performance_summary']['mean_pearson']:.4f}")
        
        return feature_performance
    
    def subtype_performance_analysis(self):
        """4. 서브타입별 성능 분석"""
        print("\n🏥 4단계: 서브타입별 성능 분석")
        
        if self.subtype_data is None:
            print("⚠️ 서브타입 정보가 없어 건너뜁니다.")
            return None
        
        # 서브타입별 성능 계산
        subtype_performance = {}
        
        for subtype in self.subtype_aligned.columns:
            subtype_samples = self.subtype_aligned[self.subtype_aligned[subtype] == 1].index
            common_subtype_samples = subtype_samples.intersection(self.common_samples)
            
            if len(common_subtype_samples) > 5:  # 최소 5개 샘플
                # 해당 서브타입 샘플들의 성능 계산
                subtype_mask = self.mask_aligned.loc[common_subtype_samples]
                subtype_original = self.original_aligned.loc[common_subtype_samples]
                subtype_train = self.train_aligned.loc[common_subtype_samples]
                subtype_valid = self.valid_aligned.loc[common_subtype_samples]
                
                # 훈련 데이터 성능
                train_metrics = calculate_imputation_metrics(
                    subtype_original.values, 
                    subtype_train.values, 
                    subtype_mask.values
                )
                
                # 검증 데이터 성능
                valid_metrics = calculate_imputation_metrics(
                    subtype_original.values, 
                    subtype_valid.values, 
                    subtype_mask.values
                )
                
                subtype_performance[subtype] = {
                    'n_samples': len(common_subtype_samples),
                    'train_performance': train_metrics,
                    'valid_performance': valid_metrics
                }
        
        self.results['subtype_analysis'] = subtype_performance
        
        if subtype_performance:
            print("서브타입별 성능:")
            for subtype, perf in subtype_performance.items():
                print(f"  {subtype}: {perf['n_samples']}개 샘플, "
                      f"RMSE={perf['train_performance']['rmse']:.4f}")
        
        return subtype_performance
    
    def dimensionality_analysis(self):
        """5. 차원 축소를 통한 데이터 구조 분석 (공동 t-SNE)"""
        print("\n🔬 5단계: 차원 축소를 통한 데이터 구조 분석")
        
        try:
            # PCA: 원본에 fit, imputed는 transform
            pca = PCA(n_components=min(50, self.original_aligned.shape[1]), random_state=42)
            original_pca = pca.fit_transform(self.original_aligned.values)
            train_pca = pca.transform(self.train_aligned.values)
            valid_pca = pca.transform(self.valid_aligned.values)
            evr = pca.explained_variance_ratio_

            # (선택) t-SNE: 세 세트를 합쳐 한 번만 수행
            # 표본 수 너무 크면 스킵(리소스 보호)
            tsne_coords = None
            max_n = 1500
            n = len(self.common_samples)
            if n * 3 <= max_n:
                X = np.vstack([original_pca, train_pca, valid_pca])
                y = (["original"]*n) + (["train_imputed"]*n) + (["valid_imputed"]*n)
                # 스케일링은 보통 PCA 후 불필요하지만, 안정성 위해 표준화 허용
                Xs = StandardScaler().fit_transform(X)
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, Xs.shape[0]-1))
                Z = tsne.fit_transform(Xs)
                tsne_coords = {
                    'x': Z[:,0].tolist(), 'y': Z[:,1].tolist(), 'label': y
                }

            self.results['dimensionality_analysis'] = {
                'pca': {
                    'original_explained_variance': evr.tolist(),
                    'cumulative_variance': np.cumsum(evr).tolist(),
                    'n_components_90': int(np.argmax(np.cumsum(evr) >= 0.9) + 1),
                },
                'tsne': {'available': tsne_coords is not None}
            }
            if tsne_coords is not None:
                self.results['dimensionality_analysis']['tsne']['coords'] = tsne_coords

            print(f"90% 분산 설명에 필요한 주성분 수: {self.results['dimensionality_analysis']['pca']['n_components_90']}")
            return evr
            
        except Exception as e:
            print(f"⚠️ 차원 축소 분석 중 오류: {e}")
            return None
    
    def clinical_relevance_analysis(self):
        """6. 임상적 관련성 분석 (실제 생존 데이터)"""
        print("\n💊 6단계: 임상적 관련성 분석")
        
        if not LIFELINES_AVAILABLE:
            print("⚠️ lifelines 패키지가 없어 생존 분석을 건너뜁니다.")
            return None
        
        if self.survival_aligned is None:
            print("⚠️ 생존 데이터가 없어 건너뜁니다.")
            return None
        
        try:
            print(f"생존 데이터 샘플 수: {len(self.survival_aligned)}")
            print(f"생존 이벤트 비율: {self.survival_aligned['OS'].mean():.2%}")
            print(f"평균 생존 시간: {self.survival_aligned['OS.time'].mean():.1f}개월")
            
            # 생존 예측 모델 성능 비교
            # 원본 데이터로 생존 예측
            original_scores = self._predict_survival_real(
                self.original_aligned, self.survival_aligned
            )
            
            # Imputed 데이터로 생존 예측
            train_scores = self._predict_survival_real(
                self.train_aligned, self.survival_aligned
            )
            
            valid_scores = self._predict_survival_real(
                self.valid_aligned, self.survival_aligned
            )
            
            # 생존 시간 예측 성능 (회귀)
            original_time_scores = self._predict_survival_time(
                self.original_aligned, self.survival_aligned
            )
            
            train_time_scores = self._predict_survival_time(
                self.train_aligned, self.survival_aligned
            )
            
            valid_time_scores = self._predict_survival_time(
                self.valid_aligned, self.survival_aligned
            )
            
            clinical_results = {
                'survival_event_prediction': {
                    'original_auc': original_scores,
                    'train_imputed_auc': train_scores,
                    'valid_imputed_auc': valid_scores,
                    'improvement_train': train_scores - original_scores if train_scores and original_scores else None,
                    'improvement_valid': valid_scores - original_scores if valid_scores and original_scores else None
                },
                'survival_time_prediction': {
                    'original_r2': original_time_scores,
                    'train_imputed_r2': train_time_scores,
                    'valid_imputed_r2': valid_time_scores,
                    'improvement_train': train_time_scores - original_time_scores if train_time_scores and original_time_scores else None,
                    'improvement_valid': valid_time_scores - original_time_scores if valid_time_scores and original_time_scores else None
                },
                'survival_statistics': {
                    'n_samples': len(self.survival_aligned),
                    'event_rate': float(self.survival_aligned['OS'].mean()),
                    'mean_survival_time': float(self.survival_aligned['OS.time'].mean()),
                    'median_survival_time': float(self.survival_aligned['OS.time'].median())
                }
            }
            
            self.results['clinical_analysis'] = clinical_results
            
            print(f"생존 이벤트 예측 AUC (원본): {original_scores:.4f}")
            print(f"생존 이벤트 예측 AUC (훈련 imputed): {train_scores:.4f}")
            print(f"생존 이벤트 예측 AUC (검증 imputed): {valid_scores:.4f}")
            print(f"생존 시간 예측 R² (원본): {original_time_scores:.4f}")
            print(f"생존 시간 예측 R² (훈련 imputed): {train_time_scores:.4f}")
            print(f"생존 시간 예측 R² (검증 imputed): {valid_time_scores:.4f}")
            
            return clinical_results
            
        except Exception as e:
            print(f"⚠️ 임상적 관련성 분석 중 오류: {e}")
            return None
    
    def _predict_survival_real(self, data, survival_df):
        """실제 생존 데이터로 생존 이벤트 예측"""
        try:
            # 공통 샘플 찾기
            common = data.index.intersection(survival_df.index)
            if len(common) < 10:
                return None
            
            X = data.loc[common].values
            y = survival_df.loc[common, 'OS'].values.astype(int)  # 생존 이벤트 (0/1)

            clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            return float(np.mean(scores))
            
        except Exception as e:
            print(f"생존 이벤트 예측 오류: {e}")
            return None
    
    def _predict_survival_time(self, data, survival_df):
        """실제 생존 데이터로 생존 시간 예측"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # 공통 샘플 찾기
            common = data.index.intersection(survival_df.index)
            if len(common) < 10:
                return None
            
            X = data.loc[common].values
            y = survival_df.loc[common, 'OS.time'].values  # 생존 시간
            
            # Random Forest 회귀기 (개선된 설정)
            reg = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12, n_jobs=-1)
            
            # 교차 검증 (R²)
            scores = cross_val_score(reg, X, y, cv=5, scoring='r2', n_jobs=-1)
            return float(np.mean(scores))
            
        except Exception as e:
            print(f"생존 시간 예측 오류: {e}")
            return None
    
    def generate_visualizations(self, output_dir):
        """시각화 생성"""
        print("\n🎨 시각화 생성 중...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 기본 분석 요약 시각화
        self._generate_basic_summary_plots(output_dir)
        
        # 2. 생존 분석 시각화 (생존 데이터가 있을 때)
        if self.survival_aligned is not None:
            self._generate_survival_plots(output_dir)
        
        print(f"✅ 모든 시각화 저장 완료!")
    
    def _generate_basic_summary_plots(self, output_dir):
        """기본 분석 요약 시각화"""
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 다운스트림 분석 종합 요약', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Imputation 성능 비교 (RMSE, MAE, Pearson)
        metrics = ['RMSE', 'MAE', 'Pearson']
        train_values = [self.results['basic_imputation']['train_performance']['rmse'], 
                       self.results['basic_imputation']['train_performance']['mae'],
                       self.results['basic_imputation']['train_performance']['pearson']]
        valid_values = [self.results['basic_imputation']['valid_performance']['rmse'],
                       self.results['basic_imputation']['valid_performance']['mae'],
                       self.results['basic_imputation']['valid_performance']['pearson']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, train_values, width, label='훈련 데이터', 
                               color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = axes[0, 0].bar(x + width/2, valid_values, width, label='검증 데이터', 
                               color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
        
        axes[0, 0].set_xlabel('평가 메트릭', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('값', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('🔍 Imputation 성능 비교', fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, fontweight='bold')
        axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 특징별 성능 분포
        if 'feature_analysis' in self.results:
            rmse_values = [f['rmse'] for f in self.results['feature_analysis']['feature_performance'] if f['rmse'] is not None]
            
            axes[0, 1].hist(rmse_values, bins=40, alpha=0.7, color=colors[2], 
                           edgecolor='white', linewidth=1)
            axes[0, 1].axvline(np.mean(rmse_values), color=colors[3], linestyle='--', 
                              linewidth=2, label=f'평균: {np.mean(rmse_values):.3f}')
            axes[0, 1].axvline(np.median(rmse_values), color=colors[1], linestyle='--', 
                              linewidth=2, label=f'중앙값: {np.median(rmse_values):.3f}')
            
            axes[0, 1].set_xlabel('RMSE', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('빈도', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('📈 특징별 RMSE 분포', fontsize=14, fontweight='bold', pad=20)
            axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 서브타입별 성능
        if 'subtype_analysis' in self.results and self.results['subtype_analysis']:
            subtypes = list(self.results['subtype_analysis'].keys())
            subtype_rmse = [self.results['subtype_analysis'][s]['train_performance']['rmse'] for s in subtypes]
            
            bars = axes[1, 0].bar(subtypes, subtype_rmse, color=colors[:len(subtypes)], 
                                 alpha=0.8, edgecolor='white', linewidth=1)
            axes[1, 0].set_xlabel('서브타입', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('🏷️ 서브타입별 Imputation 성능', fontsize=14, fontweight='bold', pad=20)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, subtype_rmse):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 생존 예측 성능 또는 PCA
        if 'clinical_analysis' in self.results and self.results['clinical_analysis']:
            clinical = self.results['clinical_analysis']
            
            if 'survival_event_prediction' in clinical:
                event_pred = clinical['survival_event_prediction']
                auc_values = [event_pred.get('original_auc', 0), 
                            event_pred.get('train_imputed_auc', 0),
                            event_pred.get('valid_imputed_auc', 0)]
                auc_labels = ['원본', '훈련\nImputed', '검증\nImputed']
                
                bars = axes[1, 1].bar(auc_labels, auc_values, color=colors[:3], alpha=0.8, 
                                     edgecolor='white', linewidth=1)
                axes[1, 1].set_xlabel('데이터', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('AUC', fontsize=12, fontweight='bold')
                axes[1, 1].set_title('💊 생존 이벤트 예측 성능', fontsize=14, fontweight='bold', pad=20)
                axes[1, 1].set_ylim(0, 1.05)
                axes[1, 1].grid(True, alpha=0.3)
                
                # AUC 값 표시
                for bar, value in zip(bars, auc_values):
                    if value > 0:
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        elif 'dimensionality_analysis' in self.results:
            pca_data = self.results['dimensionality_analysis']['pca']
            cumulative_var = pca_data['cumulative_variance']
            
            axes[1, 1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                           color=colors[0], linewidth=3, marker='o', markersize=6)
            axes[1, 1].axhline(y=0.9, color=colors[3], linestyle='--', linewidth=2, 
                              label='90% 분산 기준선')
            axes[1, 1].fill_between(range(1, len(cumulative_var) + 1), cumulative_var, 
                                   alpha=0.3, color=colors[0])
            axes[1, 1].set_xlabel('주성분 수', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('누적 분산 설명 비율', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('📊 PCA 분산 설명', fontsize=14, fontweight='bold', pad=20)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'basic_analysis_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 기본 분석 요약 시각화 저장: {output_dir}/basic_analysis_summary.png")
    
    def _generate_survival_plots(self, output_dir):
        """생존 분석 시각화"""
        print("🏥 생존 분석 시각화 생성 중...")
        
        # 1. Kaplan-Meier 생존 곡선
        self._plot_kaplan_meier_curves(output_dir)
        
        # 2. 생존 예측 성능 비교
        self._plot_survival_prediction_performance(output_dir)
        
        # 3. 생존 시간 분포 비교
        self._plot_survival_time_distributions(output_dir)
        
        # 4. 생존 예측 모델 성능 상세 분석
        self._plot_detailed_survival_analysis(output_dir)
        
        # 5. 서브타입별 생존 분석 (PAM50 등)
        self._plot_subtype_survival_analysis(output_dir)
    
    def _plot_kaplan_meier_curves(self, output_dir):
        """Kaplan-Meier 생존 곡선"""
        try:
            # 스타일 설정
            plt.style.use('seaborn-v0_8')
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle('📈 Kaplan-Meier 생존 곡선 분석', fontsize=20, fontweight='bold', y=0.95)
            
            # 전체 생존 곡선
            kmf = KaplanMeierFitter()
            kmf.fit(self.survival_aligned['OS.time'], self.survival_aligned['OS'])
            
            axes[0].plot(kmf.survival_function_.index, kmf.survival_function_.values, 
                        linewidth=3, color=colors[0], label='전체 생존 곡선')
            axes[0].fill_between(kmf.survival_function_.index, 
                               kmf.confidence_interval_survival_function_.iloc[:, 0],
                               kmf.confidence_interval_survival_function_.iloc[:, 1],
                               alpha=0.3, color=colors[0], label='95% 신뢰구간')
            
            # 중앙 생존 시간 표시
            median_survival = kmf.median_survival_time_
            if not np.isnan(median_survival):
                axes[0].axvline(median_survival, color=colors[3], linestyle='--', 
                               linewidth=2, label=f'중앙 생존 시간: {median_survival:.1f}개월')
            
            axes[0].set_xlabel('생존 시간 (개월)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('생존 확률', fontsize=12, fontweight='bold')
            axes[0].set_title('🌍 전체 환자 생존 곡선', fontsize=14, fontweight='bold', pad=20)
            axes[0].legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1.05)
            
            # 생존 이벤트별 곡선
            event_0 = self.survival_aligned[self.survival_aligned['OS'] == 0]
            event_1 = self.survival_aligned[self.survival_aligned['OS'] == 1]
            
            if len(event_0) > 0:
                kmf_0 = KaplanMeierFitter()
                kmf_0.fit(event_0['OS.time'], event_0['OS'])
                axes[1].plot(kmf_0.survival_function_.index, kmf_0.survival_function_.values,
                           linewidth=3, color=colors[1], label=f'생존 (OS=0, n={len(event_0)})')
            
            if len(event_1) > 0:
                kmf_1 = KaplanMeierFitter()
                kmf_1.fit(event_1['OS.time'], event_1['OS'])
                axes[1].plot(kmf_1.survival_function_.index, kmf_1.survival_function_.values,
                           linewidth=3, color=colors[2], label=f'사망 (OS=1, n={len(event_1)})')
            
            # Log-rank test
            if len(event_0) > 0 and len(event_1) > 0:
                try:
                    logrank_result = logrank_test(event_0['OS.time'], event_1['OS.time'], 
                                               event_0['OS'], event_1['OS'])
                    axes[1].text(0.05, 0.95, f'Log-rank test\np = {logrank_result.p_value:.4f}', 
                               transform=axes[1].transAxes, fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                except:
                    pass
            
            axes[1].set_xlabel('생존 시간 (개월)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('생존 확률', fontsize=12, fontweight='bold')
            axes[1].set_title('⚡ 생존 이벤트별 생존 곡선', fontsize=14, fontweight='bold', pad=20)
            axes[1].legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1.05)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'kaplan_meier_survival_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Kaplan-Meier 생존 곡선 저장")
            
        except Exception as e:
            print(f"⚠️ Kaplan-Meier 곡선 생성 오류: {e}")
    
    def _plot_survival_prediction_performance(self, output_dir):
        """생존 예측 성능 비교"""
        if 'clinical_analysis' not in self.results:
            return
        
        clinical = self.results['clinical_analysis']
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('🎯 생존 예측 모델 성능 비교', fontsize=20, fontweight='bold', y=0.95)
        
        # 생존 이벤트 예측 AUC
        if 'survival_event_prediction' in clinical:
            event_pred = clinical['survival_event_prediction']
            auc_values = [event_pred.get('original_auc', 0), 
                        event_pred.get('train_imputed_auc', 0),
                        event_pred.get('valid_imputed_auc', 0)]
            auc_labels = ['원본 데이터', '훈련\nImputed', '검증\nImputed']
            
            bars = axes[0].bar(auc_labels, auc_values, color=colors[:3], alpha=0.8, 
                             edgecolor='white', linewidth=2)
            axes[0].set_xlabel('데이터 유형', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('AUC', fontsize=12, fontweight='bold')
            axes[0].set_title('💊 생존 이벤트 예측 성능 (AUC)', fontsize=14, fontweight='bold', pad=20)
            axes[0].set_ylim(0, 1.05)
            axes[0].grid(True, alpha=0.3)
            
            # AUC 값 표시
            for bar, value in zip(bars, auc_values):
                if value > 0:
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # 성능 향상도 표시
            if auc_values[0] > 0:
                for i, (bar, value) in enumerate(zip(bars[1:], auc_values[1:])):
                    if value > 0:
                        improvement = value - auc_values[0]
                        color = 'green' if improvement > 0 else 'red'
                        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                   f'Δ{improvement:+.3f}', ha='center', va='bottom', 
                                   color=color, fontweight='bold', fontsize=10)
        
        # 생존 시간 예측 R²
        if 'survival_time_prediction' in clinical:
            time_pred = clinical['survival_time_prediction']
            r2_values = [time_pred.get('original_r2', 0), 
                        time_pred.get('train_imputed_r2', 0),
                        time_pred.get('valid_imputed_r2', 0)]
            r2_labels = ['원본 데이터', '훈련\nImputed', '검증\nImputed']
            
            bars = axes[1].bar(r2_labels, r2_values, color=colors[:3], alpha=0.8, 
                             edgecolor='white', linewidth=2)
            axes[1].set_xlabel('데이터 유형', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('R²', fontsize=12, fontweight='bold')
            axes[1].set_title('⏰ 생존 시간 예측 성능 (R²)', fontsize=14, fontweight='bold', pad=20)
            axes[1].set_ylim(0, 1.05)
            axes[1].grid(True, alpha=0.3)
            
            # R² 값 표시
            for bar, value in zip(bars, r2_values):
                if value > 0:
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # 성능 향상도 표시
            if r2_values[0] > 0:
                for i, (bar, value) in enumerate(zip(bars[1:], r2_values[1:])):
                    if value > 0:
                        improvement = value - r2_values[0]
                        color = 'green' if improvement > 0 else 'red'
                        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                   f'Δ{improvement:+.3f}', ha='center', va='bottom', 
                                   color=color, fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'survival_prediction_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 생존 예측 성능 비교 저장")
    
    def _plot_survival_time_distributions(self, output_dir):
        """생존 시간 분포 비교"""
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('📊 생존 시간 분포 분석', fontsize=20, fontweight='bold', y=0.95)
        
        # 생존 시간 히스토그램
        n, bins, patches = axes[0].hist(self.survival_aligned['OS.time'], bins=40, alpha=0.7, 
                                       color=colors[0], edgecolor='white', linewidth=1, density=True)
        
        # 통계 정보 표시
        mean_time = self.survival_aligned['OS.time'].mean()
        median_time = self.survival_aligned['OS.time'].median()
        std_time = self.survival_aligned['OS.time'].std()
        
        axes[0].axvline(mean_time, color=colors[3], linestyle='--', linewidth=3, 
                       label=f'평균: {mean_time:.1f}개월')
        axes[0].axvline(median_time, color=colors[2], linestyle='--', linewidth=3, 
                       label=f'중앙값: {median_time:.1f}개월')
        
        # 정규분포 곡선 추가 (SciPy 없어도 그림은 그리도록)
        try:
            from scipy.stats import norm
            x = np.linspace(bins[0], bins[-1], 100)
            y = norm.pdf(x, mean_time, std_time)
            axes[0].plot(x, y, color=colors[1], linewidth=2, label='정규분포 근사')
        except Exception:
            pass  # SciPy 없어도 그림은 그리도록
        
        axes[0].set_xlabel('생존 시간 (개월)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('밀도', fontsize=12, fontweight='bold')
        axes[0].set_title('🌍 전체 생존 시간 분포', fontsize=14, fontweight='bold', pad=20)
        axes[0].legend(frameon=True, fancybox=True, shadow=True)
        axes[0].grid(True, alpha=0.3)
        
        # 생존 이벤트별 분포
        event_0 = self.survival_aligned[self.survival_aligned['OS'] == 0]['OS.time']
        event_1 = self.survival_aligned[self.survival_aligned['OS'] == 1]['OS.time']
        
        if len(event_0) > 0:
            axes[1].hist(event_0, bins=25, alpha=0.7, color=colors[1], 
                        label=f'생존 (OS=0, n={len(event_0)})', density=True, 
                        edgecolor='white', linewidth=1)
        if len(event_1) > 0:
            axes[1].hist(event_1, bins=25, alpha=0.7, color=colors[2], 
                        label=f'사망 (OS=1, n={len(event_1)})', density=True, 
                        edgecolor='white', linewidth=1)
        
        # 통계 정보 박스
        stats_text = f'생존 (OS=0):\n평균: {event_0.mean():.1f}개월\n중앙값: {event_0.median():.1f}개월\n\n사망 (OS=1):\n평균: {event_1.mean():.1f}개월\n중앙값: {event_1.median():.1f}개월'
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor="white", alpha=0.9))
        
        axes[1].set_xlabel('생존 시간 (개월)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('밀도', fontsize=12, fontweight='bold')
        axes[1].set_title('⚡ 생존 이벤트별 시간 분포', fontsize=14, fontweight='bold', pad=20)
        axes[1].legend(frameon=True, fancybox=True, shadow=True)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'survival_time_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 생존 시간 분포 비교 저장")
    
    def _plot_detailed_survival_analysis(self, output_dir):
        """생존 예측 모델 성능 상세 분석"""
        if 'clinical_analysis' not in self.results:
            return
        
        clinical = self.results['clinical_analysis']
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # 성능 향상도 분석
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('📈 생존 예측 모델 성능 향상도 분석', fontsize=20, fontweight='bold', y=0.95)
        
        # AUC 향상도
        if 'survival_event_prediction' in clinical:
            event_pred = clinical['survival_event_prediction']
            improvements = [
                event_pred.get('improvement_train', 0),
                event_pred.get('improvement_valid', 0)
            ]
            labels = ['훈련 데이터\n향상도', '검증 데이터\n향상도']
            colors_imp = ['#2ca02c', '#ff7f0e']
            
            bars = axes[0].bar(labels, improvements, color=colors_imp, alpha=0.8, 
                              edgecolor='white', linewidth=2)
            axes[0].set_ylabel('AUC 향상도', fontsize=12, fontweight='bold')
            axes[0].set_title('💊 생존 이벤트 예측 향상도', fontsize=14, fontweight='bold', pad=20)
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 향상도 값 표시
            for bar, value in zip(bars, improvements):
                if value is not None:
                    height = bar.get_height()
                    color = 'green' if value > 0 else 'red'
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:+.3f}', ha='center', va='bottom', 
                               color=color, fontweight='bold', fontsize=11)
        
        # R² 향상도
        if 'survival_time_prediction' in clinical:
            time_pred = clinical['survival_time_prediction']
            improvements = [
                time_pred.get('improvement_train', 0),
                time_pred.get('improvement_valid', 0)
            ]
            labels = ['훈련 데이터\n향상도', '검증 데이터\n향상도']
            colors_imp = ['#2ca02c', '#ff7f0e']
            
            bars = axes[1].bar(labels, improvements, color=colors_imp, alpha=0.8, 
                              edgecolor='white', linewidth=2)
            axes[1].set_ylabel('R² 향상도', fontsize=12, fontweight='bold')
            axes[1].set_title('⏰ 생존 시간 예측 향상도', fontsize=14, fontweight='bold', pad=20)
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 향상도 값 표시
            for bar, value in zip(bars, improvements):
                if value is not None:
                    height = bar.get_height()
                    color = 'green' if value > 0 else 'red'
                    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:+.3f}', ha='center', va='bottom', 
                               color=color, fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'survival_improvement_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 생존 예측 향상도 분석 저장")
    
    def _plot_subtype_survival_analysis(self, output_dir):
        """서브타입별 생존 분석 (PAM50 등)"""
        try:
            # PAM50 서브타입 정보가 있는지 확인
            if hasattr(self, 'pam50_aligned') and self.pam50_aligned is not None:
                self._plot_pam50_survival_analysis(output_dir)
            else:
                # 기본 서브타입 분석 (예: 생존 시간 기준)
                self._plot_time_based_subtype_analysis(output_dir)
        except Exception as e:
            print(f"⚠️ 서브타입 생존 분석 오류: {e}")
    
    def _plot_time_based_subtype_analysis(self, output_dir):
        """생존 시간 기준 서브타입 분석"""
        # 스타일 설정 (전역 설정 사용)
        plt.style.use(self.plot_config.get('style', 'seaborn-v0_8'))
        colors = self.plot_config.get('colors', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'])
        
        # 생존 시간을 기준으로 그룹 분류 (사분위수 기준)
        time_quartiles = self.survival_aligned['OS.time'].quantile([0.25, 0.5, 0.75])
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('🏷️ 생존 시간 기준 서브타입 분석', fontsize=20, fontweight='bold', y=0.95)
        
        # 서브타입별 생존 곡선
        KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
        kmf = KaplanMeierFitter()
        
        # 각 그룹별 생존 곡선
        groups = ['Q1 (빠른 진행)', 'Q2 (중간)', 'Q3 (중간)', 'Q4 (느린 진행)']
        group_colors = colors[:4]
        
        for i, (group_name, color) in enumerate(zip(groups, group_colors)):
            if i == 0:  # Q1
                mask = self.survival_aligned['OS.time'] <= time_quartiles[0.25]
            elif i == 1:  # Q2
                mask = (self.survival_aligned['OS.time'] > time_quartiles[0.25]) & (self.survival_aligned['OS.time'] <= time_quartiles[0.5])
            elif i == 2:  # Q3
                mask = (self.survival_aligned['OS.time'] > time_quartiles[0.5]) & (self.survival_aligned['OS.time'] <= time_quartiles[0.75])
            else:  # Q4
                mask = self.survival_aligned['OS.time'] > time_quartiles[0.75]
            
            group_data = self.survival_aligned[mask]
            if len(group_data) > 0:
                kmf.fit(group_data['OS.time'], group_data['OS'])
                axes[0].plot(kmf.survival_function_.index, kmf.survival_function_.values,
                           linewidth=2, color=color, label=f'{group_name} (n={len(group_data)})')
        
        axes[0].set_xlabel('생존 시간 (개월)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('생존 확률', fontsize=12, fontweight='bold')
        axes[0].set_title('📈 서브타입별 생존 곡선', fontsize=14, fontweight='bold', pad=20)
        axes[0].legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.05)
        
        # 서브타입별 중앙 생존 시간
        median_times = []
        for i, group_name in enumerate(groups):
            if i == 0:
                mask = self.survival_aligned['OS.time'] <= time_quartiles[0.25]
            elif i == 1:
                mask = (self.survival_aligned['OS.time'] > time_quartiles[0.25]) & (self.survival_aligned['OS.time'] <= time_quartiles[0.5])
            elif i == 2:
                mask = (self.survival_aligned['OS.time'] > time_quartiles[0.5]) & (self.survival_aligned['OS.time'] <= time_quartiles[0.75])
            else:
                mask = self.survival_aligned['OS.time'] > time_quartiles[0.75]
            
            group_data = self.survival_aligned[mask]
            if len(group_data) > 0:
                median_time = group_data['OS.time'].median()
                median_times.append(median_time)
            else:
                median_times.append(0)
        
        bars = axes[1].bar(groups, median_times, color=group_colors, alpha=0.8, 
                          edgecolor='white', linewidth=2)
        axes[1].set_xlabel('서브타입', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('중앙 생존 시간 (개월)', fontsize=12, fontweight='bold')
        axes[1].set_title('⏰ 서브타입별 중앙 생존 시간', fontsize=14, fontweight='bold', pad=20)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, median_times):
            if value > 0:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subtype_survival_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 서브타입별 생존 분석 저장")
    
    def _plot_tsne_visualization(self, output_dir):
        """t-SNE 시각화 (공동 임베딩)"""
        if 'dimensionality_analysis' not in self.results or \
           'tsne' not in self.results['dimensionality_analysis'] or \
           not self.results['dimensionality_analysis']['tsne']['available']:
            return
        
        try:
            tsne_data = self.results['dimensionality_analysis']['tsne']['coords']
            colors = self.plot_config.get('colors', ['#2E86AB', '#A23B72', '#F18F01'])
            labels = ['original', 'train_imputed', 'valid_imputed']
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            for i, label in enumerate(labels):
                mask = [l == label for l in tsne_data['label']]
                x = [tsne_data['x'][j] for j in range(len(tsne_data['x'])) if mask[j]]
                y = [tsne_data['y'][j] for j in range(len(tsne_data['y'])) if mask[j]]
                
                ax.scatter(x, y, c=colors[i], label=label, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
            ax.set_title('🔍 t-SNE 공동 임베딩 (원본 vs Imputed)', fontsize=16, fontweight='bold', pad=20)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tsne_embedding.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ t-SNE 시각화 저장")
            
        except Exception as e:
            print(f"⚠️ t-SNE 시각화 오류: {e}")
    
    def run_complete_analysis(self, data_paths: dict, output_dir: str):
        """엔드투엔드 실행 + 결과 저장"""
        print("🚀 향상된 다운스트림 분석 시작")
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # 데이터 로드
        self.load_data(data_paths)
        
        # (신규) 메타데이터 로드 + 정렬
        if data_paths.get('subtype') and data_paths.get('survival'):
            self.load_metadata(data_paths['subtype'], data_paths['survival'])
            self.restrict_to_imputed_samples(min_samples_per_subtype=2, strict_filtering=False)
        else:
            print("⚠️ subtype/survival 경로가 없어 메타데이터 정렬을 건너뜁니다.")
        
        # 0) 마스크 안정화(0/1 → bool)
        self.mask_aligned = (self.mask_aligned.astype(float) > 0).astype(int)
        print(f"✅ 마스크 안정화 완료: 1=결측, 0=관측")
        
        # 1) 기본 임퓨테이션 성능
        self.basic_imputation_analysis()
        
        # 2) 데이터 품질
        self.data_quality_analysis()
        
        # 3) 특징별 성능
        self.feature_level_analysis()
        
        # 4) 서브타입별 성능(있을 때만)
        self.subtype_performance_analysis()
        
        # 5) 차원 축소(공동 t-SNE)
        self.dimensionality_analysis()
        
        # 6) 생존분석 (새로 추가)
        if hasattr(self, 'survival_aligned') and self.survival_aligned is not None:
            print("\n🔍 6단계: 생존분석")
            self._plot_subtype_survival_analysis(output_dir)
            
            # Cox 분석 (Original 데이터)
            if hasattr(self, 'original_aligned'):
                print("📊 Original 데이터 Cox 분석")
                self.cox_eval(self.original_aligned, "original")
            
            # Cox 분석 (Imputed 데이터)
            if hasattr(self, 'train_aligned'):
                print("📊 Imputed 데이터 Cox 분석")
                self.cox_eval(self.train_aligned, "imputed")
        else:
            print("\n⚠️ 생존 데이터가 없어 생존분석을 건너뜁니다.")
        
        # 7) 임상적 관련성(있을 때만)
        self.clinical_relevance_analysis()
        
        # 7) 결과 저장
        with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        
        if 'feature_analysis' in self.results:
            pd.DataFrame(self.results['feature_analysis']['feature_performance']) \
              .to_csv(os.path.join(output_dir, "feature_performance.tsv"), sep='\t', index=False)
        
        print(f"✅ 결과 저장 완료: {output_dir}")
        
        # 8) 그림 생성
        self.generate_visualizations(output_dir)
        
        # 9) t-SNE 시각화 (별도)
        self._plot_tsne_visualization(output_dir)
        
        return self.results

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='향상된 다운스트림 분석')
    
    # 데이터 경로
    parser.add_argument('--train_imputed', type=str, required=True,
                       help='훈련 데이터 imputation 결과')
    parser.add_argument('--valid_imputed', type=str, required=True,
                       help='검증 데이터 imputation 결과')
    parser.add_argument('--original', type=str, required=True,
                       help='원본 데이터')
    parser.add_argument('--mask', type=str, required=True,
                       help='마스크 파일')
    parser.add_argument('--subtype', type=str, default=None,
                       help='서브타입 정보 (선택사항)')
    parser.add_argument('--survival', type=str, default=None,
                       help='생존 데이터 (TCGA-BRCA.survival.tsv)')
    parser.add_argument('--output_dir', type=str, default='./enhanced_analysis',
                       help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 데이터 경로 구성
    data_paths = {
        'train_imputed': args.train_imputed,
        'valid_imputed': args.valid_imputed,
        'original': args.original,
        'mask': args.mask,
        'subtype': args.subtype,
        'survival': args.survival
    }
    
    # 분석 실행
    analyzer = EnhancedDownstreamAnalysis({})
    results = analyzer.run_complete_analysis(data_paths, args.output_dir)
    
    print("\n📊 분석 결과 요약:")
    for key, value in results.items():
        if isinstance(value, dict) and 'performance' in str(value):
            print(f"  {key}: 성능 분석 완료")
        else:
            print(f"  {key}: 분석 완료")

if __name__ == '__main__':
    main()

# ========================================
# 🎯 새로운 생존분석 실행 함수 (단순 & 직접)
# ========================================

def run_survival_once(
    X: pd.DataFrame,
    survival: pd.DataFrame,
    *,
    label: str = "dataset",
    n_comp: int = 50,
    adjust_pam50: bool = False,
    split = 0.5,   # 'median' 또는 0~1 사이 분위수
    penalizer = None,
    l1_ratio = None,
    random_state: int = 42,
    min_samples: int = 20,
):
    """
    🎯 단일 생존분석 실행 함수
    
    입력: 
      - X: (샘플×특징) DataFrame, index는 샘플 ID
      - survival: index가 동일한 샘플 ID, 필수 컬럼 ['OS.time','OS'], (선택) 'PAM50'
    
    출력(dict):
      {
        'label', 'n_samples', 'c_index', 'logrank_p',
        'risk_scores'(pd.Series), 'risk_group'(pd.Series),
        'selected_pcs'(list), 'cox_model'(CoxPHFitter), 'X_pcs'(pd.DataFrame)
      }
    """
    # lifelines 지연 임포트
    KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
    
    # 1) 정렬/검증
    if not {"OS.time", "OS"}.issubset(survival.columns):
        raise ValueError("survival에는 'OS.time'과 'OS' 컬럼이 필요합니다.")
    
    keep = X.index.intersection(survival.index)
    if len(keep) < min_samples:
        raise ValueError(f"교집합 샘플 부족: {len(keep)} < {min_samples}")
    
    y = survival.loc[keep, ["OS.time", "OS"]].copy()
    if adjust_pam50 and "PAM50" in survival.columns:
        y["PAM50"] = survival.loc[keep, "PAM50"]

    # 이벤트 비율 체크(모두 0/1 방지)
    ev_rate = float(np.mean(y["OS"]))
    if ev_rate <= 0.0 or ev_rate >= 1.0:
        raise ValueError(f"이벤트 비율이 극단적입니다(={ev_rate:.3f}). Cox 적합이 불안정할 수 있습니다.")

    print(f"🔍 {label}: {len(keep)}개 샘플, 이벤트 비율 {ev_rate:.3f}")

    # 2) (지도형) PCA 특징 만들기
    X_sel, pc_names = _supervised_pca_for_cox(X.loc[keep], y, max_k=n_comp, random_state=random_state)
    print(f"✅ 선택된 PC: {pc_names}")

    # 3) PAM50 보정(선택)
    df = X_sel.join(y[["OS.time", "OS"]])
    if adjust_pam50 and "PAM50" in y.columns:
        dummies = pd.get_dummies(y["PAM50"], prefix="PAM50")
        df = df.join(dummies)
        print(f"🔧 PAM50 더미 변수 추가: {list(dummies.columns)}")

    # 4) Cox 적합(튜닝 또는 고정 하이퍼)
    if penalizer is None or l1_ratio is None:
        print("🔧 Cox 하이퍼파라미터 자동 튜닝 중...")
        pen, l1 = _tune_cox(df)
        print(f"✅ 최적 하이퍼: penalizer={pen:.2f}, l1_ratio={l1:.2f}")
    else:
        pen, l1 = penalizer, l1_ratio
        print(f"🔧 고정 하이퍼: penalizer={pen:.2f}, l1_ratio={l1:.2f}")
    
    cph = CoxPHFitter(penalizer=pen, l1_ratio=l1)
    cph.fit(df, duration_col="OS.time", event_col="OS")
    cindex = float(cph.concordance_index_)

    # 5) 위험점수/그룹 & 로그랭크
    risk = cph.predict_partial_hazard(df).rename("risk").astype(float)
    if isinstance(split, (int, float)) and 0 < float(split) < 1:
        thr = float(risk.quantile(float(split)))
    else:
        thr = float(risk.median())
    
    group = pd.Series(np.where(risk > thr, "High", "Low"), index=risk.index, name="risk_group")

    y_low = y.loc[group == "Low"]
    y_high = y.loc[group == "High"]
    
    if len(y_low) >= 3 and len(y_high) >= 3:
        pval = float(logrank_test(y_low["OS.time"], y_high["OS.time"], y_low["OS"], y_high["OS"]).p_value)
        print(f"✅ 로그랭크 테스트: Low(n={len(y_low)}) vs High(n={len(y_high)})")
    else:
        pval = None
        print(f"⚠️ 로그랭크 테스트 불가: Low(n={len(y_low)}), High(n={len(y_high)})")

    km_blob = {
        "low":  {"t": y_low["OS.time"].to_numpy(),  "e": y_low["OS"].to_numpy(),  "n": int(len(y_low))},
        "high": {"t": y_high["OS.time"].to_numpy(), "e": y_high["OS"].to_numpy(), "n": int(len(y_high))}
    }

    print(f"🎯 {label} 완료: C-index={cindex:.4f}, Log-rank p={pval:.3e}" if pval else f"🎯 {label} 완료: C-index={cindex:.4f}")

    return {
        "label": label,
        "n_samples": int(len(keep)),
        "c_index": cindex,
        "logrank_p": pval,
        "risk_scores": risk,       # pd.Series (index=샘플)
        "risk_group": group,       # pd.Series (Low/High)
        "selected_pcs": pc_names,  # 사용된 PC 목록
        "cox_model": cph,          # lifelines 모델(원하면 저장 전 제거)
        "X_pcs": X_sel,            # 선택된 PC 행렬
        "km_blob": km_blob,        # KM 그리기용 원시값
    }


def _supervised_pca_for_cox(X: pd.DataFrame, y: pd.DataFrame, max_k=50, random_state=42):
    """
    표준화 → PCA → 각 PC 단변량 Cox p-value로 중요도 평가 → 상위 PC 선택
    반환: (선택된 PC DataFrame, 선택한 컬럼 목록)
    """
    # 표준화
    sc = StandardScaler()
    Xs = sc.fit_transform(X.values)
    
    # 과적합 방지: 표본·변수 수에 기반하여 k 제한
    k0 = max(1, min(max_k, Xs.shape[1], Xs.shape[0]-2))
    pca = PCA(n_components=k0, random_state=random_state)
    Z = pca.fit_transform(Xs)
    Zdf = pd.DataFrame(Z, index=X.index, columns=[f"PC{i+1}" for i in range(Z.shape[1])])

    # 각 PC 단변량 Cox p-value
    pvals = []
    for c in Zdf.columns:
        df1 = Zdf[[c]].join(y[["OS.time", "OS"]])
        cph1 = CoxPHFitter(penalizer=0.05)
        try:
            cph1.fit(df1, duration_col="OS.time", event_col="OS")
            p = float(cph1.summary.loc[c, "p"])
        except Exception:
            p = 1.0
        pvals.append((c, p))
    pvals.sort(key=lambda x: x[1])

    # 분산/표본 기반 상한 + p값 상위 일부 선택
    cum = np.cumsum(pca.explained_variance_ratio_)
    k_var = int(np.argmax(cum >= 0.85) + 1) if len(cum) else 1
    k_cap = max(5, min(len(pvals), X.shape[0] // 4))
    k_final = max(3, min(k_var, k_cap, 15))
    top_cols = [c for c, _ in pvals[:k_final]]
    
    return Zdf[top_cols], top_cols


def _tune_cox(df: pd.DataFrame, n_splits=5, random_state=42):
    """
    Cox 페널티 간단 튜닝 (C-index 최대화).
    작은 데이터 방어: 분할 수 자동 축소(최소 3).
    """
    n = len(df)
    k = max(3, min(n_splits, n-1, (n//6 if n >= 36 else 3)))
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    grid_pen = (0.05, 0.1, 0.2, 0.3)
    grid_l1 = (0.3, 0.5, 0.7)

    best = (-1.0, (0.1, 0.5))
    KaplanMeierFitter, CoxPHFitter, logrank_test, concordance_index = _safe_import_lifelines()
    
    for pen in grid_pen:
        for l1 in grid_l1:
            scores = []
            for tr, va in kf.split(df):
                tr_df, va_df = df.iloc[tr], df.iloc[va]
                cph = CoxPHFitter(penalizer=pen, l1_ratio=l1)
                try:
                    cph.fit(tr_df, duration_col="OS.time", event_col="OS")
                    risk = cph.predict_partial_hazard(va_df).values.ravel()
                    c = concordance_index(va_df["OS.time"], risk, va_df["OS"])
                    scores.append(c)
                except Exception:
                    scores.append(0.5)
            m = float(np.mean(scores))
            if m > best[0]: 
                best = (m, (pen, l1))
    
    return best[1]


def run_multimodal_survival(
    X_dict: dict,
    survival: pd.DataFrame,
    *,
    adjust_pam50: bool = False,
    n_comp_per_omics: int = 50,
    **kwargs
):
    """
    🚀 멀티모달 생존분석 실행 (RNA, Protein, Methyl, Ensemble)
    
    입력:
      - X_dict: {'rna': X_rna, 'protein': X_protein, 'methyl': X_methyl}
      - survival: 생존 데이터
      - adjust_pam50: PAM50 보정 여부
      - n_comp_per_omics: 각 모달리티별 PC 수
    
    출력:
      - 각 모달리티별 결과 + Ensemble 결과
    """
    results = {}
    
    # 1. 각 모달리티별 개별 분석
    for modality, X in X_dict.items():
        if X is not None and not X.empty:
            print(f"\n{'='*50}")
            print(f"🔍 {modality.upper()} 모달리티 분석 시작")
            print(f"{'='*50}")
            
            try:
                res = run_survival_once(
                    X, survival, 
                    label=modality.upper(),
                    n_comp=n_comp_per_omics,
                    adjust_pam50=adjust_pam50,
                    **kwargs
                )
                results[modality] = res
            except Exception as e:
                print(f"❌ {modality} 분석 실패: {e}")
                results[modality] = None
    
    # 2. Ensemble 분석 (모든 모달리티 통합)
    print(f"\n{'='*50}")
    print(f"🚀 ENSEMBLE 모달리티 통합 분석 시작")
    print(f"{'='*50}")
    
    try:
        # 공통 샘플 찾기
        common_samples = None
        for modality, res in results.items():
            if res is not None:
                if common_samples is None:
                    common_samples = set(res['X_pcs'].index)
                else:
                    common_samples = common_samples.intersection(set(res['X_pcs'].index))
        
        if common_samples and len(common_samples) >= 20:
            # 모든 모달리티의 PC를 통합
            ensemble_features = []
            for modality, res in results.items():
                if res is not None:
                    X_pcs = res['X_pcs'].loc[list(common_samples)]
                    ensemble_features.append(X_pcs)
            
            if ensemble_features:
                X_ensemble = pd.concat(ensemble_features, axis=1)
                print(f"✅ Ensemble 데이터: {X_ensemble.shape}")
                
                res_ensemble = run_survival_once(
                    X_ensemble, survival,
                    label="ENSEMBLE",
                    n_comp=min(50, X_ensemble.shape[1]),
                    adjust_pam50=adjust_pam50,
                    **kwargs
                )
                results['ensemble'] = res_ensemble
            else:
                print("⚠️ Ensemble 분석 불가: PC 데이터 없음")
                results['ensemble'] = None
        else:
            print(f"⚠️ Ensemble 분석 불가: 공통 샘플 부족 ({len(common_samples) if common_samples else 0})")
            results['ensemble'] = None
            
    except Exception as e:
        print(f"❌ Ensemble 분석 실패: {e}")
        results['ensemble'] = None
    
    return results
