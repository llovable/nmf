"""
새로운 다운스트림 분석 모듈
하나의 데이터셋(3개 모달리티)만 선택해서 통합하고 생존분석 결과를 반환
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

# NEW: 타입힌트 안전 & 의존성 사전 로드
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================================
# NEW: 유틸 함수들 추가
# ============================================================================

def _robust_status_to_int(x) -> int:
    """다양한 표기(0/1, Alive/Dead, True/False 등) → 0/1 표준화"""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    true_set  = {"1","true","yes","y","dead","deceased","event","event_occurred"}
    false_set = {"0","false","no","n","alive","censored","none","event_free"}
    if s in true_set:
        return 1
    if s in false_set:
        return 0
    # 숫자처럼 보이면 숫자 비교
    try:
        return 1 if float(s) == 1.0 else 0
    except:
        return 0

def _maybe_transpose_to_samples_by_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    행=샘플, 열=특징 형태로 정렬.
    일반적으로 특징 수(열)가 샘플 수(행)보다 훨씬 큼 → 행이 더 많으면 전치.
    """
    if df.shape[0] > df.shape[1]:  # 행(현재)이 열보다 많으면 특징이 행으로 있을 가능성 큼
        return df.T
    return df

def _find_modality_files(dataset_path: Path, variant: str) -> Dict[str, Path]:
    """
    variant 폴더(예: imputation_missorigin, imputation_miss10p ...) 아래에서
    rna/protein/methyl 파일 경로를 찾아 반환.
    """
    variant_dir = dataset_path / f"imputation_{variant}"
    if not variant_dir.exists():
        raise FileNotFoundError(f"변형 폴더 없음: {variant_dir}")

    # 파일 패턴들(이름이 다소 달라도 대응)
    patterns = {
        "rna":     ["rna_full_*.tsv", "*rna*full*.tsv", "*rna*.tsv"],
        "protein": ["protein_full_*.tsv", "*protein*full*.tsv", "*protein*.tsv"],
        "methyl":  ["methyl_full_*.tsv", "*methyl*full*.tsv", "*methy*.tsv"],
    }

    picked = {}
    for mod, pats in patterns.items():
        found = []
        for p in pats:
            found += list(variant_dir.glob(p))
        found = sorted(set(found))
        if found:
            picked[mod] = found[0]  # 첫 번째 매칭 사용
    if len(picked) < 2:
        raise RuntimeError(f"{variant_dir}에서 모달리티 최소 2개 파일을 못 찾음: {picked}")
    return picked

def _generate_subtitle_from_variant(variant: str) -> str:
    """
    variant 이름을 기반으로 서브타이틀을 자동 생성
    
    Args:
        variant: variant 이름 (예: "missorigin", "miss10p", "miss30p", "miss50p")
        
    Returns:
        서브타이틀 문자열
    """
    if variant == "missorigin":
        return "Original Imputation Data"
    elif variant.startswith("miss"):
        # miss10p, miss30p, miss50p 등에서 숫자 추출
        try:
            percentage = variant.replace("miss", "").replace("p", "")
            return f"Noise {percentage}% Imputation Data"
        except:
            return f"{variant.replace('miss', 'Missing ').replace('p', '%')} Data"
    else:
        return f"{variant.replace('_', ' ').title()} Data"

# ============================================================================
# 0. 메타데이터 가공 함수 (보강)
# ============================================================================

def create_metadata(pam50_file: str, survival_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """PAM50 + 생존 메타데이터 생성 및 통합"""
    print("🔧 메타데이터 생성 중...")

    # PAM50 로드
    pam = pd.read_csv(pam50_file, sep="\t")
    sid = [c for c in pam.columns if c.lower() in ["sample", "id", "barcode"]][0]
    sub = [c for c in pam.columns if c.lower() in ["pam50", "subtype"]][0]
    pam_processed = pam[[sid, sub]].rename(columns={sid: "sample", sub: "PAM50"})

    # 생존 로드
    surv = pd.read_csv(survival_file, sep="\t")
    # 샘플 식별자 유추: 첫 컬럼이 샘플이면 그걸로
    if "sample" in [c.lower() for c in surv.columns]:
        # 이미 sample 컬럼 있는 경우
        pass
    elif surv.columns[0].lower() in {"sample","id","barcode"}:
        surv = surv.rename(columns={surv.columns[0]: "sample"})
    else:
        # index 기반이면
        surv = pd.read_csv(survival_file, sep="\t", index_col=0).reset_index().rename(columns={"index":"sample"})

    # 시간/사건 컬럼 자동 탐색
    time_col = [c for c in surv.columns if any(k in c.lower() for k in ["os_time", "os.time", "time", "months", "followup"])]
    status_col = [c for c in surv.columns if any(k in c.lower() for k in ["os", "status", "event", "death"])]

    if not time_col or not status_col:
        raise ValueError(f"생존 컬럼 탐색 실패. columns={surv.columns.tolist()}")

    time_col = time_col[0]
    status_col = status_col[0]

    surv_processed = surv.rename(columns={time_col: "OS.time", status_col: "OS"})[["sample","OS.time","OS"]]

    # 단위 보정(일 → 개월 heuristic)
    if pd.to_numeric(surv_processed["OS.time"], errors="coerce").max() > 1000:
        surv_processed["OS.time"] = pd.to_numeric(surv_processed["OS.time"], errors="coerce") / 30.44

    # 상태 0/1 표준화
    surv_processed["OS"] = surv_processed["OS"].map(_robust_status_to_int)

    # 통합
    meta = (
        pd.merge(pam_processed, surv_processed, on="sample", how="inner")
        .dropna(subset=["OS.time","OS"])  # Cox 요구사항
        .set_index("sample")
    )

    print(f"✅ 메타데이터 준비 완료: {meta.shape}")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(output_file, sep="\t")

    return meta

# ============================================================================
# 1. 하나의 데이터셋 선택 및 멀티오믹스 통합 함수 (보강)
# ============================================================================

def integrate_single_dataset(
    data_dir: str,
    dataset_name: str,
    output_file: Optional[str] = None,
    n_comp_per_omics: int = 50,
    variant: str = "missorigin",            # NEW: missorigin/miss10p/miss30p/miss50p
    transpose_if_needed: bool = True        # NEW
) -> pd.DataFrame:
    """
    하나의 데이터셋(3개 모달리티)만 선택해서 통합
    - data_dir/dataset_name/imputation_{variant}/ 아래에서 rna/protein/methyl 탐색
    """
    print(f"🔬 데이터셋 '{dataset_name}'[{variant}] 3모달리티 통합 시작")

    dataset_path = Path(data_dir) / dataset_name
    if not dataset_path.exists():
        raise ValueError(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")
    print(f"   선택된 경로: {dataset_path}")

    file_map = _find_modality_files(dataset_path, variant=variant)  # NEW

    loaded_data: Dict[str, pd.DataFrame] = {}
    for modality, fpath in file_map.items():
        df = pd.read_csv(fpath, sep="\t", index_col=0)
        if transpose_if_needed:
            df = _maybe_transpose_to_samples_by_rows(df)
        loaded_data[modality] = df
        print(f"   ✅ {modality}: {df.shape}  ← {fpath.name}")

    if len(loaded_data) < 2:
        raise ValueError(f"최소 2개 모달리티 필요. 로드된 모달리티: {list(loaded_data.keys())}")

    # 공통 샘플
    common_samples = set.intersection(*[set(df.index) for df in loaded_data.values()])
    print(f"   공통 샘플 수: {len(common_samples)}")

    processed_data = []
    for modality, df in loaded_data.items():
        d = df.loc[list(common_samples)].copy()
        d = pd.DataFrame(StandardScaler().fit_transform(d), index=d.index, columns=d.columns)
        if n_comp_per_omics > 0:
            n_comp = min(n_comp_per_omics, max(1, d.shape[1]-1))
            if n_comp > 0:
                pca = PCA(n_components=n_comp, random_state=42)
                data_pca = pca.fit_transform(d)
                d = pd.DataFrame(
                    data_pca, index=d.index,
                    columns=[f"{modality}_PC{i+1}" for i in range(n_comp)]
                )
                print(f"     {modality} PCA: {d.shape[1]} → {n_comp}")
            processed_data.append(d)
        else:
            processed_data.append(d.add_prefix(f"{modality}_"))

    integrated = pd.concat(processed_data, axis=1).sort_index()

    print(f"✅ 통합 완료: {integrated.shape}")
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        integrated.to_csv(output_file, sep="\t")
        print(f"   저장 완료: {output_file}")

    return integrated

# ============================================================================
# 2. 생존분석 모델 생성 함수 (경미한 정리)
# ============================================================================

def run_survival_analysis(X: pd.DataFrame, metadata: pd.DataFrame, label: str = "survival") -> Dict:
    print("💊 생존분석 모델 생성 시작")
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    common = X.index.intersection(metadata.index)
    X_aligned = X.loc[common]
    M = metadata.loc[common, ["PAM50","OS.time","OS"]].copy()

    # PAM50 보정(dummy)
    pam_dum = pd.get_dummies(M["PAM50"], prefix="PAM50", drop_first=False)
    ana = pd.concat([X_aligned, pam_dum, M[["OS.time","OS"]]], axis=1)

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph.fit(ana, duration_col="OS.time", event_col="OS")
    c_index = float(cph.concordance_index_)

    risk = cph.predict_partial_hazard(ana)
    med = risk.median()
    groups = pd.Series(np.where(risk > med, "High", "Low"), index=risk.index, name="risk_group")

    low = ana[groups == "Low"]
    high = ana[groups == "High"]
    lr = logrank_test(low["OS.time"], high["OS.time"], low["OS"], high["OS"])

    # 3. PAM50 서브타입별 생존분석
    pam50_results = {}
    print(f"   PAM50 서브타입별 분석 시작...")
    for subtype in M["PAM50"].unique():
        if subtype != "Unknown":
            subtype_data = ana[M["PAM50"] == subtype]
            print(f"     {subtype}: {len(subtype_data)} 샘플")
            
            # 최소 샘플 수 체크 (10 → 5로 낮춤)
            if len(subtype_data) >= 5:
                # 생존 이벤트 수 체크 (최소 1개 이벤트로 완화)
                n_events = subtype_data["OS"].sum()
                if n_events >= 1:
                    try:
                        subtype_cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
                        subtype_cph.fit(subtype_data, duration_col="OS.time", event_col="OS")
                        pam50_results[subtype] = {
                            "c_index": float(subtype_cph.concordance_index_),
                            "n_samples": len(subtype_data),
                            "n_events": int(n_events),
                            "data": subtype_data,
                            "analysis_type": "cox_analysis"
                        }
                        print(f"       ✅ {subtype} Cox 분석 성공: C-index = {float(subtype_cph.concordance_index_):.4f}, 이벤트 = {int(n_events)}")
                    except Exception as e:
                        # Cox 분석 실패 시 생존 시간 분포 분석
                        pam50_results[subtype] = {
                            "error": f"Cox 분석 실패: {str(e)}",
                            "n_samples": len(subtype_data),
                            "n_events": int(n_events),
                            "survival_stats": {
                                "mean_time": float(subtype_data["OS.time"].mean()),
                                "median_time": float(subtype_data["OS.time"].median()),
                                "std_time": float(subtype_data["OS.time"].std()),
                                "min_time": float(subtype_data["OS.time"].min()),
                                "max_time": float(subtype_data["OS.time"].max())
                            },
                            "analysis_type": "survival_stats_only"
                        }
                        print(f"       ⚠️ {subtype} Cox 분석 실패, 생존 시간 통계만 제공: {str(e)}")
                else:
                    # 이벤트가 없는 경우 생존 시간 분포 분석
                    pam50_results[subtype] = {
                        "error": f"생존 이벤트 없음: {int(n_events)} < 1",
                        "n_samples": len(subtype_data),
                        "n_events": int(n_events),
                        "survival_stats": {
                            "mean_time": float(subtype_data["OS.time"].mean()),
                            "median_time": float(subtype_data["OS.time"].median()),
                            "std_time": float(subtype_data["OS.time"].std()),
                            "min_time": float(subtype_data["OS.time"].min()),
                            "max_time": float(subtype_data["OS.time"].max())
                        },
                        "analysis_type": "survival_stats_only"
                    }
                    print(f"       ℹ️ {subtype} 이벤트 없음, 생존 시간 통계 제공: 평균 = {float(subtype_data['OS.time'].mean()):.1f}개월")
            else:
                pam50_results[subtype] = {"error": f"샘플 수 부족: {len(subtype_data)} < 5"}
                print(f"       ⚠️ {subtype} 샘플 수 부족: {len(subtype_data)} < 5")

    print(f"✅ 생존분석 완료: C-index = {c_index:.4f}, Log-rank p = {float(lr.p_value):.3e}")
    return {
        "label": label,
        "c_index": c_index,
        "n_samples": len(ana),
        "risk_scores": risk,
        "risk_groups": groups,
        "logrank_p": float(lr.p_value),
        "low_risk_samples": len(low),
        "high_risk_samples": len(high),
        "cox_model": cph,
        "analysis_data": ana,
        "pam50_subtype_analysis": pam50_results  # NEW: PAM50 서브타입별 결과
    }

# ============================================================================
# 3. 카플란 그래프 그리기 함수 (타입힌트 안전)
# ============================================================================

def create_kaplan_plot(
    data: pd.DataFrame,
    risk_groups: Optional[Union[pd.Series, Dict[str, Union[int, str]]]] = None,
    *,
    plot_type: str = "risk_group",              # "risk_group" | "pam50_subtype"
    title: str = "",
    subtitle: str = "",                         # ✅ 서브타이틀 추가
    save_path: Optional[str] = None,

    # --- 컬럼 지정 ---
    time_col: str = "OS.time",
    event_col: str = "OS",
    group_col: Optional[str] = None,            # PAM50 서브타입 컬럼명 (plot_type="pam50_subtype"일 때)

    # --- 미학 옵션 ---
    colors: Optional[List[str]] = None,
    line_styles: Optional[List[str]] = None,
    line_widths: Optional[List[float]] = None,
    alpha: float = 0.95,
    background_color: str = "#FFFFFF",
    title_color: str = "#111111",
    subtitle_color: str = "#666666",            # ✅ 서브타이틀 색상
    font_size_title: int = 22,
    font_size_subtitle: int = 16,               # ✅ 서브타이틀 폰트 크기
    font_size_labels: int = 16,
    font_size_legend: int = 14,
    legend_location: str = "best",
    legend_ncol: int = 1,
    show_grid: bool = True,
    grid_alpha: float = 0.35,
    grid_color: str = "#D0D0D0",
    show_spines: bool = False,
    ci_show: bool = True,                        # ✅ 신뢰구간 음영
    ci_alpha: float = 0.25,
    annotate_stats: bool = True,                 # ✅ HR & p-value 주석
    time_unit: str = "months",                   # "months" | "days"
    dpi: int = 500,
    figsize: Tuple[int, int] = (12, 9),
):
    """
    data: time/event(0/1)와 그룹 정보가 들어있는 데이터프레임 (index는 개체 ID)
    risk_groups: 위험그룹 시리즈(0/1 또는 "low"/"high") 또는 dict(index->group)
    """
    from typing import Optional, List, Tuple, Dict, Union
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    try:
        from lifelines.statistics import multivariate_logrank_test
    except Exception:
        multivariate_logrank_test = None  # lifelines 버전에 따라 없을 수도 있음

    assert time_col in data.columns and event_col in data.columns, \
        f"'{time_col}', '{event_col}' 컬럼이 필요합니다."

    # 기본 팔레트/스타일
    if colors is None:
        if plot_type == "risk_group":
            colors = ["#4ECDC4", "#FF6B6B"]  # Low, High
        else:
            colors = ["#FF6B6B", "#F7B267", "#45B7D1", "#96CEB4", "#9B59B6"]
    if line_styles is None:
        line_styles = ["-", "--", "-.", ":", "-"]
    if line_widths is None:
        line_widths = [3.2] * len(colors)

    # 준비
    T = data[time_col].astype(float).values
    E = data[event_col].astype(int).values

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)

    # 축/그리드
    if show_grid:
        ax.grid(ls="--", alpha=grid_alpha, color=grid_color)
    if not show_spines:
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    # ===== 위험그룹 (이진) =====
    if plot_type == "risk_group":
        # 그룹 시리즈 정리
        if isinstance(risk_groups, dict):
            Gs = pd.Series(risk_groups).reindex(data.index)
        else:
            Gs = pd.Series(risk_groups, index=data.index) if risk_groups is not None else None
        assert Gs is not None, "risk_groups 가 필요합니다."
        g = Gs.astype(str).str.lower().replace({"low": "0", "high": "1"})
        g = g.astype(float).astype(int).values  # 0/1로 정규화

        # 표식·범례용 n
        n_low, n_high = int((g == 0).sum()), int((g == 1).sum())
        labels = [f"Low (n={n_low})", f"High (n={n_high})"]

        # KMF
        km_low, km_high = KaplanMeierFitter(), KaplanMeierFitter()
        km_low.fit(T[g == 0], E[g == 0], label=labels[0]).plot(
            ax=ax, ci_show=ci_show, ci_alpha=ci_alpha,
            lw=line_widths[0], ls=line_styles[0], color=colors[0], alpha=alpha
        )
        km_high.fit(T[g == 1], E[g == 1], label=labels[1]).plot(
            ax=ax, ci_show=ci_show, ci_alpha=ci_alpha,
            lw=line_widths[1], ls=line_styles[1], color=colors[1], alpha=alpha
        )

        # 통계 주석 (log-rank, Cox HR)
        if annotate_stats:
            try:
                p_lr = logrank_test(T[g == 0], T[g == 1], E[g == 0], E[g == 1]).p_value
            except Exception:
                p_lr = np.nan

            hr_txt = ""
            try:
                df_cox = pd.DataFrame({time_col: T, event_col: E, "group": g})
                cph = CoxPHFitter()
                cph.fit(df_cox, duration_col=time_col, event_col=event_col)
                hr = float(np.exp(cph.params_["group"]))
                ci = cph.confidence_intervals_.loc["group"].values
                ci_lower, ci_upper = float(np.exp(ci[0])), float(np.exp(ci[1]))
                hr_txt = f"HR = {hr:.2f}  [95% CI: {ci_lower:.2f} – {ci_upper:.2f}]"
            except Exception:
                hr_txt = "HR = NA  [95% CI: NA – NA]"

            txt = f"{hr_txt}\nlog-rank p = {p_lr:.4f}" if not np.isnan(p_lr) else hr_txt
            ax.text(
                0.03, 0.20, txt, transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
            )

    # ===== PAM50 서브타입 (다군) =====
    elif plot_type == "pam50_subtype":
        assert group_col is not None and group_col in data.columns, \
            "PAM50 서브타입 컬럼명(group_col)을 지정하세요."
        subtypes = data[group_col].dropna().astype(str)
        present = subtypes.unique().tolist()

        # 보기 좋은 순서로 정렬
        order = [s for s in ["Basal", "Her2", "LumA", "LumB", "Normal-like"] if s in present]
        if not order:  # 그래도 없으면 등장 순서대로
            order = present

        # 곡선 그리기
        for i, s in enumerate(order):
            mask = (subtypes.values == s)
            n_s = int(mask.sum())
            if n_s == 0:
                continue
            km = KaplanMeierFitter()
            km.fit(T[mask], E[mask], label=f"{s} (n={n_s})").plot(
                ax=ax, ci_show=ci_show, ci_alpha=ci_alpha,
                lw=line_widths[i % len(line_widths)],
                ls=line_styles[i % len(line_styles)],
                color=colors[i % len(colors)],
                alpha=alpha
            )

        # 전체 log-rank p (다군)
        if annotate_stats:
            p_txt = ""
            try:
                if multivariate_logrank_test is not None:
                    res = multivariate_logrank_test(T, groups=subtypes.values, event_observed=E)
                    p_txt = f"global log-rank p = {res.p_value:.4f}"
                else:
                    p_txt = "(global log-rank unavailable in this lifelines version)"
            except Exception:
                p_txt = "(log-rank failed)"
            ax.text(
                0.03, 0.20, p_txt, transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
            )

    else:
        raise ValueError("plot_type 은 'risk_group' 또는 'pam50_subtype' 이어야 합니다.")

    # 라벨/제목/범례
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Time ({'Months' if time_unit.lower().startswith('m') else 'Days'})",
                  fontsize=font_size_labels)
    ax.set_ylabel("Survival Probability", fontsize=font_size_labels)
    
    # --- 타이틀 / 서브타이틀 (겹침 방지) ---
    title_y = 0.95        # 제목은 그대로
    subtitle_y = 0.905     # 부제목을 제목 아래 훨씬 더 많이 내려서
    
    if title:
        fig.suptitle(title, fontsize=font_size_title, color=title_color, y=title_y, weight='bold')
    
    if subtitle:  # ← ax.text 대신 fig.text 사용!
        fig.text(0.5, subtitle_y, subtitle,
                 ha="center", va="top",
                 fontsize=font_size_subtitle, color=subtitle_color,
                 transform=fig.transFigure)

    leg = ax.legend(loc=legend_location, fontsize=font_size_legend, frameon=True)
    if leg is not None and leg.get_frame() is not None:
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor("#BDBDBD")
        # 그림자 효과 제거
        leg.get_frame().set_boxstyle("round,pad=0.5")
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#CCCCCC")
        leg.get_frame().set_linewidth(1.0)

    # suptitle/부제 공간만큼 위 여백 예약 (여백 최소화)
    top_margin = 0.95 if subtitle else 0.97 if title else 1.00
    plt.tight_layout(rect=[0, 0, 1, top_margin])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"✅ Graph saved: {save_path}")
    
    print("✅ Survival curve generation completed!")
    return fig

# ============================================================================
# 4. 하나의 데이터셋 분석 실행 함수
# ============================================================================

def analyze_single_dataset(pam50_file: str, 
                         survival_file: str, 
                         multiomics_data_dir: str,
                         dataset_name: str,
                         output_base: str = "/home/dyan/nmf/analysis/downstream",
                         analysis_label: Optional[str] = None,
                         n_comp_per_omics: int = 50,
                         variant: str = "missorigin") -> Dict:  # NEW: variant 인자 추가
    """
    하나의 데이터셋만 선택해서 통합하고 생존분석 실행
    
    Args:
        pam50_file: PAM50 데이터 파일 경로
        survival_file: 생존 데이터 파일 경로
        multiomics_data_dir: 멀티오믹스 데이터 루트 디렉토리
        dataset_name: 선택할 데이터셋 이름
        output_base: 결과 저장 기본 디렉토리
        analysis_label: 분석 라벨 (None이면 dataset_name 사용)
        n_comp_per_omics: 각 모달리티당 PCA 컴포넌트 수
        variant: 변형 이름 (missorigin/miss10p/miss30p/miss50p)
        
    Returns:
        생존분석 결과 (카플란 그래프용 데이터 포함)
    """
    if analysis_label is None:
        analysis_label = f"{dataset_name}_{variant}"
    
    print(f"🚀 데이터셋 '{dataset_name}'[{variant}] 분석 시작")
    print("=" * 60)
    
    try:
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 1. 메타데이터 처리
        print("\n📊 1단계: 메타데이터 처리")
        metadata = create_metadata(
            pam50_file, survival_file,
            output_base / f"{analysis_label}_metadata.tsv"
        )
        
        # 2. 선택된 데이터셋 통합
        print(f"\n🔬 2단계: 데이터셋 '{dataset_name}'[{variant}] 통합")
        integrated_data = integrate_single_dataset(
            multiomics_data_dir,
            dataset_name,
            output_base / f"{analysis_label}_integrated_multiomics.tsv",
            n_comp_per_omics,
            variant=variant  # NEW: variant 전달
        )
        
        # 3. 생존분석
        print("\n💊 3단계: 생존분석 모델 생성")
        survival_results = run_survival_analysis(integrated_data, metadata, analysis_label)
        
        if survival_results is None:
            raise ValueError("생존분석 실패")
        
        # 4. 결과 요약
        _print_analysis_summary(survival_results, analysis_label)
        
        print(f"\n🎉 데이터셋 '{dataset_name}'[{variant}] 분석 완료!")
        print("📝 이제 survival_results를 활용해서 카플란 그래프를 그릴 수 있습니다!")
        
        return survival_results
        
    except Exception as e:
        print(f"\n❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def _print_analysis_summary(results: Dict, label: str):
    """분석 결과 요약 출력"""
    print("\n" + "="*60)
    print(f"📊 데이터셋 '{label}' 분석 결과 요약")
    print("="*60)
    
    print(f"🔖 분석 라벨: {results['label']}")
    print(f"📊 샘플 수: {results['n_samples']}")
    
    print(f"\n💊 생존분석 결과:")
    print(f"   Cox C-index: {results['c_index']:.4f}")
    print(f"   Log-rank p-value: {results['logrank_p']:.3e}")
    print(f"   위험그룹: Low({results['low_risk_samples']}), High({results['high_risk_samples']})")
    
    print(f"\n📈 카플란 그래프 그리기:")
    print(f"   create_kaplan_plot(results['analysis_data'], results['risk_groups']) 사용")

# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    try:
        # 하나의 데이터셋만 선택해서 분석 (기본: missorigin)
        results = analyze_single_dataset(
            pam50_file="/home/dyan/data/data/brca-2Fpam50.tsv",
            survival_file="/home/dyan/data/data/TCGA-BRCA.survival.tsv",
            multiomics_data_dir="/home/dyan/nmf/mochi_code/results",
            dataset_name="impute_from_module",    # ← 상위 폴더
            analysis_label="BRCA_origin_analysis",
            n_comp_per_omics=50,
            variant="missorigin"  # NEW: 기본값
        )
        
        if results:
            print("\n🎯 이제 결과를 활용해서 카플란 그래프를 그릴 수 있습니다!")
            
            # 예시: 카플란 그래프 그리기
            # fig = create_kaplan_plot(
            #     results['analysis_data'],
            #     results['risk_groups'],
            #     title="BRCA 생존분석 결과",
            #     save_path="/path/to/kaplan_plot.png"
            # )
        else:
            print("❌ 분석 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
