# 새로운 다운스트림 분석 모듈

기존 `enhanced_downstream_analysis.py`를 대체하는 새로운 모듈 시스템입니다.

## 📁 모듈 구조

```
mochi_code/
├── metadata_processor.py      # 메타데이터 가공 모듈
├── multiomics_integrator.py  # 멀티오믹스 통합 모듈  
├── survival_analyzer.py       # 생존분석 모델 생성 모듈
├── kaplan_plotter.py         # 카플란 그래프 도구 모듈
├── downstream_pipeline.py     # 통합 실행 파이프라인
└── enhanced_downstream_analysis_backup.py  # 기존 모듈 백업
```

## 🚀 주요 기능

### 1. metadata_processor.py
- **PAM50 + 생존 데이터 통합**
- 컬럼명 자동 감지
- 데이터 검증 및 전처리
- 표준화된 메타데이터 출력

### 2. multiomics_integrator.py  
- **3개 모달리티 통합 (RNA, Protein, Methylation)**
- Z-score 표준화
- PCA 차원축소
- 공통 샘플 기반 통합

### 3. survival_analyzer.py
- **Cox 비례위험 모델 생성**
- Log-rank 테스트 실행
- 위험점수 및 위험그룹 계산
- 하이퍼파라미터 자동 튜닝

### 4. kaplan_plotter.py
- **matplotlib 기반 카플란 그래프**
- 완전 커스터마이징 가능
- 위험그룹별, PAM50 서브타입별 생존곡선
- 고품질 그래프 저장

### 5. downstream_pipeline.py
- **통합 실행 파이프라인**
- 단계별 또는 전체 분석 실행
- 결과 자동 저장 및 요약

## 💻 사용법

### 기본 사용법 (전체 파이프라인)

```python
from downstream_pipeline import DownstreamPipeline

# 파이프라인 초기화
pipeline = DownstreamPipeline(
    output_base="/path/to/output"
)

# 완전한 분석 실행
results = pipeline.run_complete_analysis(
    pam50_file="/path/to/pam50.tsv",
    survival_file="/path/to/survival.tsv", 
    multiomics_data_dir="/path/to/multiomics/data",
    analysis_label="BRCA_analysis",
    n_comp_per_omics=50,
    adjust_pam50=True
)
```

### 단계별 실행

```python
# 1. 메타데이터만
metadata = pipeline.run_metadata_only(
    pam50_file="/path/to/pam50.tsv",
    survival_file="/path/to/survival.tsv"
)

# 2. 멀티오믹스 통합만
integrated_data = pipeline.run_integration_only(
    multiomics_data_dir="/path/to/data",
    n_comp_per_omics=50
)

# 3. 생존분석만
survival_results = pipeline.run_survival_only(
    integrated_data=integrated_data,
    metadata=metadata
)
```

### 개별 모듈 사용

```python
# 메타데이터 처리
from metadata_processor import MetadataProcessor
processor = MetadataProcessor()
metadata = processor.integrate_metadata(
    pam50_file="/path/to/pam50.tsv",
    survival_file="/path/to/survival.tsv"
)

# 멀티오믹스 통합
from multiomics_integrator import MultiOmicsIntegrator
integrator = MultiOmicsIntegrator(n_comp_per_omics=50)
integrated = integrator.run_complete_integration(
    data_dir="/path/to/data",
    output_file="/path/to/output.tsv"
)

# 생존분석
from survival_analyzer import SurvivalAnalyzer
analyzer = SurvivalAnalyzer()
results = analyzer.run_complete_survival_analysis(
    X=integrated,
    metadata=metadata
)

# 카플란 그래프
from kaplan_plotter import KaplanPlotter
plotter = KaplanPlotter()
fig = plotter.plot_risk_group_comparison(
    survival_data=metadata,
    risk_scores=results['cox_analysis']['risk_scores'],
    risk_groups=results['cox_analysis']['risk_groups']
)
```

## 🎨 카플란 그래프 커스터마이징

```python
# 사용자 정의 설정
custom_config = {
    'title': 'BRCA 생존분석 결과',
    'xlabel': '생존 기간 (개월)',
    'ylabel': '생존 확률',
    'linewidth': 3.0,
    'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1'],
    'figsize': (14, 10),
    'style': 'seaborn-v0_8-whitegrid'
}

# 커스텀 그래프 생성
fig = plotter.create_custom_plot(
    data=plot_data,
    plot_config=custom_config,
    save_path="/path/to/custom_plot.png"
)
```

## 📊 출력 결과

### 파일 구조
```
output/
├── BRCA_analysis_metadata.tsv           # 메타데이터
├── BRCA_analysis_integrated_multiomics.tsv  # 통합 데이터
├── survival_analysis/                   # 생존분석 결과
│   ├── BRCA_analysis_risk_scores.tsv
│   └── BRCA_analysis_analysis_summary.json
├── plots/                               # 그래프
│   ├── BRCA_analysis_risk_groups_km.png
│   └── BRCA_analysis_pam50_subtypes_km.png
└── complete_results/                     # 전체 요약
    └── BRCA_analysis_complete_summary.json
```

### 주요 지표
- **Cox C-index**: 생존 예측 성능 (0.5~1.0, 높을수록 좋음)
- **Log-rank p-value**: 위험그룹 간 생존 차이 유의성
- **위험그룹 분할**: High/Low risk 그룹
- **중앙생존시간**: 각 그룹의 중앙값 생존 시간

## ⚙️ 설정 옵션

### 메타데이터 처리
- 컬럼명 자동 감지 또는 수동 지정
- 생존 시간 단위 변환 (일 ↔ 월)
- PAM50 서브타입 표준화

### 멀티오믹스 통합
- PCA 컴포넌트 수 조정 (기본값: 50)
- Z-score 표준화 적용 여부
- 공통 샘플 최소 수 설정

### 생존분석
- PAM50 서브타입 보정 여부
- Cox 모델 하이퍼파라미터 자동/수동 튜닝
- 최소 샘플 수 요구사항

### 그래프 생성
- 그래프 크기, 해상도, 폰트 설정
- 색상 팔레트 커스터마이징
- 신뢰구간, p-value 표시 옵션

## 🔧 의존성

```
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 1.0.0
lifelines >= 0.27.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

## 📝 예시 노트북

새로운 모듈을 사용한 예시는 `nmf/analysis/` 폴더의 노트북을 참조하세요.

## 🚨 주의사항

1. **기존 모듈 백업**: `enhanced_downstream_analysis_backup.py`로 백업됨
2. **데이터 형식**: TSV 파일 형식 지원
3. **메모리 사용량**: 대용량 데이터 처리 시 충분한 메모리 확보 필요
4. **결과 저장**: 자동으로 출력 디렉토리 생성 및 결과 저장

## 🤝 문제 해결

### 일반적인 오류
- **공통 샘플 부족**: 메타데이터와 멀티오믹스 데이터 간 샘플 ID 불일치
- **메모리 부족**: PCA 컴포넌트 수 줄이기
- **생존 이벤트 극단적**: 데이터 품질 확인 필요

### 디버깅
- 각 단계별 상세 로그 출력
- 중간 결과 파일 저장
- 단계별 실행으로 문제 지점 파악
