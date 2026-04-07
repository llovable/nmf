# 새로운 다운스트림 분석 모듈

하나의 파이썬 모듈(`downstream_analysis_new.py`) 안에 모든 기능을 함수로 정의했습니다.

## 📁 모듈 구조

```
mochi_code/
├── downstream_analysis_new.py      # 새로운 통합 모듈
└── enhanced_downstream_analysis_backup.py  # 기존 모듈 백업
```

## 🚀 주요 기능

### 0. 메타데이터 가공
- `create_metadata()`: PAM50 + 생존 데이터 통합

### 1. 하나의 데이터셋 선택 및 통합  
- `integrate_single_dataset()`: 선택한 데이터셋의 3개 모달리티만 통합

### 2. 생존분석 모델 생성
- `run_survival_analysis()`: Cox 비례위험 모델 + Log-rank 테스트

### 3. 카플란 그래프 도구 (별도 활용용)
- `create_kaplan_plot()`: matplotlib 기반 생존곡선 (커스터마이징 가능)

### 4. 단일 데이터셋 분석 실행
- `analyze_single_dataset()`: 하나의 데이터셋만 선택해서 전체 분석 실행

## 💻 사용법

### 하나의 데이터셋만 선택해서 분석
```python
from downstream_analysis_new import analyze_single_dataset

# 원하는 데이터셋만 선택해서 분석
results = analyze_single_dataset(
    pam50_file="/path/to/pam50.tsv",
    survival_file="/path/to/survival.tsv",
    multiomics_data_dir="/path/to/results",  # 루트 디렉토리
    dataset_name="impute_from_module",       # 원하는 데이터셋 선택
    analysis_label="BRCA_analysis"
)
```

### 단계별 실행
```python
from downstream_analysis_new import *

# 1. 메타데이터만
metadata = create_metadata(pam50_file, survival_file)

# 2. 특정 데이터셋 통합만
integrated = integrate_single_dataset(
    data_dir="/path/to/results",
    dataset_name="origin",  # 원하는 데이터셋
    n_comp_per_omics=50
)

# 3. 생존분석만
survival_results = run_survival_analysis(X, metadata)

# 4. 카플란 그래프만 (결과 활용)
fig = create_kaplan_plot(
    survival_results['analysis_data'],  # 분석용 데이터
    survival_results['risk_groups'],    # 위험그룹
    title="생존곡선"
)
```

## 🎯 핵심 특징

- **하나의 데이터셋만 선택**: `dataset_name`으로 원하는 데이터셋 지정
- **결과 활용**: 생존분석 결과를 반환하여 카플란 그래프 별도 생성
- **유연성**: 각 단계를 독립적으로 실행 가능
- **자동화**: 선택한 데이터셋의 3개 모달리티 자동 통합

## 📊 출력 결과

- **메타데이터**: PAM50 + 생존 정보 통합
- **통합 데이터**: 선택한 데이터셋의 3개 모달리티 PCA 통합
- **생존분석**: Cox C-index, Log-rank p-value, 위험그룹, 분석용 데이터
- **카플란 그래프**: 별도 함수로 생성 (완전 커스터마이징 가능)

## 🎨 카플란 그래프 활용

```python
# 생존분석 결과에서 데이터 추출
analysis_data = results['analysis_data']  # 생존분석용 데이터
risk_groups = results['risk_groups']      # 위험그룹

# 커스텀 카플란 그래프 생성
fig = create_kaplan_plot(
    data=analysis_data,
    risk_groups=risk_groups,
    title="BRCA 생존분석 결과",
    figsize=(14, 10),
    save_path="/path/to/custom_plot.png"
)
```

## 🔧 의존성

```
pandas, numpy, scikit-learn, lifelines, matplotlib
```

## 🚨 주의사항

1. **기존 모듈 백업**: `enhanced_downstream_analysis_backup.py`로 백업됨
2. **데이터셋 선택**: `dataset_name`으로 원하는 데이터셋만 지정
3. **결과 활용**: 생존분석 결과를 활용해서 카플란 그래프 별도 생성
4. **파일 관리**: 하나의 모듈로 모든 기능 통합
