# 🧬 향상된 멀티오믹스 NMF+TGAN 시스템

## 📖 개요

이 시스템은 **3개 오믹스 데이터(RNA, DNA methylation, Protein)** 간의 상호 결측치 보정을 위한 향상된 NMF+TGAN 모델입니다.

### ✨ 주요 개선사항

1. **🔍 Masked MSE**: 관측/결측 가중 손실로 더 정확한 복원
2. **⚡ NMF-loss 가속**: 고정 사전 D + Ridge 회귀로 빠른 구조 손실 계산
3. **🧠 경량 Cross-Attention**: 특징 간 상호작용 포착을 위한 Transformer 블록
4. **🎯 타깃별 최적화**: RNA/methylation(비음수) vs Protein(음수 허용) 자동 설정

## 🏗️ 시스템 구조

```
📊 원본 데이터 (TCGA-BRCA)
    ↓
🔧 데이터 전처리 (PAM50 서브타입 필터링)
    ↓
📋 3가지 데이터셋 생성:
   ├── original: 원본 결측치
   ├── noisy: RNA에 노이즈 추가
   └── complete: 결측치 완전 제거
    ↓
🤖 NMF+TGAN 모델 학습
   ├── Generator: Cross-Attention + FC 레이어
   ├── Critic: WGAN 구조
   └── 손실: Masked MSE + 가속 NMF + WGAN
    ↓
📈 결과 분석 및 검증
```

## 🚀 Quick Start

### 1. 환경 설정
```bash
# 가상환경 활성화
source /home/dyan/nmf/.nmf/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비 (향상된 버전)
```bash
# 향상된 멀티오믹스 데이터 전처리
python data_prep_enhanced.py \
  --rna /path/to/TCGA-BRCA.star_tpm.tsv \
  --methy /path/to/TCGA-BRCA.methylation450.tsv \
  --protein /path/to/TCGA-BRCA.protein.tsv \
  --subtype /path/to/brca-2Fpam50.tsv \
  --out_dir ./processed_data \
  --prefix BRCA_PAM50
```

### 3. 모델 학습
```bash
# 멀티오믹스 NMF+TGAN 학습
python multi_omics_train.py \
  --config config.json \
  --device cuda
```

### 4. 전체 파이프라인 실행 (권장)
```bash
# 원스톱 파이프라인 실행
python run_complete_pipeline.py \
  --data_dir /path/to/your/data \
  --subtype_file /path/to/brca-2Fpam50.tsv \
  --output_dir ./results \
  --use_attention \
  --epochs 100 \
  --batch_size 32
```

### 5. 결측치 보정 (Imputation)
```bash
# 학습된 모델로 결측치 보정 수행
python imputation_module.py \
  --model_path ./results/tri_joint/tri_epoch_10.ckpt \
  --data_dir /path/to/data \
  --output_dir ./results/impute_from_module
```

### 6. 다운스트림 분석
```bash
# 생존분석 및 카플란 그래프 생성
python -c "
from downstream_analysis_new import analyze_single_dataset

results = analyze_single_dataset(
    pam50_file='/path/to/pam50.tsv',
    survival_file='/path/to/survival.tsv',
    multiomics_data_dir='./results',
    dataset_name='impute_from_module',
    analysis_label='BRCA_analysis'
)
"
```

## 📁 파일 구조

```
mochi_code/
├── 📊 핵심 모듈
│   ├── models.py              # Generator, Critic, Cross-Attention
│   ├── utils.py               # NMF 유틸리티 함수
│   ├── dataloader.py          # 데이터 로더
│   └── inference.py           # 추론 모듈
│
├── 🔧 데이터 처리
│   ├── data_prep_enhanced.py  # 향상된 데이터 전처리
│   └── imputation_module.py   # 결측치 보정 모듈
│
├── 🚀 실행 스크립트
│   ├── multi_omics_train.py   # 멀티오믹스 학습 (최신)
│   ├── run_complete_pipeline.py # 전체 파이프라인 실행
│   └── run_multi_omics_pipeline.py # (구버전, archive로 이동됨)
│
├── 📈 다운스트림 분석
│   └── downstream_analysis_new.py # 생존분석 및 카플란 그래프
│
├── 📦 기타
│   ├── requirements.txt       # 의존성 패키지
│   └── README.md             # 이 파일
│
└── 📚 Archive (이전 버전)
    ├── archive/old_code/      # 이전 코드 파일들
    └── archive/old_readme/    # 이전 README 파일들
```

## ⚙️ 주요 설정

### Cross-Attention 설정
```python
# Generator 초기화 시
netG = Generator(
    input_size=source_dim,
    output_size=target_dim,
    use_attn=True,           # Cross-Attention 사용
    nonneg_output=True,      # RNA/methylation 타깃
    n_src_tokens=8          # 소스를 8개 가상 토큰으로 분해
)
```

### 타깃 오믹스별 자동 설정
- **RNA/Methylation**: `nonneg_output=True` (ReLU 활성화)
- **Protein**: `nonneg_output=False` (음수 허용)

### NMF-loss 가속
```python
# 고정 사전 D 사용
D = torch.tensor(V_Global.T, dtype=torch.float32)
DtD_inv = LA.inv(DtD + ridge_alpha * torch.eye(DtD.shape[0]))

# 빠른 U 추정
U_hat = (fake @ D) @ DtD_inv
loss_U = nn.functional.mse_loss(fake_rec, fake)
```

## 🔧 하이퍼파라미터

### 기본값
```python
{
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "n_critic": 5,
    "clip_value": 0.01,
    "k_nmf": 50,
    "U_loss_multiplier": 0.1,
    "mse_loss_multiplier": 1.0
}
```

### Cross-Attention 설정
```python
{
    "d_model": 256,      # 임베딩 차원
    "d_qkv": 64,         # Q/K/V 차원
    "ff": 512,           # Feed-forward 차원
    "pdrop": 0.1,        # Dropout 비율
    "n_src_tokens": 8    # 소스 토큰 수 (1×m 어텐션)
}
```

## 📊 결과 분석

### 생성된 파일들
- **데이터셋**: `BRCA_PAM50.{omics}.{variant}.tsv`
- **학습 쌍**: `BRCA_PAM50.{source}_to_{target}.{variant}.{source/target}.csv`
- **요약**: `BRCA_PAM50.summary.tsv`
- **모델 결과**: `results/{variant}/` 디렉토리
- **체크포인트**: `results/{variant}/{model_name}_epoch_{N}.ckpt`

### 성능 지표
- **Masked MSE**: 관측/결측 가중 손실
- **구조 손실**: NMF 재구성 오차
- **WGAN 손실**: 생성 품질 지표

### 다운스트림 분석 결과
- **통합 데이터**: 멀티오믹스 PCA 통합 결과
- **생존분석**: Cox C-index, Log-rank p-value
- **카플란 그래프**: 위험그룹별 생존곡선

## 🧪 실험 예시

### 1. 기본 실험
```bash
# Cross-Attention 없이
python run_complete_pipeline.py \
  --data_dir ./data \
  --subtype_file ./brca-2Fpam50.tsv \
  --output_dir ./results_basic
```

### 2. Cross-Attention 실험
```bash
# Cross-Attention 사용
python run_complete_pipeline.py \
  --data_dir ./data \
  --subtype_file ./brca-2Fpam50.tsv \
  --output_dir ./results_attention \
  --use_attention
```

### 3. 하이퍼파라미터 튜닝
```bash
# 에포크 수와 배치 크기 조정
python multi_omics_train.py \
  --config config.json \
  --epochs 200 \
  --batch_size 64 \
  --use_attention
```

## 🔍 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   --batch_size 16
   ```

2. **학습 불안정**
   ```bash
   # Critic 업데이트 횟수 줄이기
   --n_critic 3
   ```

3. **과적합**
   ```bash
   # Dropout 증가
   # Early stopping 사용
   ```

### 디버깅 팁

- **TensorBoard 로그 확인**: `tensorboard --logdir ./logs`
- **그래디언트 클리핑**: `clip_value` 조정
- **학습률 스케줄링**: `learning_rate` 점진적 감소

## 📚 참고 자료

- **원본 NMF+TGAN**: [논문 링크]
- **Cross-Attention**: Transformer 아키텍처
- **WGAN**: Wasserstein GAN
- **TCGA 데이터**: The Cancer Genome Atlas

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스



---

**🎯 목표**: 멀티오믹스 데이터의 상호 결측치 보정을 통한 정확한 생물학적 패턴 복원

**🚀 핵심**: Cross-Attention + Masked MSE + NMF-loss 가속으로 성능과 속도 모두 향상
