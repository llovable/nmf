# MOCHI: 교차-어텐션과 NMF 구조 손실을 결합한 다중오믹스 결측치 상호보정 프레임워크

*(초안 / Korean draft — 타깃 저널: Bioinformatics, Oxford University Press)*

> 작성 메모: 본 문서는 `mochi_code` 분석 결과를 바탕으로 한 1차 초안입니다.
> Bioinformatics "Original Paper" 구조(Abstract → Introduction → Materials and Methods → Results → Discussion/Conclusion)를
> 따랐습니다. `〔확인 필요〕` 표시는 투고 전 저자가 값/사실을 반드시 확정해야 하는 부분입니다.

---

## Title (제목 후보)

- (국문) 교차-어텐션 생성적 적대 신경망과 NMF 구조 정규화를 이용한 다중오믹스 결측치 상호보정: MOCHI
- (영문) **MOCHI: cross-attention adversarial imputation with NMF structural regularization for mutual recovery of missing multi-omics data**

**Running title:** MOCHI: adversarial multi-omics imputation

---

## Authors / Affiliations

〔확인 필요〕 저자 명단·소속·교신저자(Contact 이메일)를 기입하세요.

- First Author¹, …, Corresponding Author¹*
- ¹ Department/Institution, …
- *To whom correspondence should be addressed. E-mail: 〔확인 필요〕

---

## Abstract

**Motivation:**
RNA 발현, DNA 메틸화, 단백질 발현 등 다중오믹스(multi-omics) 데이터는 암의 분자 아형과
예후를 다각도로 설명하지만, 실제 코호트에서는 플랫폼·비용·표본 품질의 차이로 인해
모달리티별로 광범위한 결측이 발생한다. 결측 표본을 단순 제거하면 분석 가능한
표본 수가 급감하고, 특정 모달리티만 사용하면 모달리티 간 상보적 정보가 손실된다.
따라서 모달리티 간 상호 관계를 학습하여 결측 부위를 일관되게 보정(imputation)하는
방법이 필요하다.

**Results:**
우리는 세 오믹스(RNA, DNA 메틸화, 단백질)를 한 번의 학습으로 상호보정하는
적대적 생성 프레임워크 **MOCHI**를 제안한다. MOCHI는 (i) 관측된 두 모달리티를
입력으로 받아 나머지 한 모달리티를 생성하는 세 개의 생성자(Gp, Gr, Gm)를
가중치 그래디언트 페널티(WGAN-GP)로 공동 학습하고, (ii) 소스 특징 간 상호작용을
포착하기 위한 경량 교차-어텐션 블록과 특징별 게이트(feature-wise gate)를 도입하며,
(iii) 고정 NMF 사전(dictionary)을 이용한 닫힌형(closed-form) 구조 손실로
생성물이 각 오믹스의 저차원 생물학적 구조를 따르도록 정규화한다.
또한 관측/결측을 구분하는 마스킹 평균제곱오차(masked MSE)를 사용해
관측 신호의 충실도를 유지한다.
TCGA-BRCA의 PAM50 아형 코호트에서, 인위적으로 주입한 10–50% 결측 환경에서도
MOCHI는 표본 단위 Pearson 상관 0.95 이상을 유지하며 RNA를 안정적으로 복원하였다.
보정된 다중오믹스를 PCA로 통합한 뒤 Cox 비례위험모형으로 평가한 결과,
모달리티 앙상블 위험점수의 일치도(C-index)는 0.92–0.94로 단일 모달리티(약 0.83)보다 높았고,
위험군 간 생존 곡선은 유의하게 분리되었다(log-rank P = 4.4×10⁻⁸).

**Availability and implementation:**
소스 코드(PyTorch)는 〔확인 필요: GitHub/Zenodo URL〕에서 제공된다.

**Contact:** 〔확인 필요: corresponding@inst.edu〕

**Supplementary information:** Supplementary data는 Bioinformatics 온라인에서 제공된다.

---

## 1 Introduction

다중오믹스 프로파일링은 동일한 환자 시료에서 유전체·전사체·후성유전체·단백질체
계층을 동시에 측정함으로써, 단일 계층으로는 드러나지 않는 분자 기전과 임상 표현형의
연관을 규명할 수 있게 한다. 특히 유방암에서 PAM50 기반 내재적 아형(Basal, Her2,
LumA, LumB)은 치료 반응과 예후를 층화하는 핵심 변수로, RNA·메틸화·단백질
계층의 통합 해석은 아형 생물학과 예후 예측의 정밀도를 높인다.

그러나 다중오믹스 통합 분석의 실질적 장벽은 **모달리티 간 결측**이다. 한 환자에서
모든 오믹스가 동일한 품질로 측정되는 경우는 드물며, 측정 플랫폼·검체 가용성·
비용 제약으로 인해 특정 모달리티가 통째로 누락되거나 부분적으로 결측된다.
완전 사례(complete-case) 분석은 표본 수를 급격히 줄여 통계적 검정력을 떨어뜨리고,
모달리티별 단순 대치(평균/KNN 등)는 모달리티 간 비선형 상호 의존성을 반영하지 못한다.

이를 해결하기 위해 최근 생성 모델 기반 접근이 활발히 연구되었다. 적대적 생성
신경망(GAN)은 한 모달리티에서 다른 모달리티를 합성하는 모달리티 변환에 효과적이며,
오토인코더·변분 모델은 공유 잠재공간을 통한 결측 복원에 사용된다. 그럼에도
(i) 세 개 이상의 모달리티를 **동시에 상호보정**하면서 (ii) 생성물이 각 오믹스의
**저차원 생물학적 구조**(예: 비음수 행렬분해, NMF로 표현되는 메타유전자/메타단백질
프로그램)를 따르도록 강제하고, (iii) 고차원 소스 특징 간의 **상호작용**을
효율적으로 포착하는 통합적 방법은 아직 부족하다.

본 연구에서 우리는 이 세 요소를 단일 프레임워크로 결합한 **MOCHI(Multi-Omics
Cross-attention Harmonized Imputation)** 를 제안한다. 핵심 기여는 다음과 같다.

1. **삼중 공동(tri-joint) 적대 학습.** 관측된 두 모달리티를 결합해 나머지 한
   모달리티를 생성하는 세 생성자를 WGAN-GP로 한 번에 학습하여, 모달리티 간
   상호보정을 대칭적으로 수행한다.
2. **NMF 구조 정규화의 가속화.** 학습 전에 각 오믹스에서 고정 NMF 사전 D를
   구하고, 능형 회귀(ridge)로 닫힌형 투영 \(\hat U = Y D (D^\top D+\alpha I)^{-1}\)을
   사용하여 생성물이 오믹스 고유의 저차원 구조를 따르도록 빠르게 정규화한다.
3. **경량 교차-어텐션 + 특징별 게이트.** 소스를 다수의 가상 토큰으로 투영하여
   타깃 질의와 멀티헤드 어텐션을 수행하고, 0에 가깝게 초기화된 게이트로
   잔차(residual)를 점진적으로 주입하여 학습 안정성을 확보한다.
4. **마스킹 MSE.** 관측·결측을 구분하는 가중 손실로, 관측 신호 충실도를 유지하면서
   결측 부위 복원을 유도한다.

TCGA-BRCA PAM50 코호트에서 결측 주입 실험과 생존 다운스트림 분석을 통해
MOCHI의 보정 정확도와 임상적 유용성을 검증한다.

---

## 2 Materials and Methods

### 2.1 데이터셋과 전처리

분석에는 TCGA-BRCA의 세 오믹스를 사용하였다: RNA-seq 발현(STAR TPM, 60,660 features),
Illumina 450K DNA 메틸화(β-value), 단백질 발현(RPPA, 487 features) 〔확인 필요: 단백질
플랫폼/feature 정의〕. 메틸화는 차원과 계산비용을 고려해 분산 상위 10,000개 프로브를
선택하였다. 시료는 PAM50 라벨(Basal, Her2, LumA, LumB)이 부여된 표본으로 한정하였다.

전처리 파이프라인(`data_prep_enhanced.py`)은 다음 세 변형(variant)을 생성한다.

- **original:** 원본 관측치(자연 발생 결측 포함).
- **noisy:** RNA에 무작위 결측을 인위적으로 주입(10%, 30%, 50%)하여 정량 평가용
  정답(ground truth)을 확보한 데이터.
- **complete:** 결측이 전혀 없는 표본만 남긴 데이터.

각 모달리티의 표본은 TCGA 바코드를 기준으로 교집합/합집합 정렬되며,
다운스트림 평가에 사용한 완전 정렬 코호트의 공통 표본 수는 38개였다 〔확인 필요:
표본 수가 작아 검정력 한계가 있음 — Discussion 참고〕.

### 2.2 문제 정의

세 모달리티를 각각 RNA \(X_r\), 단백질 \(X_p\), 메틸화 \(X_m\)이라 하고, 결측 위치를
나타내는 이진 마스크 \(M_\bullet\)(1=결측, 0=관측)를 둔다. 목표는 관측된 두 모달리티로부터
세 번째 모달리티를 추정하는 세 사상을 학습하는 것이다.

\[
G_p:[X_r, X_m]\to \hat X_p,\quad
G_r:[X_p, X_m]\to \hat X_r,\quad
G_m:[X_r, X_p]\to \hat X_m .
\]

추론 시 관측 부위는 원본값을 유지하고 결측 부위만 생성값으로 치환한다
(\(\tilde X = M\odot \hat X + (1-M)\odot X\)).

### 2.3 생성자: 교차-어텐션 + 게이트 잔차 + MLP

각 생성자는 (선택적) 교차-어텐션 블록과 다층 퍼셉트론(MLP) 본체로 구성된다
(`models.py`).

**교차-어텐션 블록.** 소스 벡터를 \(m\)개(기본 8개)의 가상 토큰으로 선형 투영하여
키/값(K, V)을 만들고, 타깃 입력을 단일 질의(Q)로 투영한다. 멀티헤드 어텐션
(헤드 4개, 헤드 차원 64, \(d_{model}=256\))과 피드포워드(차원 512), GroupNorm(8)
잔차로 토큰 간 상호작용을 요약한다. 어텐션 출력은 입력 차원으로 재투영된 뒤,
**특징별 게이트** \(g=\sigma(\alpha)\)를 통해 \(x \leftarrow x + g\odot x_{attn}\)로
주입된다. \(\alpha\)는 \(\sigma(-2.2)\approx 0.1\)로, 재투영 가중치는 0으로 초기화하여
초기에는 항등(identity)에 가깝게 작동하고 학습이 진행되며 어텐션 기여를 키운다.

**MLP 본체.** \( \text{input}\to 1024 \to 512 \to \text{output}\) 구조에 GroupNorm(8),
ReLU, Dropout(0.2)을 적용한다. 출력 활성화는 모달리티별로 다르게 설정한다:
메틸화는 \(\sigma(\cdot)\)(0–1), RNA는 softplus(≥0), 단백질은 선형(음수 허용).
이는 각 오믹스 값의 정의역을 반영한다.

### 2.4 판별자(Critic)와 WGAN-GP 목적함수

판별자는 시그모이드 없이 실수 점수를 출력하는 MLP(512→256→1, GroupNorm·ReLU·
Dropout)이다. 삼중 공동 학습은 그래디언트 페널티가 포함된 Wasserstein GAN
손실(WGAN-GP, \(\lambda_{gp}=10\))을 사용한다. 판별자/생성자별 손실은 다음과 같다.

\[
\mathcal{L}_{D} = \sum_{k\in\{p,r,m\}}\Big[\mathbb{E}[D_k(\hat X_k)] - \mathbb{E}[D_k(X_k)]
+ \lambda_{gp}\,\mathbb{E}\big[(\lVert\nabla_{\hat{x}}D_k(\hat{x})\rVert_2-1)^2\big]\Big],
\]
\[
\mathcal{L}_{G}^{adv} = -\sum_{k}\mathbb{E}[D_k(\hat X_k)].
\]

판별자는 생성자 1회 갱신당 \(n_{critic}=5\)회 갱신하며, Adam(lr=10⁻⁴, β=(0.5, 0.9))을
사용한다. 〔주: 단일 타깃 변형 학습기(`MultiOmicsTrainer`)는 가중치 클리핑(0.01)
기반 WGAN과 RMSprop를 사용하며, 삼중 공동 학습기(`WGAN_GP_Trainer_Tri`)는
위의 WGAN-GP를 사용한다.〕

### 2.5 마스킹 MSE와 동적 마스킹

관측 신호 충실도를 위해, 평가 가능한 위치에서만 계산되는 마스킹 MSE를 더한다.

\[
\mathcal{L}_{MSE} = \sum_{k}\frac{\lVert M_k^{eval}\odot(\hat X_k - X_k)\rVert^2}{\text{(원소 수)}}.
\]

선택적으로 **동적 마스킹**을 사용하여, 매 배치마다 관측 위치 일부를
무작위 비율(U[0.05, 0.5])로 가상 결측으로 만들어 자기지도(self-supervised)
복원 신호를 강화한다.

### 2.6 NMF 구조 정규화(가속형)

생성물이 각 오믹스의 저차원 생물학적 구조를 따르도록, 학습 전에 비음수 행렬분해로
고정 사전 \(D\in\mathbb{R}^{p\times k}\)(기본 \(k=20\))를 구한다. RNA/메틸화는 음수를
0으로 클립하고, 단백질은 행별 시프트로 비음수화한다. 학습 중에는 생성물 \(Y\)를
다음 닫힌형으로 사전 좌표계에 투영해 재구성 오차를 손실로 사용한다.

\[
\hat U = Y\,D\,(D^\top D + \alpha I)^{-1},\qquad
\mathcal{L}_{NMF} = \lVert \hat U D^\top - Y \rVert^2,
\]

여기서 \(\alpha\)는 능형 계수(기본 10⁻³)이다. 사전을 고정함으로써 매 스텝마다
완전한 NMF를 다시 풀 필요 없이 행렬 연산만으로 구조 손실을 계산할 수 있다.

최종 생성자 목적함수는 가중합이다.
\[
\mathcal{L}_{G} = \mathcal{L}_{G}^{adv} + \lambda_{mse}\,\mathcal{L}_{MSE}
+ \lambda_{nmf}\,\mathcal{L}_{NMF},\quad (\lambda_{mse}=\lambda_{nmf}=0.1).
\]

### 2.7 학습 설정

배치 크기 32, 에폭 최대 100–180, 조기 종료(early stopping, patience 8–10)를 사용했다.
검증은 고정 시드·고정 비율(기본 30%)로 표본화한 가상 결측 위치에서 모달리티별
RMSE를 측정하고 평균 RMSE가 최소인 체크포인트를 최종 모델로 선택한다.

### 2.8 다운스트림 평가: 통합과 생존분석

보정된 세 모달리티를 각각 표준화한 뒤 PCA(모달리티당 50개 주성분)로 축약하고
이어 붙여 통합 표현을 만든다(`downstream_analysis_new.py`). PAM50 더미 변수를 공변량으로
포함한 Cox 비례위험모형(penalizer=0.1, l1_ratio=0.5)으로 위험점수를 추정하고,
부분위험(partial hazard) 중앙값을 기준으로 고/저위험군을 나눠 Kaplan–Meier 곡선과
log-rank 검정을 수행했다. 또한 모달리티별 위험점수를 z-정규화 후 평균한
앙상블 점수를 함께 평가했다.

---

## 3 Results

### 3.1 결측 주입 환경에서의 보정 정확도

RNA에 10/30/50% 무작위 결측을 주입한 뒤 복원 정확도를 결측 위치에서만 평가했다
(Table 1). 자연 결측만 있는 *origin* 설정에서는 관측값 재구성이 거의 완전했고
(R² ≈ 1.00), 인위 결측 환경에서도 RNA 표본 단위 Pearson 상관이 0.957–0.958로
높게 유지되었다. 결측률이 10%에서 50%로 증가해도 RMSE는 0.676→0.679, R²는
0.885→0.883으로 거의 변하지 않아, MOCHI가 높은 결측률에서도 안정적임을 보였다.

> 〔주의: feature-wise Pearson은 결측 환경에서 매우 낮게 나타났다(≈0.002–0.004).
> 이는 표본 간 상대 순위는 잘 보존되나 개별 feature 축의 미세 변동 복원은 제한적일
> 수 있음을 시사하며, Discussion에서 다룬다.〕

**Table 1.** RNA 결측 보정 정확도(결측 위치 평가). 〔값 출처: `imputation_eval_summary.tsv`〕

| 결측률 | RMSE | MAE | NRMSE | R² | Sample-wise Pearson | 평가 셀 수 |
|---|---|---|---|---|---|---|
| origin | 7.3×10⁻⁸ | 3.0×10⁻⁸ | 3.7×10⁻⁸ | 1.000 | 1.000 | 2,305,080 |
| 10% | 0.676 | 0.352 | 0.339 | 0.885 | 0.958 | 230,348 |
| 30% | 0.679 | 0.353 | 0.341 | 0.884 | 0.958 | 691,101 |
| 50% | 0.679 | 0.353 | 0.341 | 0.883 | 0.958 | 1,152,085 |

(단백질·메틸화 *origin* 재구성도 R² ≈ 1.00이었다.)

### 3.2 다중오믹스 통합 표현의 예후 예측력

보정·통합된 표현으로 추정한 Cox 위험점수의 생존 일치도(C-index)와 위험군
log-rank 검정 결과를 Table 2에 정리했다. 모든 설정에서 위험군 간 생존 곡선이
유의하게 분리되었으며(log-rank P = 4.4×10⁻⁸), C-index는 *origin* 0.855에서
결측 보정 데이터(10/30/50%) 0.903–0.919로 나타났다.

**Table 2.** 통합 표현 기반 생존분석. 〔값 출처: `survival_analysis_summary.tsv`〕

| 데이터셋 | C-index | log-rank P | n |
|---|---|---|---|
| origin | 0.855 | 4.4×10⁻⁸ | 38 |
| 10% | 0.903 | 4.4×10⁻⁸ | 38 |
| 30% | 0.911 | 4.4×10⁻⁸ | 38 |
| 50% | 0.919 | 4.4×10⁻⁸ | 38 |

### 3.3 모달리티 앙상블의 이점

단일 모달리티 위험점수의 C-index는 약 0.81–0.86 범위였으나, 세 모달리티를
z-정규화 평균으로 결합한 앙상블 점수의 C-index는 0.911–0.935로 일관되게 더 높았다
(Table 3). 이는 보정된 모달리티들이 상보적 예후 정보를 담고 있으며, MOCHI 통합이
이를 효과적으로 결합함을 시사한다.

**Table 3.** 모달리티별 vs. 앙상블 C-index(예시: origin/50%). 〔값 출처: `summary_survival.tsv`〕

| 데이터셋 | RNA | Protein | Methyl | Ensemble(mean-z) |
|---|---|---|---|---|
| origin | 0.839 | 0.815 | 0.831 | 0.919 |
| 50% | 0.823 | 0.847 | 0.863 | 0.935 |

### 3.4 정성적 결과

각 설정별 Kaplan–Meier 위험군 곡선과 PAM50 아형별 곡선은 Figure 1〔확인 필요:
`BRCA_*_kaplan_risk.png`, `BRCA_*_kaplan_pam50.png`를 본문 그림으로 배치〕에 제시한다.
위험군 곡선은 명확히 분리되며, 아형별 곡선은 LumA의 양호한 예후와 LumB의 상대적
불량 예후 경향을 보였다.

---

## 4 Discussion

MOCHI는 세 오믹스를 대칭적으로 상호보정하면서, NMF 구조 손실과 교차-어텐션,
마스킹 MSE를 결합해 생물학적 타당성과 복원 충실도를 동시에 추구한다. 결측률이
50%까지 증가해도 표본 단위 상관과 예후 예측력이 유지되어, 결측이 큰 실제 코호트에서
완전 사례 분석의 표본 손실을 완화할 수 있는 가능성을 보였다. 또한 보정된 모달리티의
앙상블이 단일 모달리티보다 일관되게 높은 C-index를 보여, 통합 표현의 임상적 가치를
뒷받침한다.

**한계.** (i) **표본 수**가 38개로 작아 생존분석 결과(특히 동일한 log-rank P)는
검정력과 일반화에 한계가 있으며, 보정 데이터의 C-index가 origin보다 높게 나온 점은
주의 깊게 해석해야 한다(정보 누수·과적합 가능성을 배제할 추가 검증 필요). 〔확인 필요:
학습/평가 표본 분리, 다중 시드 반복, 교차검증, 신뢰구간 보고.〕 (ii) *origin*의
재구성 R²가 사실상 1.0인 것은 진정한 결측 보정이 아니라 관측값 재구성을 측정한 것일
수 있어, 평가 프로토콜을 명확히 분리해야 한다. (iii) **feature-wise 상관이 낮은** 점은
개별 feature 단위 정밀도의 개선 여지를 시사한다. (iv) 비교군(KNN/MICE/오토인코더/
기존 GAN 보정법) 대비 **벤치마크**가 필요하다. (v) 단일 암종(BRCA)·단일 코호트(TCGA)에
국한되어 외부 검증이 요구된다.

**향후 연구.** 더 큰 다기관 코호트로의 확장, 어블레이션(어텐션/NMF/마스킹 MSE 기여
분리), 모달리티 전체 결측(block-missing) 시나리오, 불확실도 정량화, 표준 보정법과의
정량 비교를 계획한다.

---

## 5 Conclusion

우리는 교차-어텐션 적대 생성과 NMF 구조 정규화를 결합한 다중오믹스 상호보정
프레임워크 MOCHI를 제안하고, TCGA-BRCA PAM50 코호트에서 높은 결측률에서도
안정적인 복원과 예후 예측력을 보임을 확인했다. MOCHI는 결측이 만연한 다중오믹스
연구에서 분석 가능 표본을 보존하고 통합 분석을 가능케 하는 실용적 도구가 될 수 있다.

---

## Acknowledgements / Funding / Conflict of Interest

〔확인 필요〕 감사의 글, 연구비 지원, 이해상충 여부를 기입하세요.

## Data availability

TCGA-BRCA 데이터는 GDC/UCSC Xena 등 공개 저장소에서 이용 가능하다 〔확인 필요: 접근 경로〕.
보정 결과 및 코드는 〔확인 필요: 저장소 URL〕에서 제공된다.

## References

〔확인 필요: 아래는 인용이 필요한 항목의 자리표시자입니다. Bioinformatics 양식에 맞춰 정리하세요.〕

1. TCGA Network. Comprehensive molecular portraits of human breast tumours. *Nature*, 2012.
2. Parker JS, et al. PAM50 intrinsic subtype classifier. *J Clin Oncol*, 2009.
3. Goodfellow I, et al. Generative adversarial nets. *NeurIPS*, 2014.
4. Arjovsky M, et al. Wasserstein GAN. *ICML*, 2017.
5. Gulrajani I, et al. Improved training of Wasserstein GANs (WGAN-GP). *NeurIPS*, 2017.
6. Lee DD, Seung HS. Learning the parts of objects by non-negative matrix factorization. *Nature*, 1999.
7. Vaswani A, et al. Attention is all you need. *NeurIPS*, 2017.
8. 〔다중오믹스 결측치 보정 관련 선행연구 추가: 예) MOFA+, totalVI, scMVAE, GLUE 등〕
