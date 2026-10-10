# 인수인계: 원고 전 마지막 다섯 작업

저장소: https://github.com/llovable/nmf
브랜치: `cursor/fix-runbook-commands-a549`

이 파일만 읽고 아래 명령을 그대로 돌리시면 됩니다.

**부탁드리는 일:** 다섯 가지를 돌리고 결과를 보내주시면 됩니다.
**셋은 학습이 필요 없고**(이미 학습된 모델을 불러다 평가만), 둘만 학습합니다.
학습까지 다 합쳐 **약 한 시간 반**입니다. 작업 3의 30–40분은 GPU 두 장 기준입니다.

이걸로 논문 본문에 들어갈 표와 그림이 전부 채워집니다.

---

## 0. 먼저

```bash
cd /home/dyan/nmf
git fetch origin
git checkout cursor/fix-runbook-commands-a549
git pull --ff-only origin cursor/fix-runbook-commands-a549
git log --oneline -1
source /home/dyan/nmf/.nmf/bin/activate
cd mochi_code && python tests/test_patches.py
```

마지막 줄이 `모든 검증 통과` 여야 합니다. 통과 안 하면 **돌리지 말고 알려주세요.**

아래 다섯은 서로 독립이라 **순서를 바꿔도 되고 따로 돌려도 됩니다.**
급한 순서는 3 → 2 → 1 → 4 → 5 입니다.

---

## 작업 1. 단계 실험 다시 계산 + 결측 비율별 곡선

**학습 없음. 기존 체크포인트만 읽습니다. 몇 분.**

### 왜

지난번 표를 **한 조건의 흔들림 폭**으로 판정했는데, 이 실험들은 시작 조건을
공유하므로 **같은 시작 조건끼리 뺀 차이**로 봐야 합니다. 공통 흔들림이
상쇄되어 훨씬 예민해집니다(확인해 보니 약 8배).

그리고 결측 비율을 30% 한 점에서만 쟀습니다. 주의집중 방식을 넣은 근거가
"결측이 심할 때 유리하다"였으니, **비율을 올려가며 봐야** 그 설명이 섭니다.

### 명령

```bash
cd /home/dyan/nmf/mochi_code
source /home/dyan/nmf/.nmf/bin/activate

python eval_cell_rate.py \
  --root results/current/lr_stage5c \
  --cohorts brca luad kirc --stages c0 c1 c2 --seeds 0 1 2 \
  --rates 0.1 0.3 0.5 0.7 0.9 --split test --gpu 0 \
  --out results/current/lr_stage5c/cell_rate.tsv 2>&1 | tee ~/task1.log
```

### 보는 곳

`짝지은 차이` 절의 **평균/sd** 와 **부호** 열입니다. 평균/sd 는 차이 평균을
차이의 표준편차로 나눈 값입니다. 절대값이 1을 넘고 부호가 `3/3`처럼
한쪽으로 몰리면 신호이고, 부호가 반반이면 잡음입니다.

---

## 작업 2. 단순 방법과 비교  ← **심사 첫 질문을 막는 표**

**학습 없음. 몇 분.**

### 왜

우리 오차 0.88이 좋은 숫자인지 나쁜 숫자인지 **기준이 없습니다.**
그리고 심사에서 반드시 이 질문이 옵니다 —
**"복잡한 구조를 다 빼고 오토인코더만 쓰면 얼마 나옵니까?"**

지금 부분 결측에서의 이득이 *자기 데이터를 10배 믿는* 설정에서 나오기
때문입니다. 비슷하게 나와도 논문이 먼저 말해야 합니다.

### 명령

세 암종 각각에 대해, **생물학 지식을 전부 끈 조건**의 체크포인트를 씁니다.

```bash
cd /home/dyan/nmf/mochi_code
source /home/dyan/nmf/.nmf/bin/activate
for COH in brca luad kirc; do
  python eval_baselines.py \
    --data_dir processed_data/gate_$COH \
    --ckpt results/current/lr_ctl/none/${COH}_hybrid/nmf_tf_best.ckpt \
    --split test --rate 0.3 --gpu 0 \
    --out results/current/baselines_$COH.tsv
done 2>&1 | tee ~/task2.log
```

> 작업 3을 먼저 돌리셨다면 체크포인트는
> `results/current/lr_ctl/none/${COH}_s0/nmf_tf_best.ckpt` 입니다.
> `_hybrid` 와 `_s0` 중 있는 쪽을 쓰시면 됩니다.

### 보는 곳

`MOCHI` 줄과 `AE-only` 줄의 **부분 결측** 칸입니다. 둘이 비슷하면
부분 결측의 이득은 융합 구조의 공이 아닙니다.

**정상 동작 확인:** 로그에 `정상: 통째 결측에서 KNN과 softImpute가 같습니다`
가 찍혀야 합니다. 표에서도 두 칸이 **같은 값**이고 1.0 근처입니다.
한 오믹스가 통째로 비면 두 방법 다 "평균 찍기"로 퇴화하기 때문입니다.
`softImpute-concat` 은 학습 집합을 같이 넣으므로 이 값과 달라도 됩니다.
그 줄이 사실상 기준선입니다.

---

## 작업 3. 중심 대조를 세 번씩 반복  ← **타협 불가**

**학습 필요. GPU 두 장이면 약 30~40분. 하나면 약 두 배.**

### 왜

**논문의 중심 표가 반복 1회입니다.** 중심 주장이 "차이가 없다"인데,
차이가 없음을 보이려면 **같은 실험을 여러 번 돌렸을 때 얼마나 흔들리는지**를
반드시 보여야 합니다. 흔들림이 차이보다 크면 "차이 없음"이 증명되고,
작으면 주장이 뒤집힙니다. **지금은 어느 쪽인지 알 수 없습니다.**

심사에서 가장 먼저 걸릴 자리입니다.

### 명령

조건은 일곱 개입니다. `ctl_base`, `w0`, `nmf0`, `lw0_nmf0`, `none`,
`resid`, `resid_nmf0`. 암종 세 개 × 시드 세 개라 학습은 63번이고,
9월에 한 번씩 돈 것의 세 배입니다.

```bash
cd /home/dyan/nmf/mochi_code
bash scripts/train_ablate_p0.sh 2>&1 | tee ~/task3.log
```

이 명령은 GPU 0과 1을 같이 씁니다. GPU가 하나면 0번만 지정하세요.
그때는 약 두 배 걸립니다.

```bash
GPU=0 bash scripts/train_ablate_p0.sh 2>&1 | tee ~/task3.log
```

- 결과는 `results/current/lr_ctl/<조건>/<암종>_s<번호>/` 에 쌓입니다.
- **기존 `_hybrid` 폴더는 그대로 둡니다.** 2026-09-03 단일 시드 런입니다.
  논문 표는 새로 나온 `_s0/_s1/_s2` 쪽을 씁니다.
- 끊기면 같은 명령으로 이어집니다. 체크포인트가 있는 조건은 `skip train`
  이 찍히고, 평가 파일(`amplitude.tsv`)이 있으면 `skip eval` 이 찍힙니다.

---

## 작업 4. 학습 환자 수를 줄여가며 곡선

**학습 필요. 약 30~40분.**

### 왜

"모델 크기가 아니라 환자 수가 병목"이라는 주장의 근거가 지금은 음성 결과
하나뿐이고, **그마저 교란돼 있습니다**(모델을 키운 실험이 크기만 바꾼 게
아니라 설정도 달랐습니다). 441명에서도 곡선이 가파르게 떨어지는 중이면
**그림 한 장으로 증명**됩니다.

### 명령

```bash
cd /home/dyan/nmf/mochi_code
GPU=0 bash scripts/train_nsweep.sh 2>&1 | tee ~/task4.log
```

환자 수 100 · 200 · 300 · 전체를 돌립니다. 스크립트가 학습 파일의 열을
세어, 그 인원 이상이면 건너뜁니다. 폐선암 학습은 223명이고 신장암 학습은
300보다 적어서, 그 둘은 300이 생략됩니다. 유방암 학습 441명은 300도 돕니다.

끝나는 표는 **검증 집합**의 통째 결측 오차입니다. 조기 종료에 쓴 숫자이고
검사 집합 숫자가 아닙니다. `±` 는 초기화 흔들림입니다. 세 시드는 같은
환자 부분표집을 쓰고, 초기화만 다릅니다.

끊기면 같은 명령으로 이어집니다. 체크포인트가 있는 칸은 건너뜁니다.

---

## 작업 5. 생존 분석과 하위 과제

**학습 없음. 몇 분.**

### 왜

지금 생물학 쪽 지표가 "환자 묶음 정확도" 하나뿐인데 **거기서는 기존 방법에
집니다.** 다른 축에서 하나라도 서면 논문이 훨씬 좋아지고, 안 서면 그
사실을 알고 쓰는 게 낫습니다.

### 명령

하위 과제부터입니다. **유방암만** 임상 정보로 돌아갑니다.

```bash
cd /home/dyan/nmf/mochi_code
source /home/dyan/nmf/.nmf/bin/activate
python eval_downstream.py \
  --data_dir processed_data/gate_brca \
  --clinical processed_data/clinical/BRCA_clinicalMatrix \
  --mimir_dir results/current/mimir \
  --out_dir results/current/down_final --gpu 0 \
  --runs "MOCHI=results/current/lr_ctl/none/brca_hybrid/nmf_tf_best.ckpt" \
  2>&1 | tee ~/task5.log
```

그다음 생존 분석입니다. `--runs` 를 받도록 고쳐 뒀습니다.

```bash
python eval_survival.py \
  --data_dir processed_data/gate_brca \
  --surv processed_data/clinical/TCGA-BRCA.survival.tsv.gz \
  --out_dir results/current/surv_final --gpu 0 \
  --runs "MOCHI=results/current/lr_ctl/none/brca_hybrid/nmf_tf_best.ckpt" \
  2>&1 | tee -a ~/task5.log
```

생존 파일이 없으면 먼저 받으세요. `fetch_xena_cohort.sh` 에 추가해 뒀습니다.

```bash
bash scripts/fetch_xena_cohort.sh BRCA      # 이미 있는 파일은 건너뜁니다
```

옛 비교 모델(MOCHI-v5, 공식 구현)은 **파일이 있을 때만** 쓰고 없으면 건너뜁니다.
그래서 `mean / TOBMI / Ridge / MOCHI` 넷만 비교해도 정상적으로 돕니다.
맨 윗줄에 `비교 대상 N개: ...` 이 찍히니 거기서 확인하시면 됩니다.

### 보는 곳

`C=` 값(C-index)입니다. **0.5면 무작위**, 높을수록 생존 예측이 잘 되는 것입니다.
`MOCHI` 줄이 `mean`·`Ridge` 줄보다 높으면, 채운 값이 임상적으로 의미 있는
구조를 보존한다는 뜻입니다. **정확도에서 지더라도 이 축에서 이기면 논문이
훨씬 좋아집니다.**

## 보내주실 것

```
~/task1.log ~/task2.log ~/task3.log ~/task4.log ~/task5.log
results/current/lr_stage5c/cell_rate.tsv
results/current/baselines_*.tsv
results/current/lr_ctl/*/*_s*/train.log
results/current/lr_nsweep/*/*/train.log
```

각 로그 끝의 **요약표만 먼저 붙여주셔도** 판단은 가능합니다.

---

## 문제가 생기면

| 증상 | 대처 |
|---|---|
| `git checkout` 또는 `git pull` 이 거부됨 | 로컬 수정이 남아 있습니다. `git stash` 후 다시 |
| 검증(`test_patches.py`) 실패 | **돌리지 말고 알려주세요** |
| `빠진 가중치` / `예상 밖 가중치` | 코드가 옛 버전입니다. 0번을 다시 확인 |
| `경로 없음` 으로 건너뜀 | 그 조건이 학습되지 않았다는 뜻입니다. 나머지는 계속 돕니다 |
| `invalid device ordinal` | GPU가 하나뿐입니다. 작업 3은 `GPU=0 bash scripts/train_ablate_p0.sh` |
| 작업 2에서 통째 결측 값이 전부 비슷 | 정상입니다. 로그의 `정상:` 줄을 보시면 됩니다 |
| 작업 2 로그에 `Mean of empty slice` | 옛 코드입니다. 0번을 다시 받아 주세요. 지금은 그 경고 없이 0으로 채웁니다 |
| CUDA 메모리 부족 | 알려주시면 배치 크기를 줄이겠습니다 |
| 어떤 표의 숫자가 **완전히 동일** | 알려주세요. 설정이 안 먹었을 수 있습니다 |

---

## 이 다섯이 왜 지금인가

논문 설계가 끝났고, 본문에 들어갈 표와 그림 중 **아직 숫자가 없는 칸이
정확히 이 다섯 개**입니다. 이게 채워지면 바로 원고를 씁니다.

그리고 작업 3은 **거절 사유 하나를 GPU 두 장으로 30–40분에 없애는 일**입니다.
다른 걸 못 돌리시더라도 이건 꼭 부탁드립니다.
