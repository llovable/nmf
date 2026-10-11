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
python -c "import lifelines" 2>/dev/null || pip install 'lifelines>=0.27.0'
cd mochi_code && python tests/test_patches.py
```

마지막 줄이 `모든 검증 통과` 여야 합니다. 통과 안 하면 **돌리지 말고 알려주세요.**

### GPU 확인

학습(작업 3·4)은 **GPU에서만 돕니다.** torch가 CPU 휠로 깔려 있으면
예전에는 말없이 CPU로 떨어져 몇 분짜리가 몇 시간이 됐습니다. 이제는
거기서 멈추게 해 뒀습니다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`True` 와 카드 수가 나와야 합니다. `False` 면 torch를 다시 까세요.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

평가만 하는 작업 1·2·5는 CPU로도 돌아가지만 느립니다. 그때는
`!! CUDA 가 없어 CPU 로 돕니다` 경고가 찍힙니다.

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

**모델 두 개**를 각각 잽니다. 논문에서 어느 쪽을 보고할지 아직 정하지
않았기 때문입니다 — 네 경로 해부의 "전부 끔" 조건과 사다리 끝 모델입니다.
한쪽만 재면 나중에 다시 돌려야 합니다.

아래는 있는 체크포인트만 알아서 골라 돕니다. 판단하실 것 없습니다.

```bash
cd /home/dyan/nmf/mochi_code
source /home/dyan/nmf/.nmf/bin/activate
for COH in brca luad kirc; do
  # 해부 실험의 "전부 끔". 작업 3을 돌렸으면 _s0, 아니면 9월 _hybrid.
  ABL=results/current/lr_ctl/none/${COH}_s0
  [[ -f "$ABL/nmf_tf_best.ckpt" ]] || ABL=results/current/lr_ctl/none/${COH}_hybrid
  # 사다리 끝 모델.
  LAD=results/current/lr_stage5c/c1/${COH}_s0
  for CK in "$ABL" "$LAD"; do
    if [[ ! -f "$CK/nmf_tf_best.ckpt" ]]; then echo "없음, 건너뜀: $CK"; continue; fi
    echo "===== $COH  $(basename "$(dirname "$CK")") ====="
    python eval_baselines.py \
      --data_dir processed_data/gate_$COH \
      --ckpt "$CK/nmf_tf_best.ckpt" \
      --split test --rate 0.3 --gpu 0 \
      --out "results/current/baselines_${COH}_$(basename "$(dirname "$CK")").tsv"
  done
done 2>&1 | tee ~/task2.log
```

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

이 명령은 GPU 0과 1을 같이 씁니다. **프로세스를 둘 띄우지 마세요** —
네 작업이 두 카드에 몰립니다.

**카드가 한 장이면** 0번만 지정합니다. 조건이 순차로 돌아 약 두 배 걸립니다.

```bash
GPU=0 bash scripts/train_ablate_p0.sh 2>&1 | tee ~/task3.log
```

그 두 배가 부담스러우면 암종을 갈라 두 프로세스로 돌리세요.

```bash
COHORTS="brca"      GPU=0 bash scripts/train_ablate_p0.sh > ~/task3_brca.log 2>&1 &
COHORTS="luad kirc" GPU=0 bash scripts/train_ablate_p0.sh > ~/task3_rest.log 2>&1 &
wait
cat ~/task3_brca.log ~/task3_rest.log > ~/task3.log
```

두 로그 끝에 `ALL DONE` 이 있어야 합니다. 한쪽만 있으면 그 로그를 보내주세요.

> 한 카드에 두 작업이 올라가므로 메모리를 두 배 씁니다. 모델이 작아
> (약 540만 개) 보통은 들어가지만, `out of memory` 가 나면 그냥 위의
> 순차 실행으로 돌리시면 됩니다.


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
집니다.** 생존 예측은 그 다음 축입니다. 정확도에서는 이미 졌더라도 채운
값이 예후를 보존하면, "정확도는 뒤져도 임상적으로 의미 있는 구조를
보존한다"가 됩니다. 그 문장은 순환 논증이 아닙니다.

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
  --runs "MOCHI-abl=results/current/lr_ctl/none/brca_s0/nmf_tf_best.ckpt" \
         "MOCHI-abl-old=results/current/lr_ctl/none/brca_hybrid/nmf_tf_best.ckpt" \
         "MOCHI-ladder=results/current/lr_stage5c/c1/brca_s0/nmf_tf_best.ckpt" \
  2>&1 | tee ~/task5.log
```

그다음 생존 분석입니다. `--runs` 는 하위 과제와 같은 `이름=경로` 형식이고,
몇 개든 넣을 수 있습니다. 타깃 오믹스를 통째로 가린 뒤 그 체크포인트로
복원하고, 채운 행렬로 생존을 예측합니다.

생존 테이블은 임상 행렬과 다른 파일입니다. 없으면 먼저 받으세요.
이미 있는 파일은 건너뜁니다.

```bash
cd /home/dyan/nmf/mochi_code
bash scripts/fetch_xena_cohort.sh BRCA
```

```bash
source /home/dyan/nmf/.nmf/bin/activate
python eval_survival.py \
  --data_dir processed_data/gate_brca \
  --surv processed_data/clinical/TCGA-BRCA.survival.tsv.gz \
  --out_dir results/current/surv_final --gpu 0 \
  --runs "MOCHI-abl=results/current/lr_ctl/none/brca_s0/nmf_tf_best.ckpt" \
         "MOCHI-abl-old=results/current/lr_ctl/none/brca_hybrid/nmf_tf_best.ckpt" \
         "MOCHI-ladder=results/current/lr_stage5c/c1/brca_s0/nmf_tf_best.ckpt" \
  2>&1 | tee -a ~/task5.log
```

위의 세 경로 중 파일이 없는 것은 건너뜁니다. 작업 3을 안 돌렸어도
9월 `_hybrid` 와 사다리 체크포인트는 그대로 잽니다.

옛 비교 모델 넷(MOCHI-v5 1→1, MOCHI-v5 2→1, 공식 OmicsNMF, 공식 OmiTrans)은
**파일이 있을 때만** 쓰고 없으면 건너뜁니다. 그래서 그 파일이 없어도
`mean` / `TOBMI` / `Ridge` / `MOCHI` 는 돕니다. 맨 윗줄의
`비교 대상 N개:` 에서 확인하시면 됩니다.

### 보는 곳

`C=` 값이 C-index 입니다. **0.5면 무작위**이고, 높을수록 생존 예측이
잘 되는 것입니다. `MOCHI` 줄이 `mean`·`Ridge` 보다 높으면, 채운 값이
임상적으로 의미 있는 구조를 보존한다는 뜻입니다. 정확도에서 지더라도
이 축에서 이기면 논문이 훨씬 좋아집니다.

---

## 보내주실 것

```
~/task1.log ~/task2.log ~/task3.log ~/task4.log ~/task5.log
results/current/lr_stage5c/cell_rate.tsv
results/current/baselines_*.tsv
results/current/lr_ctl/*/*_s*/train.log
results/current/lr_nsweep/*/*/train.log
results/current/down_final/downstream_auc.tsv
results/current/surv_final/survival.tsv
```

폴더째 묶어 보내주셔도 됩니다.

```bash
cd /home/dyan/nmf/mochi_code
tar czf ~/mochi_results.tgz \
  results/current/lr_stage5c/cell_rate.tsv \
  results/current/baselines_*.tsv \
  results/current/lr_ctl/*/*_s*/train.log \
  results/current/lr_nsweep \
  results/current/down_final results/current/surv_final \
  2>/dev/null; ls -lh ~/mochi_results.tgz
```

각 로그 끝의 **요약표만 먼저 붙여주셔도** 판단은 가능합니다.

---

## 문제가 생기면

| 증상 | 대처 |
|---|---|
| `git checkout` 또는 `git pull` 이 거부됨 | 로컬 수정이 남아 있습니다. `git stash` 후 다시 |
| 검증(`test_patches.py`) 실패 | **돌리지 말고 알려주세요** |
| `빠진 가중치` / `예상 밖 가중치` | 코드가 옛 버전입니다. 0번을 다시 확인 |
| `No module named lifelines` | `pip install 'lifelines>=0.27.0'` 후 작업 5만 다시 |
| 작업 5 로그의 `[건너뜀] MOCHI-v5` / `OmicsNMF` | 정상입니다. 옛 파일이 없는 것이고 mean·TOBMI·Ridge·MOCHI는 계속 돕니다 |
| `없음, 건너뜀` / `[건너뜀]` / `skip ... 없음` | 그 조건이 학습되지 않았다는 뜻입니다. 나머지는 계속 돕니다 |
| `invalid device ordinal` | GPU가 하나뿐입니다. 작업 3은 `GPU=0 bash scripts/train_ablate_p0.sh` |
| `CUDA 없음` / `CUDA 가 없습니다` | torch가 CPU 휠입니다. 0번의 GPU 확인을 보세요 |
| `!! CUDA 가 없어 CPU 로 돕니다` | 평가는 계속 돕니다. 느리면 GPU를 고치세요 |
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
