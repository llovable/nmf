#!/usr/bin/env bash
# 학습 환자 수를 줄여가며 채움 오차 곡선을 그린다.
#
# 왜: "모델 크기가 병목이 아니라 환자 수가 병목"이라는 주장의 근거가 지금은
# 음성 결과 하나뿐이다(키웠더니 나빠짐). 그마저 교란돼 있다. 441명에서도
# 곡선이 가파르게 떨어지는 중이면, 그 주장이 그림 한 장으로 증명된다.
#
# 설정은 보고 모형과 같다 — 저랭크 잔차 없음, 출력단 NMF 손실 0, GAN 0,
# 표현 쪼갬, 통째 결측은 평균 융합, 재료 토큰 없음.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES="${RES:-$ROOT/results/current/lr_nsweep}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
COHORTS="${COHORTS:-brca luad kirc}"
SEEDS="${SEEDS:-0 1 2}"
NS="${NS:-100 200 300 0}"      # 0 = 전체
cd "$ROOT"
export PYTHONUNBUFFERED=1

"$PY" - <<'PYCHK'
import sys
try:
    import torch
except ImportError:
    sys.exit("torch 없음")
if not torch.cuda.is_available():
    sys.exit(f"CUDA 없음 (torch {torch.__version__}). CPU로는 돌리지 않는다.")
print(f"cuda {torch.cuda.get_device_name(0)}")
PYCHK

COMMON=(--gpu "$GPU" --gamma_nonneg --w_head_act softplus --gan_to_mse 0
        --aux_w_only --w_from_others --lambda_w 2.0 --lambda_nmf 0 --detach_w_head
        --split_latent --block_mean --no_nmf_tokens)

for coh in $COHORTS; do
  DATA="$ROOT/processed_data/gate_$coh"
  [[ -f "$DATA/rna.train.tsv" ]] || { echo "skip $coh: $DATA 없음"; continue; }
  TOTAL=$(( $(head -1 "$DATA/rna.train.tsv" | tr '\t' '\n' | wc -l) - 1 ))
  echo "=== $coh  학습 환자 $TOTAL 명 ==="
  for n in $NS; do
    if [[ "$n" != "0" && "$n" -ge "$TOTAL" ]]; then
      echo "  skip n=$n (전체 $TOTAL 이상)"; continue
    fi
    for sd in $SEEDS; do
      tag=$([[ "$n" == "0" ]] && echo full || echo "n$n")
      dir="$RES/${tag}/${coh}_s${sd}"
      [[ -f "$dir/nmf_tf_best.ckpt" ]] && { echo "  skip $tag $coh seed$sd"; continue; }
      mkdir -p "$dir"
      echo "  ----- $coh $tag seed=$sd -----"
      "$PY" -u train_nmf_tf.py "${COMMON[@]}" --n_train "$n" \
        --data_dir "$DATA" --save_dir "$dir" --seed "$sd" 2>&1 | tee "$dir/train.log"
    done
  done
done

echo
echo "=== 환자 수별 통째 결측 채움 오차 (반복 평균 ± 흔들림) ==="
"$PY" - "$RES" <<'PYSUM'
import sys, glob, os
import numpy as np, torch
rows = {}
for ck in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "nmf_tf_best.ckpt"))):
    parts = ck.split(os.sep)
    tag, run = parts[-3], parts[-2]
    coh = run.rsplit("_s", 1)[0]
    d = torch.load(ck, map_location="cpu", weights_only=False)
    rows.setdefault((coh, tag), []).append(
        (d.get("n_train"), d.get("val", {}).get("avg", float("nan"))))
if not rows:
    sys.exit("체크포인트 없음")
print(f"{'암종':<8}{'학습 환자':>10}{'채움 오차':>22}")
for (coh, tag) in sorted(rows, key=lambda t: (t[0], rows[t][0][0] or 10**9)):
    vals = [v for _, v in rows[(coh, tag)]]
    n = rows[(coh, tag)][0][0]
    print(f"{coh:<8}{str(n):>10}{f'{np.mean(vals):.4f}±{np.std(vals):.4f}':>22}")
print()
print("읽는 법: 전체 환자 수에서도 곡선이 가파르면 표본이 병목이다.")
print("         평평해졌으면 환자를 더 모아도 안 좋아진다는 뜻이고, 그때는")
print("         구조나 특징 선택 쪽을 봐야 한다.")
PYSUM
