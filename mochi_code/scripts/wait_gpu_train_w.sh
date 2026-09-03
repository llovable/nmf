#!/usr/bin/env bash
# vLLM 등이 GPU를 놓을 때까지 기다렸다가, 다른 오믹스 은닉 → W 경로로 재학습한다.
# 성공선: BRCA RNA 블록 z-RMSE ≤ 0.70, 경로 k-means ARI ≥ 0.58
set -euo pipefail
cd /home/dyan/nmf/mochi_code
# shellcheck disable=SC1091
source /home/dyan/nmf/.nmf/bin/activate
export OMP_NUM_THREADS=8
FREE_NEED="${FREE_NEED:-20000}"
POLL="${POLL:-30}"
LOGDIR=results/current/lr_wothers
mkdir -p "$LOGDIR"

pick_gpu() {
  local idx free util
  while IFS=',' read -r idx free util; do
    idx=${idx// /}
    free=${free// /}
    util=${util// /}
    if [ "$free" -ge "$FREE_NEED" ] && [ "$util" -le 30 ]; then
      echo "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

echo "$(date) GPU ${FREE_NEED} MiB 빈 자리 대기 (poll ${POLL}s)"
GPU=""
while true; do
  if GPU=$(pick_gpu); then
    echo "$(date) GPU $GPU 확보"
    break
  fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
  sleep "$POLL"
done
export CUDA_VISIBLE_DEVICES="$GPU"

run_one() {
  local cohort=$1
  local data=processed_data/gate_$cohort
  local save=$LOGDIR/${cohort}_hybrid
  local mimir=results/current/mimir
  [ "$cohort" = brca ] || mimir=results/current/mimir_$cohort
  mkdir -p "$save"
  echo "$(date) === train $cohort GPU=$GPU ==="
  python -u train_nmf_tf.py \
    --data_dir "$data" \
    --save_dir "$save" \
    --gpu 0 \
    --w_from_others \
    --freeze_protein_gamma \
    --lambda_w 2.0 \
    | tee -a "$save/train.log"
  echo "$(date) === eval amplitude $cohort ==="
  local clin_args=(--clinical "")
  if [ "$cohort" = brca ]; then
    clin_args=()
  fi
  python -u eval_amplitude.py \
    --data_dir "$data" \
    "${clin_args[@]}" \
    --mimir_dir "$mimir" \
    --out_dir "$LOGDIR/amp_$cohort" \
    --gpu 0 \
    --runs "MOCHI=$save/nmf_tf_best.ckpt" \
           "MOCHI-knockout=$save/nmf_tf_best.ckpt=gamma0" \
    | tee -a "$LOGDIR/amp_$cohort.log"
}

run_one brca
run_one luad
run_one kirc
echo "$(date) ALL DONE"
echo "통과선: BRCA RNA 블록 zRMSE<=0.70  ARI>=0.58"
echo "결과: $LOGDIR"
