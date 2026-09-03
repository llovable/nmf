#!/usr/bin/env bash
# P0: aux_w_only 고정, NMF가 들어오는 두 채널을 하나씩 끈다.
#   w0   : λ_w=0     (계수 보조 손실 off, λ_nmf=0.1 유지)
#   nmf0 : λ_nmf=0   (출력단 NMF 정규화 off, λ_w=2.0 유지 — 보고 런과 같음)
# 그 외는 lr_auxw 와 동일: d_model=128, Linear AE, --w_from_others, gan_to_mse 기본 0.1
set -euo pipefail
cd /home/dyan/nmf/mochi_code
# shellcheck disable=SC1091
source /home/dyan/nmf/.nmf/bin/activate
export OMP_NUM_THREADS=8
ROOT=results/current/lr_ablate

train_one() {
  local cond=$1 cohort=$2 gpu=$3
  local extra=()
  case $cond in
    w0) extra=(--lambda_w 0.0 --lambda_nmf 0.1) ;;
    nmf0) extra=(--lambda_w 2.0 --lambda_nmf 0.0) ;;
    *) echo "unknown cond $cond" >&2; return 1 ;;
  esac
  local save=$ROOT/$cond/${cohort}_hybrid
  mkdir -p "$save"
  echo "$(date) === $cond $cohort GPU=$gpu ${extra[*]} ==="
  python -u train_nmf_tf.py \
    --data_dir processed_data/gate_$cohort \
    --save_dir "$save" \
    --gpu "$gpu" \
    --w_from_others \
    --aux_w_only \
    "${extra[@]}" \
    | tee "$save/train.log"
}

eval_one() {
  local cond=$1 cohort=$2 gpu=$3
  local save=$ROOT/$cond/${cohort}_hybrid
  local mimir=results/current/mimir
  [ "$cohort" = brca ] || mimir=results/current/mimir_$cohort
  local clin_args=(--clinical "")
  [ "$cohort" = brca ] && clin_args=()
  echo "$(date) === eval $cond $cohort ==="
  python -u eval_amplitude.py \
    --data_dir processed_data/gate_$cohort \
    "${clin_args[@]}" \
    --mimir_dir "$mimir" \
    --out_dir "$ROOT/$cond/amp_$cohort" \
    --gpu "$gpu" \
    --runs "MOCHI=$save/nmf_tf_best.ckpt" \
    | tee "$ROOT/$cond/amp_$cohort.log"
}

mkdir -p "$ROOT"
# BRCA를 두 GPU에 동시에 걸어 보고 문장을 먼저 가른다.
train_one w0 brca 0 &
pid_a=$!
train_one nmf0 brca 1 &
pid_b=$!
wait $pid_a $pid_b
eval_one w0 brca 0 &
eval_one nmf0 brca 1 &
wait

train_one w0 luad 0 &
pid_a=$!
train_one nmf0 luad 1 &
pid_b=$!
wait $pid_a $pid_b

train_one w0 kirc 0 &
pid_a=$!
train_one nmf0 kirc 1 &
pid_b=$!
wait $pid_a $pid_b

eval_one w0 luad 0 &
eval_one nmf0 luad 1 &
eval_one w0 kirc 0
wait
eval_one nmf0 kirc 1

echo "$(date) ALL DONE"
echo "대조: results/current/lr_auxw/  (λ_w=2.0 λ_nmf=0.1)"
echo "결과: $ROOT"
