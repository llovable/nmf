#!/usr/bin/env bash
# 보고 앵커(ctl_base)와 NMF 채널 대조군. 전부 GAN off.
#   ctl_base : --aux_w_only --w_from_others --lambda_w 2.0 --lambda_nmf 0.1 --gan_to_mse 0
#   w0       : λ_w=0
#   nmf0     : λ_nmf=0
# 옛 results/current/lr_auxw/ 는 gan_to_mse 기본 0.1 로 돌아간 런이라 은퇴.
set -euo pipefail
cd /home/dyan/nmf/mochi_code
# shellcheck disable=SC1091
source /home/dyan/nmf/.nmf/bin/activate
export OMP_NUM_THREADS=8
ROOT=results/current/lr_ctl

train_one() {
  local cond=$1 cohort=$2 gpu=$3
  local extra=()
  case $cond in
    ctl_base) extra=(--lambda_w 2.0 --lambda_nmf 0.1) ;;
    w0) extra=(--lambda_w 0.0 --lambda_nmf 0.1) ;;
    nmf0) extra=(--lambda_w 2.0 --lambda_nmf 0.0) ;;
    *) echo "unknown cond $cond" >&2; return 1 ;;
  esac
  local save=$ROOT/$cond/${cohort}_hybrid
  mkdir -p "$save"
  echo "$(date) === $cond $cohort GPU=$gpu ${extra[*]} gan_to_mse=0 ==="
  python -u train_nmf_tf.py \
    --data_dir processed_data/gate_$cohort \
    --save_dir "$save" \
    --gpu "$gpu" \
    --w_from_others \
    --aux_w_only \
    --gan_to_mse 0 \
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
# BRCA 앵커와 두 대조군을 먼저. GPU 2장이라 앵커+ w0 다음 nmf0.
train_one ctl_base brca 0 &
train_one w0 brca 1 &
wait
train_one nmf0 brca 0
eval_one ctl_base brca 0 &
eval_one w0 brca 1 &
wait
eval_one nmf0 brca 0

train_one ctl_base luad 0 &
train_one w0 luad 1 &
wait
train_one nmf0 luad 0

train_one ctl_base kirc 0 &
train_one w0 kirc 1 &
wait
train_one nmf0 kirc 0

eval_one ctl_base luad 0 &
eval_one w0 luad 1 &
eval_one nmf0 luad 0
wait
eval_one ctl_base kirc 0 &
eval_one w0 kirc 1 &
wait
eval_one nmf0 kirc 0

echo "$(date) ALL DONE"
echo "앵커: $ROOT/ctl_base  (λ_w=2.0 λ_nmf=0.1 gan_to_mse=0)"
echo "대조: $ROOT/{w0,nmf0}"
