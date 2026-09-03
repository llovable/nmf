#!/usr/bin/env bash
# 전략 1: NMF 계수는 보조 손실만, 디코더에 γWH를 더하지 않는다.
set -euo pipefail
cd /home/dyan/nmf/mochi_code
# shellcheck disable=SC1091
source /home/dyan/nmf/.nmf/bin/activate
export OMP_NUM_THREADS=8
LOGDIR=results/current/lr_auxw
mkdir -p "$LOGDIR"

train_one() {
  local cohort=$1 gpu=$2
  local save=$LOGDIR/${cohort}_hybrid
  mkdir -p "$save"
  echo "$(date) === train $cohort GPU=$gpu aux_w_only ==="
  python -u train_nmf_tf.py \
    --data_dir processed_data/gate_$cohort \
    --save_dir "$save" \
    --gpu "$gpu" \
    --w_from_others \
    --aux_w_only \
    --lambda_w 2.0 \
    | tee "$save/train.log"
}

eval_one() {
  local cohort=$1 gpu=$2
  local save=$LOGDIR/${cohort}_hybrid
  local mimir=results/current/mimir
  [ "$cohort" = brca ] || mimir=results/current/mimir_$cohort
  local clin_args=(--clinical "")
  [ "$cohort" = brca ] && clin_args=()
  echo "$(date) === eval amplitude $cohort ==="
  python -u eval_amplitude.py \
    --data_dir processed_data/gate_$cohort \
    "${clin_args[@]}" \
    --mimir_dir "$mimir" \
    --out_dir "$LOGDIR/amp_$cohort" \
    --gpu "$gpu" \
    --runs "MOCHI=$save/nmf_tf_best.ckpt" \
           "MOCHI-knockout=$save/nmf_tf_best.ckpt=gamma0" \
    | tee "$LOGDIR/amp_$cohort.log"
}

train_one brca 0 &
pid_b=$!
train_one luad 1 &
pid_l=$!
wait $pid_b $pid_l
train_one kirc 0
eval_one brca 0
eval_one luad 1 &
eval_one kirc 0
wait
echo "$(date) ALL DONE"
echo "통과선: BRCA RNA 블록 zRMSE<=0.70  ARI>=0.58 (평균 k=2..5)"
echo "결과: $LOGDIR"
