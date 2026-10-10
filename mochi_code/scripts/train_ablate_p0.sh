#!/usr/bin/env bash
# 보고 앵커와 NMF 채널 대조군. 전부 GAN off, --w_from_others.
# 앞 다섯은 --aux_w_only (디코더에 γWH를 더하지 않음):
#   ctl_base   : λ_w=2.0 λ_nmf=0.1  토큰 on   — 앵커
#   w0         : λ_w=0   λ_nmf=0.1  토큰 on   — 계수 보조 손실만 제거
#   nmf0       : λ_w=2.0 λ_nmf=0    토큰 on   — 출력단 NMF 정규화만 제거
#   lw0_nmf0   : λ_w=0   λ_nmf=0    토큰 on   — 손실 두 채널 off, 토큰은 남음
#   none       : λ_w=0   λ_nmf=0    토큰 off  — NMF 사전지식 전부 제거
# 뒤 둘은 잔차 on (--aux_w_only 없음). 원고의 핵심 비교는 "잔차 대 보조 손실"인데
# aux 쪽만 있으면 절반이 빈다. 옛 잔차 숫자(ARI 0.252)는 gan_to_mse=0.1 런이라
# GAN=0인 이 표와 같은 축에 놓을 수 없다:
#   resid      : λ_w=2.0 λ_nmf=0.1  토큰 on   — ctl_base 와 짝. λ_nmf 맞춘 잔차 효과
#   resid_nmf0 : λ_w=2.0 λ_nmf=0    토큰 on   — nmf0 와 짝. λ_nmf 없는 잔차 효과
# 옛 results/current/lr_auxw/ 는 gan_to_mse 기본 0.1 런이라 은퇴.
set -euo pipefail
cd /home/dyan/nmf/mochi_code
# shellcheck disable=SC1091
source /home/dyan/nmf/.nmf/bin/activate
export OMP_NUM_THREADS=8
ROOT=results/current/lr_ctl
# SEEDS 를 주면 시드별로 돌리고 경로에 _s<시드> 를 붙인다. 기존 _hybrid 결과는
# 2026-09-03 단일 시드 런이라 그대로 둔다. 논문 표는 시드 3개 쪽을 쓴다.
SEEDS="${SEEDS:-0 1 2}"
# 쓸 GPU. 두 개를 주면 조건 두 개가 동시에 돈다. 하나만 주면 순차로 돈다.
# 예전에는 0과 1이 코드에 박혀 있어서 GPU 한 장짜리 장비에서 그냥 죽었다.
GPUS="${GPUS:-0 1}"
COHORTS="${COHORTS:-brca luad kirc}"
set -- $GPUS
G0="$1"; G1="${2:-$1}"

train_one() {
  local cond=$1 cohort=$2 gpu=$3
  local extra=()
  case $cond in
    ctl_base) extra=(--aux_w_only --lambda_w 2.0 --lambda_nmf 0.1) ;;
    w0) extra=(--aux_w_only --lambda_w 0.0 --lambda_nmf 0.1) ;;
    nmf0) extra=(--aux_w_only --lambda_w 2.0 --lambda_nmf 0.0) ;;
    lw0_nmf0) extra=(--aux_w_only --lambda_w 0.0 --lambda_nmf 0.0) ;;
    none) extra=(--aux_w_only --lambda_w 0.0 --lambda_nmf 0.0 --no_nmf_tokens) ;;
    resid) extra=(--lambda_w 2.0 --lambda_nmf 0.1) ;;
    resid_nmf0) extra=(--lambda_w 2.0 --lambda_nmf 0.0) ;;
    *) echo "unknown cond $cond" >&2; return 1 ;;
  esac
  local sd=${4:-0}
  local save=$ROOT/$cond/${cohort}_s${sd}
  mkdir -p "$save"
  echo "$(date) === $cond $cohort seed=$sd GPU=$gpu ${extra[*]} gan_to_mse=0 ==="
  python -u train_nmf_tf.py \
    --data_dir processed_data/gate_$cohort \
    --save_dir "$save" \
    --gpu "$gpu" \
    --w_from_others \
    --gan_to_mse 0 \
    --seed "$sd" \
    "${extra[@]}" \
    | tee "$save/train.log"
}

eval_one() {
  local cond=$1 cohort=$2 gpu=$3
  local sd=${4:-0}
  local save=$ROOT/$cond/${cohort}_s${sd}
  local mimir=results/current/mimir
  [ "$cohort" = brca ] || mimir=results/current/mimir_$cohort
  local clin_args=(--clinical "")
  [ "$cohort" = brca ] && clin_args=()
  echo "$(date) === eval $cond $cohort seed=$sd ==="
  python -u eval_amplitude.py \
    --data_dir processed_data/gate_$cohort \
    "${clin_args[@]}" \
    --mimir_dir "$mimir" \
    --out_dir "$ROOT/$cond/amp_${cohort}_s${sd}" \
    --gpu "$gpu" \
    --runs "MOCHI=$save/nmf_tf_best.ckpt" \
    | tee "$ROOT/$cond/amp_${cohort}_s${sd}.log"
}

pair() {   # pair <train_one|eval_one> <조건A> <조건B> <암종> <시드>
  # GPU 가 둘이면 동시에, 하나면 순차로. 예전에는 0과 1이 박혀 있어서
  # 카드 한 장짜리 장비에서 cuda:1 을 잡다가 그냥 죽었다.
  local fn=$1 a=$2 b=$3 coh=$4 sd=$5
  if [[ "$G0" == "$G1" ]]; then
    "$fn" "$a" "$coh" "$G0" "$sd"
    "$fn" "$b" "$coh" "$G0" "$sd"
  else
    "$fn" "$a" "$coh" "$G0" "$sd" &
    "$fn" "$b" "$coh" "$G1" "$sd" &
    wait
  fi
}

run_cohort() {
  local cohort=$1 sd=${2:-0}
  pair train_one ctl_base   w0         "$cohort" "$sd"
  pair train_one nmf0       lw0_nmf0   "$cohort" "$sd"
  pair train_one none       resid      "$cohort" "$sd"
  train_one resid_nmf0 "$cohort" "$G0" "$sd"
  pair eval_one  ctl_base   w0         "$cohort" "$sd"
  pair eval_one  nmf0       lw0_nmf0   "$cohort" "$sd"
  pair eval_one  none       resid      "$cohort" "$sd"
  eval_one resid_nmf0 "$cohort" "$G0" "$sd"
}

mkdir -p "$ROOT"
for sd in $SEEDS; do
  for coh in $COHORTS; do
    run_cohort "$coh" "$sd"
  done
done

echo "$(date) ALL DONE"
echo "앵커: $ROOT/ctl_base  (λ_w=2.0 λ_nmf=0.1 tokens on gan_to_mse=0)"
echo "대조: $ROOT/{w0,nmf0,lw0_nmf0,none}"
echo "잔차: $ROOT/{resid,resid_nmf0}  (ctl_base / nmf0 와 각각 짝)"
