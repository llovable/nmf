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
#
# GPU 를 비우면 0번과 1번을 짝지어 쓴다 (9월 스크립트와 같음).
# GPU=0 처럼 지정하면 그 장치에서만 한 조건씩 돈다.
# COHORTS 로 암종을 갈라 두 카드에 나눠 돌릴 수 있다. 카드가 한 장뿐이면
# GPU 지정이 순차 실행이라 작업 3이 두 배로 길어지는데, 이때 암종을 쪼개
# 두 프로세스로 돌리는 것이 유일한 병렬화 수단이다.
# 체크포인트가 있으면 학습을 건너뛴다. 저장 경로는 항상 _s<시드> 이고
# 2026-09-03 단일 시드 런인 _hybrid 는 읽지도 쓰지도 않는다.
# DRY=1 은 학습 없이 이 계약만 확인한다. DRY_ROOT 아래만 쓴다.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE"
VENV="${MOCHI_VENV:-/home/dyan/nmf/.nmf/bin/activate}"
if [[ -f "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "$VENV"
elif [[ "${DRY:-}" != "1" ]]; then
  echo "가상환경이 없습니다: $VENV" >&2
  exit 1
fi
export OMP_NUM_THREADS=8
ROOT=results/current/lr_ctl
if [[ "${DRY:-}" == "1" ]]; then
  : "${DRY_ROOT:?DRY=1 이면 DRY_ROOT 가 필요합니다}"
  ROOT="$DRY_ROOT"
fi
# SEEDS 를 주면 시드별로 돌리고 경로에 _s<시드> 를 붙인다. 기존 _hybrid 결과는
# 2026-09-03 단일 시드 런이라 그대로 둔다. 논문 표는 시드 3개 쪽을 쓴다.
SEEDS="${SEEDS:-0 1 2}"
COHORTS="${COHORTS:-brca luad kirc}"

dry_note() {
  [[ "${DRY:-}" == "1" ]] || return 0
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$ROOT/calls.tsv"
}

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
  if [[ -f "$save/nmf_tf_best.ckpt" ]]; then
    echo "$(date) === skip train $cond $cohort seed=$sd ==="
    return 0
  fi
  mkdir -p "$save"
  if [[ "${DRY:-}" == "1" && "${FAIL_COND:-}" == "$cond" ]]; then
    echo "DRY fail train $cond" >&2
    return 1
  fi
  echo "$(date) === train $cond $cohort seed=$sd GPU=$gpu ${extra[*]} gan_to_mse=0 ==="
  dry_note train "$cond" "$cohort" "$sd" "$gpu"
  if [[ "${DRY:-}" == "1" ]]; then
    echo dry > "$save/nmf_tf_best.ckpt"
    return 0
  fi
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
  local outdir=$ROOT/$cond/amp_${cohort}_s${sd}
  if [[ -f "$outdir/amplitude.tsv" ]]; then
    echo "$(date) === skip eval $cond $cohort seed=$sd ==="
    return 0
  fi
  local mimir=results/current/mimir
  [ "$cohort" = brca ] || mimir=results/current/mimir_$cohort
  local clin_args=(--clinical "")
  [ "$cohort" = brca ] && clin_args=()
  echo "$(date) === eval $cond $cohort seed=$sd GPU=$gpu ==="
  dry_note eval "$cond" "$cohort" "$sd" "$gpu"
  if [[ "${DRY:-}" == "1" ]]; then
    mkdir -p "$outdir"
    echo dry > "$outdir/amplitude.tsv"
    return 0
  fi
  python -u eval_amplitude.py \
    --data_dir processed_data/gate_$cohort \
    "${clin_args[@]}" \
    --mimir_dir "$mimir" \
    --out_dir "$outdir" \
    --gpu "$gpu" \
    --runs "MOCHI=$save/nmf_tf_best.ckpt" \
    | tee "$ROOT/$cond/amp_${cohort}_s${sd}.log"
}

# 두 조건을 돌린다. GPU가 지정되면 그 카드에서 순서대로, 아니면 0번과 1번에 하나씩.
# 백그라운드 한쪽이 죽으면 wait의 기본 반환값(0)에 묻히지 않게 상태를 확인한다.
pair() {
  local kind=$1 c1=$2 c2=$3 cohort=$4 sd=$5
  if [[ -n "${GPU:-}" ]]; then
    if [[ "$kind" == train ]]; then
      train_one "$c1" "$cohort" "$GPU" "$sd"
      train_one "$c2" "$cohort" "$GPU" "$sd"
    else
      eval_one "$c1" "$cohort" "$GPU" "$sd"
      eval_one "$c2" "$cohort" "$GPU" "$sd"
    fi
    return 0
  fi
  local p1 p2 s1=0 s2=0
  if [[ "$kind" == train ]]; then
    train_one "$c1" "$cohort" 0 "$sd" &
    p1=$!
    train_one "$c2" "$cohort" 1 "$sd" &
    p2=$!
  else
    eval_one "$c1" "$cohort" 0 "$sd" &
    p1=$!
    eval_one "$c2" "$cohort" 1 "$sd" &
    p2=$!
  fi
  wait "$p1" || s1=$?
  wait "$p2" || s2=$?
  if [[ $s1 -ne 0 || $s2 -ne 0 ]]; then
    echo "실패: $kind $c1=$s1 $c2=$s2" >&2
    exit 1
  fi
}

solo() {
  local kind=$1 cond=$2 cohort=$3 sd=$4
  local gpu="${GPU:-0}"
  if [[ "$kind" == train ]]; then
    train_one "$cond" "$cohort" "$gpu" "$sd"
  else
    eval_one "$cond" "$cohort" "$gpu" "$sd"
  fi
}

run_cohort() {
  local cohort=$1 sd=${2:-0}
  pair train ctl_base w0 "$cohort" "$sd"
  pair train nmf0 lw0_nmf0 "$cohort" "$sd"
  pair train none resid "$cohort" "$sd"
  solo train resid_nmf0 "$cohort" "$sd"
  pair eval ctl_base w0 "$cohort" "$sd"
  pair eval nmf0 lw0_nmf0 "$cohort" "$sd"
  pair eval none resid "$cohort" "$sd"
  solo eval resid_nmf0 "$cohort" "$sd"
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
echo "저장 경로는 <암종>_s<시드> 입니다. _hybrid 는 그대로 둡니다."
