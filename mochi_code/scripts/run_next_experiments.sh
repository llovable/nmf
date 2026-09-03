#!/usr/bin/env bash
# 보고 모형 재학습 → λ_W 스윕 → 표 7–9 재평가. GPU 필수.
# 저장소 어디에 클론했든 이 스크립트 위치(mochi_code/)를 ROOT로 쓴다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$ROOT/processed_data/gate_brca}"
CLIN="${CLIN:-$ROOT/processed_data/clinical}"
RES="${RES:-$ROOT/results/current}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
cd "$ROOT"
export PYTHONUNBUFFERED=1
echo "ROOT=$ROOT"

"$PY" - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("torch 없음")
if not torch.cuda.is_available():
    sys.exit(f"CUDA 없음 (torch {torch.__version__}). CPU로는 돌리지 않는다.")
print(f"cuda {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

need() { [[ -f "$1" ]] || { echo "missing $1"; exit 1; }; }
need "$DATA/rna.train.tsv"
need "$DATA/protein.train.tsv"
need "$DATA/methy.train.tsv"

COMMON=(--data_dir "$DATA" --gpu "$GPU" --gamma_nonneg --w_head_act softplus --gan_to_mse 0)

train_one() {
  local tag="$1"; shift
  local dir="$RES/$tag"
  if [[ -f "$dir/nmf_tf_best.ckpt" ]]; then
    echo "skip train $tag (ckpt exists)"
    return
  fi
  echo "===== train $tag ====="
  mkdir -p "$dir"
  "$PY" -u train_nmf_tf.py "${COMMON[@]}" --save_dir "$dir" "$@" 2>&1 | tee "$dir/train.log"
}

# 1) 새 보고 모형. phase1 AE를 스윕이 재사용한다.
train_one "lr_brca_nogan_sp" --lambda_w 0.3
AE="$RES/lr_brca_nogan_sp/ae_phase1.pt"

# 2) λ_W 스윕. 같은 AE.
train_one "lr_brca_lw15" --lambda_w 1.5 --ae_ckpt "$AE"
train_one "lr_brca_lw30" --lambda_w 3.0 --ae_ckpt "$AE"

# 3) 표 7 제거 실험 (보고 모형과 같은 머리·GAN=0).
train_one "lr_brca_notf" --lambda_w 0.3 --no_transformer --ae_ckpt "$AE"
train_one "lr_brca_nonmf" --lambda_w 0.3 --no_nmf_tokens --ae_ckpt "$AE"
train_one "lr_brca_n100" --lambda_w 0.3 --n_train 100
# GAN을 켠 대조. 보고 모형이 GAN=0이므로 표의 "GAN 있음" 행.
train_one "lr_brca_gan01" --lambda_w 0.3 --gan_to_mse 0.1 --ae_ckpt "$AE"

REP="$RES/lr_brca_nogan_sp/nmf_tf_best.ckpt"
need "$REP"

wstat() {
  local ckpt="$1" out="$2"
  [[ -f "$ckpt" ]] || return 0
  "$PY" -u eval_w_stats.py --data_dir "$DATA" --ckpt "$ckpt" --out "$out" --gpu "$GPU"
}
wstat "$REP" "$RES/wstats_nogan_sp.tsv"
wstat "$RES/lr_brca_lw15/nmf_tf_best.ckpt" "$RES/wstats_lw15.tsv"
wstat "$RES/lr_brca_lw30/nmf_tf_best.ckpt" "$RES/wstats_lw30.tsv"

# 표 5에 해당하는 ablation (보고 모형 + ω=0 + 녹아웃 + notf/nonmf/gan).
"$PY" -u eval_ablation.py --data_dir "$DATA" --gpu "$GPU" --out_dir "$RES/abl_brca_sp" --runs \
  "full=$REP" \
  "knockout=$REP=gamma0" \
  "gan01=$RES/lr_brca_gan01/nmf_tf_best.ckpt" \
  "mean=$RES/lr_brca_notf/nmf_tf_best.ckpt" \
  "nonmf=$RES/lr_brca_nonmf/nmf_tf_best.ckpt"

# 표 7–8 스트레스·분포. MIMIR이 있으면 같이.
STRESS_RUNS=("MOCHI=$REP")
[[ -f "$RES/lr_brca_gan01/nmf_tf_best.ckpt" ]] && STRESS_RUNS+=("MOCHI-gan=$RES/lr_brca_gan01/nmf_tf_best.ckpt")
[[ -f "$RES/lr_brca_notf/nmf_tf_best.ckpt" ]] && STRESS_RUNS+=("MOCHI-notf=$RES/lr_brca_notf/nmf_tf_best.ckpt")
[[ -f "$RES/lr_brca_nonmf/nmf_tf_best.ckpt" ]] && STRESS_RUNS+=("MOCHI-nonmf=$RES/lr_brca_nonmf/nmf_tf_best.ckpt")
MIMIR_ARGS=()
[[ -d "$RES/mimir" ]] && MIMIR_ARGS+=(--mimir_dir "$RES/mimir")
"$PY" -u eval_stress.py --data_dir "$DATA" --gpu "$GPU" --out_dir "$RES/stress_brca_sp" \
  "${MIMIR_ARGS[@]}" --runs "${STRESS_RUNS[@]}"

# 표 9 하위 과제.
if [[ -f "$CLIN/BRCA_clinicalMatrix" ]]; then
  "$PY" -u eval_downstream.py --data_dir "$DATA" --gpu "$GPU" --clinical "$CLIN/BRCA_clinicalMatrix" \
    --out_dir "$RES/down_brca_sp" "${MIMIR_ARGS[@]}" --runs "${STRESS_RUNS[@]}"
fi

# 생물학 (녹아웃 포함).
BIO_RUNS=("MOCHI=$REP" "MOCHI-knockout=$REP=gamma0")
BIO_EXTRA=()
[[ -f "$CLIN/gencode.probeMap" ]] && BIO_EXTRA+=(--probemap "$CLIN/gencode.probeMap")
[[ -f "$CLIN/hallmark.gmt" ]] && BIO_EXTRA+=(--gmt "$CLIN/hallmark.gmt")
[[ -f "$CLIN/BRCA_clinicalMatrix" ]] && BIO_EXTRA+=(--clinical "$CLIN/BRCA_clinicalMatrix")
"$PY" -u eval_biology.py --data_dir "$DATA" --gpu "$GPU" --out_dir "$RES/bio_brca_sp" \
  "${MIMIR_ARGS[@]}" "${BIO_EXTRA[@]}" --runs "${BIO_RUNS[@]}"

echo "pipeline done"
