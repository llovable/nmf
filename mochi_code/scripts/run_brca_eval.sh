#!/usr/bin/env bash
# BRCA --gamma_nonneg 체크포인트의 생물·녹아웃·자기정보 평가.
# MIMIR 체크포인트가 있으면 biology / eval_nmf_tf에 붙인다.
set -euo pipefail
cd /workspace/mochi_code
export PYTHONUNBUFFERED=1
CKPT=results/current/lr_brca_hybrid/nmf_tf_best.ckpt
DATA=processed_data/gate_brca
CLIN=processed_data/clinical
MIMIR_DIR=results/current/mimir
MIMIR_ARG=()
if [[ -f "$MIMIR_DIR/shared_best.pt" ]]; then
  MIMIR_ARG=(--mimir_dir "$MIMIR_DIR")
else
  MIMIR_ARG=(--mimir_dir "")
fi

python3 -u eval_biology.py \
  --data_dir "$DATA" \
  --out_dir results/current/bio_brca \
  --clinical "$CLIN/BRCA_clinicalMatrix" \
  --probemap "$CLIN/gencode.probeMap" \
  --gmt "$CLIN/hallmark.gmt" \
  "${MIMIR_ARG[@]}" \
  --runs \
    MOCHI="$CKPT" \
    MOCHI-knockout="$CKPT"=gamma0

python3 -u eval_ablation.py \
  --data_dir "$DATA" \
  --out_dir results/current/abl_brca \
  --runs \
    full="$CKPT" \
    knockout="$CKPT"=gamma0

python3 -u eval_nmf_tf.py \
  --data_dir "$DATA" \
  --out_dir results/current/self_brca \
  --nmf_tf_ckpt "$CKPT" \
  "${MIMIR_ARG[@]}"
