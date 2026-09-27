#!/usr/bin/env bash
# 5안 사다리. 한 번에 하나씩만 바꿔서 차이의 출처를 가른다.
#
# 공통: aux_w_only(저랭크 잔차 없음) · gan_to_mse 0 · lambda_nmf 0 · w_from_others · lambda_w 2.0
# 출력단 NMF 손실(lambda_nmf)은 ari_k2 를 0.068 로 만든 항이라 전 단계에서 끈다.
#
#   s0  기준선        평균 융합, 토큰 없음, 분리 없음
#   s1  + 분리        공유/전용 분리. 전용 절반은 블록 손실의 기울기를 못 받는다
#   s2  + 경로 배치   블록은 평균 유지, 칸에만 Transformer
#   s3  + 성분 토큰   계수 스칼라 대신 인코더(W[j]·H[j])
#
# s1 이 s0 와 같게 나오는 것은 기각 사유가 아니다. 분리는 성능 장치가 아니라
# 뒤 단계가 설 기반이며, "공짜로 얹힌다"가 통과 조건이다.
#
# 시드를 여러 개 돌린다. 코호트당 한 런이 1~2분이라 차이가 잡음인지 가르려면 필요하다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES="${RES:-$ROOT/results/current/lr_stage5}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
COHORTS="${COHORTS:-brca luad kirc}"
SEEDS="${SEEDS:-0 1 2}"
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
        --aux_w_only --w_from_others --lambda_w 2.0 --lambda_nmf 0 --detach_w_head)

stage_args() {
  case "$1" in
    s0) echo "--no_transformer --no_nmf_tokens" ;;
    s1) echo "--no_transformer --no_nmf_tokens --split_latent" ;;
    s2) echo "--no_nmf_tokens --split_latent --block_mean" ;;
    s3) echo "--split_latent --block_mean --content_tokens" ;;
    *)  echo "unknown stage $1" >&2; return 1 ;;
  esac
}

for coh in $COHORTS; do
  DATA="$ROOT/processed_data/gate_$coh"
  if [[ ! -f "$DATA/rna.train.tsv" ]]; then
    echo "skip $coh: $DATA 없음"; continue
  fi
  for sd in $SEEDS; do
    for st in s0 s1 s2 s3; do
      dir="$RES/${st}/${coh}_s${sd}"
      if [[ -f "$dir/nmf_tf_best.ckpt" ]]; then
        echo "skip $st $coh seed$sd (ckpt exists)"; continue
      fi
      mkdir -p "$dir"
      echo "===== $st $coh seed=$sd ====="
      # shellcheck disable=SC2046
      "$PY" -u train_nmf_tf.py "${COMMON[@]}" $(stage_args "$st") \
        --data_dir "$DATA" --save_dir "$dir" --seed "$sd" 2>&1 | tee "$dir/train.log"
    done
  done
done

echo
echo "=== 요약: 단계별 val 블록 z-RMSE (시드 평균 ± 표준편차) ==="
"$PY" - "$RES" <<'PYSUM'
import sys, glob, os
import numpy as np, torch
rows = {}
for ck in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "nmf_tf_best.ckpt"))):
    parts = ck.split(os.sep)
    stage, run = parts[-3], parts[-2]
    coh = run.rsplit("_s", 1)[0]
    d = torch.load(ck, map_location="cpu", weights_only=False)
    rows.setdefault((coh, stage), []).append(d.get("val", {}).get("avg", float("nan")))
if not rows:
    sys.exit("체크포인트 없음")
cohorts = sorted({c for c, _ in rows})
stages = ["s0", "s1", "s2", "s3"]
print("코호트   " + "".join(f"{s:>18}" for s in stages))
for c in cohorts:
    line = f"{c:<9}"
    for s in stages:
        v = rows.get((c, s))
        line += f"{(f'{np.mean(v):.4f}±{np.std(v):.4f}' if v else '-'):>18}"
    print(line)
print()
print("읽는 법: s1 이 s0 와 표준편차 안에서 같으면 분리는 공짜로 얹힌 것이고 통과다.")
print("         s2, s3 가 s1 을 표준편차 넘게 줄여야 그 부품이 값을 한 것이다.")
PYSUM
