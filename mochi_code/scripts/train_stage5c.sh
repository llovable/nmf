#!/usr/bin/env bash
# 칸 경로 사다리. 5안의 s2·s3 를 제대로 재는 실험이다.
#
# 왜 따로 있나: 지난 s0..s3 에서 s1/s2/s3 숫자가 표준편차까지 같았다. 버그가
# 아니라 구조였다. 학습 손실과 검증 지표가 둘 다 블록 경로만 쓰는데,
# cell_attn 과 content_tokens 는 칸 경로에만 작동한다. 그래서 그 둘은 학습에도
# 평가에도 영향을 줄 수 없었고, 세 단계가 같은 모델이었다.
#
# 두 군데를 고쳐야 재진다.
#   --lambda_cell > 0   칸 결측을 학습 손실에 넣는다. 없으면 칸 경로는 추론에서
#                       처음 쓰인다 (제목은 Hybrid 인데 한쪽만 학습되던 자리).
#   --select_on cell    조기 종료를 칸 지표로. 칸을 학습하고 블록으로 고르면 어긋난다.
#
#   c0  기준선     칸 손실 on, 칸 융합은 평균, 토큰 없음
#   c1  + attention 칸 경로에만 Transformer     ← 경로 배치가 값을 하는가
#   c2  + 성분 토큰 내용 담은 토큰              ← 토큰이 값을 하는가
#
# 블록 경로는 세 단계 모두 평균이고 분리는 켜져 있다. s0..s3 에서 확정된 설정이다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES="${RES:-$ROOT/results/current/lr_stage5c}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
COHORTS="${COHORTS:-brca luad kirc}"
SEEDS="${SEEDS:-0 1 2}"
LAMBDA_CELL="${LAMBDA_CELL:-1.0}"
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
        --split_latent --block_mean
        --lambda_cell "$LAMBDA_CELL" --select_on cell)

stage_args() {
  case "$1" in
    c0) echo "--cell_mean --no_nmf_tokens" ;;
    c1) echo "--no_nmf_tokens" ;;
    c2) echo "--content_tokens" ;;
    *)  echo "unknown stage $1" >&2; return 1 ;;
  esac
}

for coh in $COHORTS; do
  DATA="$ROOT/processed_data/gate_$coh"
  [[ -f "$DATA/rna.train.tsv" ]] || { echo "skip $coh: $DATA 없음"; continue; }
  for sd in $SEEDS; do
    AE="$RES/c0/${coh}_s${sd}/ae_phase1.pt"
    for st in c0 c1 c2; do
      dir="$RES/${st}/${coh}_s${sd}"
      [[ -f "$dir/nmf_tf_best.ckpt" ]] && { echo "skip $st $coh seed$sd"; continue; }
      mkdir -p "$dir"
      reuse=()
      if [[ "$st" != "c0" ]]; then
        [[ -f "$AE" ]] || { echo "phase1 AE 없음: $AE (c0 부터)"; exit 1; }
        reuse=(--ae_ckpt "$AE")
      fi
      echo "===== $st $coh seed=$sd ====="
      # shellcheck disable=SC2046
      "$PY" -u train_nmf_tf.py "${COMMON[@]}" $(stage_args "$st") "${reuse[@]}" \
        --data_dir "$DATA" --save_dir "$dir" --seed "$sd" 2>&1 | tee "$dir/train.log"
    done
  done
done

echo
echo "=== 요약: 칸 val z-RMSE (시드 평균 ± 표준편차). 괄호는 블록 ==="
"$PY" - "$RES" <<'PYSUM'
import sys, glob, os
import numpy as np, torch
cell, block = {}, {}
for ck in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "nmf_tf_best.ckpt"))):
    parts = ck.split(os.sep)
    stage, run = parts[-3], parts[-2]
    coh = run.rsplit("_s", 1)[0]
    d = torch.load(ck, map_location="cpu", weights_only=False)
    vc = (d.get("val_cell") or {}).get("avg", float("nan"))
    cell.setdefault((coh, stage), []).append(vc)
    block.setdefault((coh, stage), []).append(d.get("val", {}).get("avg", float("nan")))
if not cell:
    sys.exit("체크포인트 없음")
stages = ["c0", "c1", "c2"]
print("코호트   " + "".join(f"{s:>26}" for s in stages))
for c in sorted({x for x, _ in cell}):
    line = f"{c:<9}"
    for s in stages:
        v, b = cell.get((c, s)), block.get((c, s))
        line += f"{(f'{np.mean(v):.4f}±{np.std(v):.4f} ({np.mean(b):.3f})' if v else '-'):>26}"
    print(line)
print()
print("읽는 법: c1 이 c0 를 표준편차 넘게 줄여야 경로 배치가 값을 한 것이다.")
print("         c2 가 c1 을 표준편차 넘게 줄여야 성분 토큰이 값을 한 것이다.")
print("         괄호 안 블록 값은 참고용 — 세 단계가 비슷해야 정상이다.")
PYSUM
