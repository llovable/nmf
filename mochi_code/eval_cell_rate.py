#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""칸 결측률별 재평가 + 짝지은 비교. 재학습 없이 기존 체크포인트만 읽는다.

두 가지를 한 번에 본다.

1) 짝지은 통계.
   사다리는 (코호트, 시드)마다 phase1 AE 와 시드를 공유하므로 c0/c1/c2 는
   같은 조건에서 짝지어져 있다. 그러면 단계별 표준편차가 아니라 **차이의**
   표준편차로 판정해야 한다. 시드 사이 공통 잡음이 차감되기 때문에 보통
   훨씬 작고, 단계 표준편차로 재면 실제보다 약하게 읽힌다.

2) 결측률 의존성.
   칸 경로에 attention 을 둔 근거는 "MCAR 70~90% 에서 attention 이 앞선다"
   였는데 학습·검증은 30% 한 점에서만 쟀다. 이득이 결측률과 함께 커지면
   기전이 맞는 것이고, 평평하면 그 서사는 못 쓴다.

가리는 자리는 전 단계·전 시드에 대해 같은 마스크 시드로 고정한다. 그래야
차이가 모델 차이지 마스크 차이가 아니다.

사용:
  python eval_cell_rate.py --root results/current/lr_stage5c \
      --cohorts brca luad kirc --stages c0 c1 c2 --seeds 0 1 2 \
      --rates 0.1 0.3 0.5 0.7 0.9 --split test \
      --out results/current/lr_stage5c/cell_rate.tsv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_nmf_tf import load_nmf_tf
from train_gate import TripleSplitDataset
from train_nmf_tf import cell_zrmse


def main():
    ap = argparse.ArgumentParser(description="칸 결측률별 재평가와 짝지은 비교")
    ap.add_argument("--root", required=True, help="사다리 결과 루트 (예: results/current/lr_stage5c)")
    ap.add_argument("--data_root", default="processed_data", help="gate_<코호트> 가 있는 곳")
    ap.add_argument("--cohorts", nargs="+", default=["brca", "luad", "kirc"])
    ap.add_argument("--stages", nargs="+", default=["c0", "c1", "c2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="기본 test. val 은 조기 종료에 쓰였으므로 낙관적이다")
    ap.add_argument("--mask_seed", type=int, default=12345,
                    help="가리는 자리의 시드. 전 단계·전 시드가 같은 마스크를 봐야 한다")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  split={args.split}  mask_seed={args.mask_seed}")
    root = Path(args.root)

    rows = []
    for coh in args.cohorts:
        data_dir = Path(args.data_root) / f"gate_{coh}"
        if not (data_dir / "rna.train.tsv").is_file():
            print(f"[건너뜀] {coh}: {data_dir} 없음")
            continue
        train = TripleSplitDataset(str(data_dir), "train")
        ds = TripleSplitDataset(str(data_dir), args.split, stats=train.stats)
        print(f"\n=== {coh}  n({args.split})={len(ds)} ===")
        for stage, sd in itertools.product(args.stages, args.seeds):
            ck = root / stage / f"{coh}_s{sd}" / "nmf_tf_best.ckpt"
            if not ck.is_file():
                print(f"  [없음] {stage} seed{sd}")
                continue
            model = load_nmf_tf(ck, device)
            sw = float(torch.load(ck, map_location="cpu",
                                  weights_only=False).get("self_weight", 10.0))
            for rate in args.rates:
                met = cell_zrmse(model, ds, device, rate=rate,
                                 seed=args.mask_seed, self_weight=sw)
                rows.append({"cohort": coh, "stage": stage, "seed": sd, "rate": rate,
                             "cell_avg": met["avg"], "rna": met["rna"],
                             "protein": met["protein"], "methyl": met["methyl"]})
            print(f"  {stage} seed{sd} 완료")

    if not rows:
        raise SystemExit("읽은 체크포인트가 없습니다.")
    df = pd.DataFrame(rows)

    print("\n=== 결측률별 칸 z-RMSE (시드 평균 ± 표준편차) ===")
    piv = df.groupby(["cohort", "rate", "stage"])["cell_avg"].agg(["mean", "std"])
    for coh in df["cohort"].unique():
        print(f"\n[{coh}]")
        hdr = "  rate  " + "".join(f"{s:>20}" for s in args.stages)
        print(hdr)
        for rate in sorted(df["rate"].unique()):
            line = f"  {rate:<6.2f}"
            for s in args.stages:
                try:
                    m, sd_ = piv.loc[(coh, rate, s)]
                    line += f"{f'{m:.4f}±{sd_:.4f}':>20}"
                except KeyError:
                    line += f"{'-':>20}"
            print(line)

    # 짝지은 비교: 같은 (코호트, 시드, 결측률)에서 단계 간 차이를 먼저 만든다.
    print("\n=== 짝지은 차이 (같은 시드끼리 뺀 뒤 통계) ===")
    print("  단계별 표준편차가 아니라 '차이의' 표준편차로 본다. 공통 잡음이 차감된다.\n")
    wide = df.pivot_table(index=["cohort", "seed", "rate"], columns="stage",
                          values="cell_avg")
    pairs = [(args.stages[i], args.stages[i + 1]) for i in range(len(args.stages) - 1)]
    if len(args.stages) >= 3:
        pairs.append((args.stages[0], args.stages[-1]))
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        print(f"  --- {b} − {a}  (음수면 {b} 가 낫다) ---")
        print(f"  {'코호트':<8}{'rate':>6}{'평균Δ':>12}{'표준편차Δ':>12}{'평균/sd':>10}{'부호':>8}")
        for coh in wide.index.get_level_values("cohort").unique():
            for rate in sorted(wide.index.get_level_values("rate").unique()):
                sub = wide.xs((coh, rate), level=("cohort", "rate"))
                d = (sub[b] - sub[a]).dropna()
                if len(d) < 2:
                    continue
                sd_ = float(d.std())
                ratio = float(d.mean() / sd_) if sd_ > 0 else float("inf")
                sign = f"{int((d < 0).sum())}/{len(d)}"
                print(f"  {coh:<8}{rate:>6.2f}{d.mean():>12.5f}{sd_:>12.5f}"
                      f"{ratio:>10.2f}{sign:>8}")
        alld = (wide[b] - wide[a]).dropna()
        neg = int((alld < 0).sum())
        print(f"  전체 {len(alld)}개 비교 중 {b} 가 나은 경우: {neg}개  "
              f"(무작위면 {len(alld)/2:.1f}개 기대)\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
