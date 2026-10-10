#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""단순 방법 대비 기준표. 학습 없이 기존 체크포인트와 고전 보간법만 쓴다.

왜 필요한가
-----------
우리 채움 오차 0.88 이 좋은 숫자인지 나쁜 숫자인지 판단할 기준이 없다.
그리고 부분 결측에서의 이득이 '자기 데이터를 10배 믿는' 설정에서 나오므로,
심사에서 반드시 이 질문이 온다 — "복잡한 구조를 다 빼고 오토인코더만 쓰면?"

여기서 재는 것
--------------
  MOCHI        체크포인트 그대로
  AE-only      같은 체크포인트의 오믹스별 오토인코더만. 융합을 전혀 안 쓴다.
               부분 결측에서만 정의된다(통째 결측은 자기 데이터가 없다).
  KNN          이웃 환자 평균
  softImpute   저랭크 행렬 완성(오믹스별)
  softImpute-concat  세 오믹스를 이어붙여 완성. 통째 결측에는 train 행을
               같이 넣어야 한다(그러지 않으면 빈 열이 0에 고정된다).

두 체제를 따로 잰다. 통째 결측은 오믹스 하나를 전부 가리고, 부분 결측은
관측된 칸의 일부를 가린다. 가리는 자리는 시드로 고정해 방법 간 비교가
같은 문제 위에서 이뤄지게 한다.

사용:
  python eval_baselines.py --data_dir processed_data/gate_brca \
      --ckpt results/current/lr_stage5c/c1/brca_s0/nmf_tf_best.ckpt \
      --split test --gpu 0 --out results/current/baselines_brca.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baselines_self import MODS, all_self_predictions
from models_nmf_tf import load_nmf_tf, predict_nmf_tf
from train_gate import TripleSplitDataset

_TAB = {"protein": "prot_f", "rna": "rna_f", "methyl": "methy_f"}
_MASK = {"protein": "m_prot", "rna": "m_rna", "methyl": "m_methy"}


def _tabs(ds):
    return ({m: getattr(ds, _TAB[m]) for m in MODS},
            {m: getattr(ds, _MASK[m]) < 0.5 for m in MODS})


def _score(hat, truth, sel):
    return float(np.sqrt(np.mean((truth[sel] - hat[sel]) ** 2))) if sel.any() else float("nan")


@torch.no_grad()
def block_table(model, ds, train_ds, device, si_rank):
    """통째 결측: 오믹스 하나를 전부 가리고 나머지로 복원."""
    tabs, obs = _tabs(ds)
    tr_tabs, _ = _tabs(train_ds)
    rows = []
    for tgt in MODS:
        sel = obs[tgt]
        hat_m = predict_nmf_tf(model, tabs, device, missing=tgt)[tgt]
        rows.append({"method": "MOCHI", "regime": "통째", "target": tgt,
                     "zrmse": _score(hat_m, tabs[tgt], sel)})
        nan_tabs = {m: (np.full_like(tabs[m], np.nan) if m == tgt
                        else tabs[m].astype(np.float64)) for m in MODS}
        preds = all_self_predictions(nan_tabs, si_rank=si_rank, anchor_tabs=tr_tabs)
        for name, p in preds.items():
            rows.append({"method": name, "regime": "통째", "target": tgt,
                         "zrmse": _score(p[tgt], tabs[tgt], sel)})
    return rows


@torch.no_grad()
def cell_table(model, ds, train_ds, device, rate, seed, self_weight, si_rank):
    """부분 결측: 관측된 칸의 일부를 가리고 복원."""
    tabs, obs = _tabs(ds)
    tr_tabs, _ = _tabs(train_ds)
    rng = np.random.default_rng(seed)
    hide = {m: (rng.random(tabs[m].shape) < rate) & obs[m] for m in MODS}
    tabs_in = {m: np.where(hide[m], 0.0, tabs[m]).astype(np.float32) for m in MODS}

    rows = []
    hat_mochi = predict_nmf_tf(model, tabs_in, device, missing=None, self_weight=self_weight)
    # AE-only: 융합을 전혀 쓰지 않고 자기 오믹스 오토인코더만
    hat_ae = {}
    n = next(iter(tabs.values())).shape[0]
    for i in range(0, n, 64):
        sl = slice(i, i + 64)
        xs = {m: torch.from_numpy(tabs_in[m][sl]).to(device) for m in MODS}
        own = model.reconstruct_own(xs)
        for m in MODS:
            hat_ae.setdefault(m, []).append(own[m].cpu().numpy())
    hat_ae = {m: np.concatenate(v, 0) for m, v in hat_ae.items()}

    nan_tabs = {m: np.where(hide[m], np.nan, tabs[m]).astype(np.float64) for m in MODS}
    preds = all_self_predictions(nan_tabs, si_rank=si_rank, anchor_tabs=tr_tabs)
    preds["MOCHI"] = hat_mochi
    preds["AE-only"] = hat_ae
    for name, p in preds.items():
        for m in MODS:
            rows.append({"method": name, "regime": "부분", "target": m,
                         "zrmse": _score(p[m], tabs[m], hide[m])})
    return rows


def main():
    ap = argparse.ArgumentParser(description="단순 방법 대비 기준표")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="비교 기준이 될 MOCHI 체크포인트")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--rate", type=float, default=0.3, help="부분 결측에서 가리는 비율")
    ap.add_argument("--mask_seed", type=int, default=12345)
    ap.add_argument("--si_rank", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_ds = TripleSplitDataset(args.data_dir, "train")
    ds = TripleSplitDataset(args.data_dir, args.split, stats=train_ds.stats)
    print(f"device={device}  split={args.split}  n={len(ds)}  부분결측률={args.rate}")

    model = load_nmf_tf(args.ckpt, device)
    sw = float(torch.load(args.ckpt, map_location="cpu",
                          weights_only=False).get("self_weight", 10.0))

    rows = block_table(model, ds, train_ds, device, args.si_rank)
    rows += cell_table(model, ds, train_ds, device, args.rate,
                       args.mask_seed, sw, args.si_rank)
    df = pd.DataFrame(rows)

    print("\n=== 채움 오차 (낮을수록 좋음. 1.0 = 평균 찍기와 같음) ===")
    piv = df.pivot_table(index="method", columns=["regime", "target"], values="zrmse")
    piv["평균"] = df.pivot_table(index="method", columns="regime",
                                 values="zrmse").mean(axis=1)
    order = [m for m in ("MOCHI", "AE-only", "KNN", "softImpute",
                         "softImpute-concat", "Ridge-self") if m in piv.index]
    print(piv.loc[order].to_string(float_format=lambda v: f"{v:.4f}", na_rep="-"))
    print("\n참고: AE-only 는 부분 결측에서만 정의됩니다(통째 결측은 자기 데이터가 없음).")
    print("      MOCHI 가 AE-only 와 비슷하면, 부분 결측 이득은 융합 구조의 공이 아닙니다.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
