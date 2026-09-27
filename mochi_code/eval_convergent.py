#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""수렴 검증: 보간값을 되짚어 나온 불일치가 실제 오차를 예측하는가.

무엇을 재는가
-------------
블록 결측(오믹스 하나를 통째로 가림)으로 타깃을 채운 뒤, 그 채운 값이
얼마나 믿을 만한지를 **정답을 안 보고** 추정할 수 있는지 본다. 신호 둘을 잰다.

  coef  (4안) 성분 좌표 불일치.
        채운 값의 실제 NMF 계수 W(x̂) 와, 융합이 예측한 계수 Ŵ 를 k차원에서 비교.
        Ŵ 는 학습 중 참값의 계수를 맞추도록 감독됐으므로, 보간이 좋으면 둘이 같고
        나쁘면 벌어진다. k차원이라 2000차원 원본보다 안정적이고, 어느 성분이
        어긋났는지까지 지목할 수 있다.

  cycle (2안) 특징 공간 순환 일치.
        채운 x̂ 를 입력으로 되돌려 **관측된** 다른 오믹스를 역예측하고 참값과 비교.
        정답을 아는 쪽으로 되짚는 것이라 감독이 필요 없다.

판정
----
신호와 실제 오차의 순위상관(Spearman)이 핵심이다. 상관이 있으면 "어떤 환자의
어떤 보간값을 믿으면 안 되는지" 말할 수 있고, 없으면 4안의 토대가 무너진다.
top20_recall 은 신호로 상위 20%를 걸러낼 때 진짜 최악 20%를 몇 % 잡는지다
(무작위면 0.20).

주의
----
- split 은 test 가 기본이다. 학습이 val 로 조기 종료했으므로 val 상관은 낙관적이다.
- coef 신호는 계수 머리가 학습된 체크포인트에서만 뜻이 있다. lambda_w=0 이면
  머리가 코호트 평균 상수를 뱉으므로 이 스크립트가 감지해서 표시한다.

사용
----
  python eval_convergent.py --data_dir processed_data/gate_brca --gpu 0 \
      --runs ctl_base=results/current/lr_ctl/ctl_base/brca_hybrid/nmf_tf_best.ckpt \
      --out results/current/lr_ctl/convergent_brca.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_nmf_tf import MODS, load_nmf_tf
from train_gate import TripleSplitDataset

_TAB = {"protein": "prot_f", "rna": "rna_f", "methyl": "methy_f"}
_MASK = {"protein": "m_prot", "rna": "m_rna", "methyl": "m_methy"}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위상관. scipy 없이도 돌게 직접 계산한다 (동점은 평균 순위)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _top_recall(signal: np.ndarray, err: np.ndarray, frac=0.2) -> float:
    """신호로 상위 frac 을 걸렀을 때 실제 최악 frac 을 몇 비율 잡는가."""
    ok = np.isfinite(signal) & np.isfinite(err)
    n = int(ok.sum())
    k = max(1, int(round(n * frac)))
    if n < 5:
        return float("nan")
    s, e = signal[ok], err[ok]
    flagged = set(np.argsort(-s)[:k].tolist())
    worst = set(np.argsort(-e)[:k].tolist())
    return len(flagged & worst) / k


@torch.no_grad()
def measure(model, ds, device, batch_size=64):
    """타깃별로 (실제 오차, coef 불일치, cycle 불일치)를 환자 단위로 낸다."""
    tabs = {m: getattr(ds, _TAB[m]) for m in MODS}
    obs = {m: getattr(ds, _MASK[m]) < 0.5 for m in MODS}
    n = next(iter(tabs.values())).shape[0]
    out = {}

    for tgt in MODS:
        err, d_coef, d_cycle = [], [], []
        w_pred_all, w_hat_all = [], []

        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            xs, present = {}, {}
            b = None
            for m in MODS:
                if m == tgt:
                    continue
                t = torch.from_numpy(np.asarray(tabs[m][sl], dtype=np.float32)).to(device)
                xs[m] = t
                present[m] = torch.ones(t.size(0), dtype=torch.bool, device=device)
                b = t.size(0)
            present[tgt] = torch.zeros(b, dtype=torch.bool, device=device)

            # 1) 타깃을 채운다. loo_parts 가 예측 계수 Ŵ 도 같이 준다.
            h, w_pred = model.loo_parts(xs, present, tgt)
            x_hat = model.decode_target(tgt, h, w_pred)

            # 2) 실제 오차 (관측된 칸만 채점)
            truth = torch.from_numpy(np.asarray(tabs[tgt][sl], dtype=np.float32)).to(device)
            keep = torch.from_numpy(np.asarray(obs[tgt][sl])).to(device)
            se = ((x_hat - truth) ** 2) * keep.float()
            cnt = keep.float().sum(dim=1).clamp_min(1.0)
            err.append(torch.sqrt(se.sum(dim=1) / cnt).cpu().numpy())

            # 3) coef 신호: 채운 값의 실제 계수 vs 융합이 예측한 계수
            if w_pred is not None:
                w_hat = model.tokenizers[tgt].encode(x_hat)
                d = torch.linalg.norm(w_hat - w_pred, dim=1)
                d_coef.append(d.cpu().numpy())
                w_pred_all.append(w_pred.cpu().numpy())
                w_hat_all.append(w_hat.cpu().numpy())
            else:
                d_coef.append(np.full(b, np.nan, dtype=np.float32))

            # 4) cycle 신호: 채운 값으로 관측된 다른 오믹스를 역예측
            cyc = torch.zeros(b, device=device)
            n_src = 0
            for src in MODS:
                if src == tgt:
                    continue
                xs2 = {tgt: x_hat}
                pres2 = {tgt: torch.ones(b, dtype=torch.bool, device=device),
                         src: torch.zeros(b, dtype=torch.bool, device=device)}
                third = [m for m in MODS if m not in (tgt, src)]
                for m in third:
                    xs2[m] = xs[m]
                    pres2[m] = present[m]
                back = model.loo_reconstruct(xs2, pres2, src)
                tru_s = torch.from_numpy(np.asarray(tabs[src][sl], dtype=np.float32)).to(device)
                keep_s = torch.from_numpy(np.asarray(obs[src][sl])).to(device)
                se_s = ((back - tru_s) ** 2) * keep_s.float()
                cnt_s = keep_s.float().sum(dim=1).clamp_min(1.0)
                cyc = cyc + torch.sqrt(se_s.sum(dim=1) / cnt_s)
                n_src += 1
            d_cycle.append((cyc / max(n_src, 1)).cpu().numpy())

        rec = {
            "err": np.concatenate(err),
            "coef": np.concatenate(d_coef),
            "cycle": np.concatenate(d_cycle),
        }
        # 계수 머리가 상수인지 (lambda_w=0 이면 코호트 평균만 뱉는다)
        if w_pred_all:
            wp = np.concatenate(w_pred_all, 0)
            rec["w_pred_spread"] = float(np.mean(np.std(wp, axis=0)))
        else:
            rec["w_pred_spread"] = float("nan")
        out[tgt] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description="수렴 검증 신호와 실제 오차의 상관")
    ap.add_argument("--data_dir", required=True,
                    help="processed_data/gate_<코호트>")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="이름=체크포인트경로 형식. 여러 개 가능")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="기본 test. val 은 조기 종료에 쓰였으므로 상관이 낙관적이다")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="", help="요약 TSV 저장 경로")
    ap.add_argument("--dump", default="", help="환자 단위 원자료 TSV 저장 경로(선택)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  split={args.split}")

    train = TripleSplitDataset(args.data_dir, "train")
    ds = TripleSplitDataset(args.data_dir, args.split, stats=train.stats)
    print(f"n({args.split})={len(ds)}")

    rows, dumps = [], []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs 형식은 이름=경로 입니다: {spec}")
        name, path = spec.split("=", 1)
        if not Path(path).is_file():
            print(f"[건너뜀] {name}: {path} 없음")
            continue
        print(f"\n=== {name} ===")
        model = load_nmf_tf(path, device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        lam_w = ck.get("lambda_w")
        res = measure(model, ds, device, batch_size=args.batch_size)

        for tgt, r in res.items():
            spread = r["w_pred_spread"]
            constant_head = np.isfinite(spread) and spread < 1e-4
            for sig in ("coef", "cycle"):
                rows.append({
                    "run": name, "split": args.split, "target": tgt, "signal": sig,
                    "n": int(np.isfinite(r["err"]).sum()),
                    "spearman": _spearman(r[sig], r["err"]),
                    "pearson": _pearson(r[sig], r["err"]),
                    "top20_recall": _top_recall(r[sig], r["err"]),
                    "err_mean": float(np.nanmean(r["err"])),
                    "lambda_w": lam_w,
                    "w_head_constant": bool(constant_head) if sig == "coef" else "",
                })
            if args.dump:
                dumps.append(pd.DataFrame({
                    "run": name, "split": args.split, "target": tgt,
                    "sample": np.arange(len(r["err"])),
                    "err": r["err"], "coef": r["coef"], "cycle": r["cycle"],
                }))
            if constant_head:
                print(f"  [주의] {tgt}: 계수 머리가 사실상 상수다 "
                      f"(환자 간 표준편차 {spread:.2e}). lambda_w={lam_w} 런이면 정상이며, "
                      f"이때 coef 신호는 검증이 아니라 '코호트 평균에서의 거리'를 잰다.")

    if not rows:
        raise SystemExit("측정된 런이 없습니다.")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== 상관 요약 (무작위면 spearman 0, top20_recall 0.20) ===")
    print(df[["run", "target", "signal", "n", "spearman", "pearson",
              "top20_recall", "err_mean"]].to_string(index=False,
                                                     float_format=lambda v: f"{v:.3f}"))

    print("\n=== 판정 ===")
    for (run, sig), g in df.groupby(["run", "signal"]):
        sp = g["spearman"].mean()
        rc = g["top20_recall"].mean()
        if not np.isfinite(sp):
            verdict = "계산 불가"
        elif sp >= 0.4:
            verdict = "강한 신호 — 채택"
        elif sp >= 0.2:
            verdict = "약한 신호 — 코호트 확인 필요"
        else:
            verdict = "신호 없음 — 이 경로 기각"
        print(f"  {run:12s} {sig:6s} 평균 spearman={sp:+.3f}  top20_recall={rc:.3f}  →  {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False)
        print(f"\n저장: {args.out}")
    if args.dump and dumps:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(dumps).to_csv(args.dump, sep="\t", index=False)
        print(f"저장: {args.dump}")


if __name__ == "__main__":
    main()
