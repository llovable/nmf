#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""수렴 검증: 보간값을 되짚어 나온 불일치가 실제 오차를 예측하는가.

무엇을 재는가
-------------
블록 결측(오믹스 하나를 통째로 가림)으로 타깃을 채운 뒤, 그 채운 값이
얼마나 믿을 만한지를 **정답을 안 보고** 추정할 수 있는지 본다. 신호 둘을 잰다.

  coef  성분 좌표 불일치.  **단독 채택 금지.**
        채운 값의 실제 계수 W(x̂) 와 융합이 예측한 계수 Ŵ 를 k차원에서 비교한다.
        문제는 Ŵ 가 참 계수 W(x) 를 맞히도록 학습됐다는 점이다. 그 학습이 잘 됐다면
            ‖W(x̂) − Ŵ‖ ≈ ‖W(x̂) − W(x)‖
        이고, 오른쪽은 보간 오차를 NMF 좌표로 다시 쓴 것일 뿐이다. 그러면 실제
        오차와의 상관은 검증이 아니라 동어반복이다. 이 스크립트는 그 가설을
        직접 잰다 — coef 와 참 NMF 좌표 오차의 상관을 coef_tautology 로 낸다.
        이 값이 높으면 coef 는 오차의 재표현이므로 근거로 쓰지 않는다.

  cycle 특징 공간 순환 일치.  **난이도 대조를 통과해야 함.**
        채운 x̂ 를 입력으로 되돌려 **관측된** 다른 오믹스를 역예측하고 참값과 비교한다.
        타깃 정답을 보지 않으므로 coef 같은 동어반복은 없다. 다만 맞추기 어려운
        환자는 어느 방향으로 예측해도 오차가 크다. 그래서 상관이 있어도 "이 보간이
        틀렸다"가 아니라 "이 환자가 어렵다"일 수 있다. 아래 두 대조를 통제한
        편상관(partial)이 남아야 신호로 인정한다.

  대조 dist_mean  관측 오믹스가 코호트 평균에서 떨어진 정도. 모델을 전혀 안 쓴다.
       self_recon 관측 오믹스를 자기 자신으로 복원했을 때의 오차. 환자 난이도.

판정
----
핵심은 raw Spearman 이 아니라 **편상관** partial_min 이다. cycle 이 두 난이도
대조를 통제하고도 실제 오차를 가리키면, 그때만 "언제 믿으면 안 되는지"를
말할 수 있다. raw 상관만 높고 편상관이 사라지면 그것은 환자 난이도를 잰 것이다.
top20_recall 은 신호로 상위 20%를 거를 때 진짜 최악 20%를 몇 비율 잡는지다
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


def _partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """c 를 통제한 a-b 순위 편상관. 난이도를 빼고도 신호가 남는지 본다."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if ok.sum() < 5:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    rc = pd.Series(c[ok]).rank().to_numpy()
    if min(ra.std(), rb.std(), rc.std()) == 0:
        return float("nan")
    rab = np.corrcoef(ra, rb)[0, 1]
    rac = np.corrcoef(ra, rc)[0, 1]
    rbc = np.corrcoef(rb, rc)[0, 1]
    den = np.sqrt(max(1e-12, (1 - rac ** 2) * (1 - rbc ** 2)))
    return float((rab - rac * rbc) / den)


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
        d_dist, d_self, nmf_err = [], [], []
        w_pred_all = []

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

            # 3) coef 신호 + 동어반복 진단용 참 NMF 좌표 오차
            #    w_true 는 진단에만 쓴다. 신호 계산에는 절대 들어가지 않는다.
            if w_pred is not None:
                w_hat = model.tokenizers[tgt].encode(x_hat)
                d_coef.append(torch.linalg.norm(w_hat - w_pred, dim=1).cpu().numpy())
                w_true = model.tokenizers[tgt].encode(truth)
                nmf_err.append(torch.linalg.norm(w_hat - w_true, dim=1).cpu().numpy())
                w_pred_all.append(w_pred.cpu().numpy())
            else:
                d_coef.append(np.full(b, np.nan, dtype=np.float32))
                nmf_err.append(np.full(b, np.nan, dtype=np.float32))

            # 3b) 대조 1 — 관측 오믹스가 코호트 평균에서 떨어진 정도.
            #     데이터는 train 통계로 z-정규화돼 있으므로 원점이 코호트 평균이다.
            #     모델을 전혀 쓰지 않는 순수 난이도 대용치다.
            dm = torch.zeros(b, device=device)
            for m in xs:
                dm = dm + torch.linalg.norm(xs[m], dim=1) / np.sqrt(xs[m].size(1))
            d_dist.append((dm / max(len(xs), 1)).cpu().numpy())

            # 3c) 대조 2 — 관측 오믹스를 자기 자신으로 복원한 오차.
            own = model.reconstruct_own(xs)
            sr = torch.zeros(b, device=device)
            for m, rec in own.items():
                tru_m = torch.from_numpy(np.asarray(tabs[m][sl], dtype=np.float32)).to(device)
                keep_m = torch.from_numpy(np.asarray(obs[m][sl])).to(device)
                se_m = ((rec - tru_m) ** 2) * keep_m.float()
                cnt_m = keep_m.float().sum(dim=1).clamp_min(1.0)
                sr = sr + torch.sqrt(se_m.sum(dim=1) / cnt_m)
            d_self.append((sr / max(len(own), 1)).cpu().numpy())

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
            "dist_mean": np.concatenate(d_dist),
            "self_recon": np.concatenate(d_self),
            "_nmf_err": np.concatenate(nmf_err) if nmf_err else np.full(n, np.nan),
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
            # coef 가 오차의 재표현인지: 참 NMF 좌표 오차와 얼마나 같은 순위인가
            taut = _spearman(r["coef"], r["_nmf_err"])
            for sig in ("cycle", "coef", "dist_mean", "self_recon"):
                is_control = sig in ("dist_mean", "self_recon")
                ctrls = [c for c in ("dist_mean", "self_recon") if c != sig]
                parts = [_partial_spearman(r[sig], r["err"], r[c]) for c in ctrls]
                pmin = float(np.nanmin(parts)) if np.any(np.isfinite(parts)) else float("nan")
                rows.append({
                    "run": name, "split": args.split, "target": tgt, "signal": sig,
                    "n": int(np.isfinite(r["err"]).sum()),
                    "spearman": _spearman(r[sig], r["err"]),
                    "partial_min": float("nan") if is_control else pmin,
                    "top20_recall": _top_recall(r[sig], r["err"]),
                    "coef_tautology": taut if sig == "coef" else float("nan"),
                    "err_mean": float(np.nanmean(r["err"])),
                    "lambda_w": lam_w,
                    "w_head_constant": bool(constant_head) if sig == "coef" else "",
                })
            if args.dump:
                dumps.append(pd.DataFrame({
                    "run": name, "split": args.split, "target": tgt,
                    "sample": np.arange(len(r["err"])),
                    "err": r["err"], "coef": r["coef"], "cycle": r["cycle"],
                    "dist_mean": r["dist_mean"], "self_recon": r["self_recon"],
                }))
            if constant_head:
                print(f"  [주의] {tgt}: 계수 머리가 사실상 상수다 "
                      f"(환자 간 표준편차 {spread:.2e}). lambda_w={lam_w} 런이면 정상이며, "
                      f"이때 coef 신호는 검증이 아니라 '코호트 평균에서의 거리'를 잰다.")

    if not rows:
        raise SystemExit("측정된 런이 없습니다.")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== 상관 요약 ===")
    print("  spearman    신호와 실제 오차의 순위상관 (무작위 0)")
    print("  partial_min 두 난이도 대조를 각각 통제한 편상관 중 낮은 쪽 — 이게 핵심")
    print("  dist_mean / self_recon 행은 대조 자체의 상관. cycle 이 이보다 못하면 의미 없음\n")
    print(df[["run", "target", "signal", "n", "spearman", "partial_min",
              "top20_recall", "err_mean"]].to_string(
                  index=False, na_rep="-", float_format=lambda v: f"{v:.3f}"))

    print("\n=== 판정 ===")
    for (run, sig), g in df.groupby(["run", "signal"]):
        sp = g["spearman"].mean()
        rc = g["top20_recall"].mean()
        if sig in ("dist_mean", "self_recon"):
            print(f"  {run:12s} {sig:10s} 평균 spearman={sp:+.3f}  (대조 기준선)")
            continue
        if sig == "coef":
            taut = g["coef_tautology"].mean()
            const = any(g["w_head_constant"] == True)  # noqa: E712
            if const:
                note = "계수 머리가 상수 — 검증 신호 아님"
            elif np.isfinite(taut) and taut >= 0.7:
                note = f"참 NMF 오차와 순위상관 {taut:+.2f} — 오차의 재표현, 근거로 쓰지 말 것"
            else:
                note = f"참 NMF 오차와 순위상관 {taut:+.2f} — 동어반복 여지 낮음, 그래도 단독 채택 금지"
            print(f"  {run:12s} {sig:10s} 평균 spearman={sp:+.3f}  →  {note}")
            continue
        pm = g["partial_min"].mean()
        if not np.isfinite(pm):
            verdict = "계산 불가"
        elif pm >= 0.30:
            verdict = "난이도를 빼고도 신호가 남음 — 보강 문장으로 채택"
        elif pm >= 0.15:
            verdict = "약함 — 코호트 셋 모두에서 같은 방향인지 확인 필요"
        else:
            verdict = "난이도로 설명됨 — 신호 아님, 보강 없이 닫을 것"
        print(f"  {run:12s} {sig:10s} 평균 spearman={sp:+.3f}  편상관={pm:+.3f}  "
              f"top20_recall={rc:.3f}  →  {verdict}")

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
