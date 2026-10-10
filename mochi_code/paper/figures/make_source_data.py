#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""그림 1의 원자료 CSV를 평가 출력에서 직접 만든다.

왜 필요한가
-----------
plot_pathway_knockout.py는 source_pathway_knockout.csv를 읽는데, 그 CSV를
만드는 코드가 없었다. 수치를 eval_biology.py / eval_ablation.py 출력에서
손으로 옮기면 (1) 재현이 안 되고 (2) 전사 오류가 검출되지 않고 (3) 재학습할
때마다 수작업이 반복된다. 이 스크립트가 그 경로를 코드로 고정한다.

입력
----
  eval_biology.py  -> <out_dir>/biology.tsv        (Panel A, B)
  eval_ablation.py -> <out_dir>/eval_summary.tsv   (Panel C)

두 스크립트 모두 --runs에 저랭크 녹아웃 항목이 있어야 한다:
  eval_biology.py  : MOCHI-knockout=<ckpt>=gamma0
  eval_ablation.py : knockout=<ckpt>=gamma0
(둘 다 기본 --runs에 들어 있다.)

사용
----
  python paper/figures/make_source_data.py \\
      --biology  BRCA=results/current/gate_biology/biology.tsv \\
                 LUAD=results/current/gate_biology_luad/biology.tsv \\
                 KIRC=results/current/gate_biology_kirc/biology.tsv \\
      --ablation BRCA=results/current/gate_ablate/eval_summary.tsv \\
                 LUAD=results/current/gate_ablate_luad/eval_summary.tsv \\
                 KIRC=results/current/gate_ablate_kirc/eval_summary.tsv

설계 원칙: **없는 값은 만들어내지 않는다.** 필요한 행이 없으면 무엇이
없는지 정확히 말하고 종료한다. 조용한 기본값이나 자리표시자를 넣지 않는다.
CSV 옆에 source_provenance.txt를 함께 써서 어떤 파일에서 왔는지 남긴다.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import pandas as pd

# biology.tsv의 method 값 -> 그림 표시 이름. Panel A의 행 순서이기도 하다.
PANEL_A_METHODS = [
    ("mean", "Mean"),
    ("Ridge-2to1", "Ridge 2→1"),
    ("MIMIR", "MIMIR"),
    ("MOCHI", "MOCHI"),
]
PANEL_A_METRICS = ("pathway_sd_ratio", "pathway_r")

# Panel B: 같은 가중치에서 저랭크만 끈 paired 비교.
PANEL_B_PAIR = [("MOCHI", "MOCHI"), ("MOCHI-knockout", "Knockout")]
PANEL_B_METRICS = ("pathway_sd_ratio", "pathway_r")

# Panel C: 블록 z-RMSE의 (녹아웃 − 보고 모형). 양수 = 저랭크가 오차를 줄인다.
OMICS = ("rna", "methyl", "protein")
ABL_REPORTED = ("MOCHI", "self-w10")     # (method, setting)
ABL_KNOCKOUT = ("MOCHI-knockout", None)  # setting은 보지 않는다


class MissingRow(SystemExit):
    pass


def _parse_kv(items, what):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--{what} 형식 오류: {it} (COHORT=경로)")
        k, v = it.split("=", 1)
        p = Path(v)
        if not p.exists():
            raise SystemExit(f"--{what}의 {k}: 파일 없음 {p}")
        out[k] = p
    if not out:
        raise SystemExit(f"--{what}가 비어 있습니다")
    return out


def _one(df, where, cohort, src, what):
    """조건에 맞는 행 하나를 꺼낸다. 0개면 무엇이 없는지 말하고 종료한다."""
    sub = df[where]
    if len(sub) == 0:
        raise MissingRow(
            f"[{cohort}] {what}에 해당하는 행이 {src}에 없습니다.\n"
            f"  eval을 녹아웃 항목을 포함해 다시 돌렸는지 확인하세요:\n"
            f"    eval_biology.py  --runs ... MOCHI-knockout=<ckpt>=gamma0\n"
            f"    eval_ablation.py --runs ... knockout=<ckpt>=gamma0"
        )
    if len(sub) > 1:
        raise MissingRow(
            f"[{cohort}] {what}에 해당하는 행이 {src}에 {len(sub)}개입니다. "
            f"하나여야 합니다:\n{sub.to_string(index=False)}"
        )
    return sub.iloc[0]


def build_panel_a(bio, cohort, src):
    rows = []
    for metric in PANEL_A_METRICS:
        for key, label in PANEL_A_METHODS:
            if metric not in bio.columns:
                raise MissingRow(f"[{cohort}] biology.tsv에 '{metric}' 열이 없습니다 ({src})")
            r = _one(bio, (bio["modality"] == "rna") & (bio["method"] == key),
                     cohort, src, f"rna / method={key}")
            rows.append({"panel": "A", "cohort": cohort, "method": label,
                         "omics": "rna", "metric": metric, "value": float(r[metric])})
    return rows


def build_panel_b(bio, cohort, src):
    rows = []
    for metric in PANEL_B_METRICS:
        for key, label in PANEL_B_PAIR:
            r = _one(bio, (bio["modality"] == "rna") & (bio["method"] == key),
                     cohort, src, f"rna / method={key}")
            rows.append({"panel": "B", "cohort": cohort, "method": label,
                         "omics": "rna", "metric": metric, "value": float(r[metric])})
    return rows


def build_panel_c(abl, cohort, src, split="test"):
    rows = []
    base = abl[(abl["split"] == split) & (abl["mechanism"] == "block")]
    for om in OMICS:
        sel = base[(base["missing"] == om) & (base["modality"] == om)]
        m_name, m_setting = ABL_REPORTED
        cond = (sel["method"] == m_name)
        if m_setting is not None:
            cond = cond & (sel["setting"] == m_setting)
        rep = _one(sel, cond, cohort, src, f"block/{om} 보고 모형 {m_name}({m_setting})")
        k_name, _ = ABL_KNOCKOUT
        kno = _one(sel, sel["method"] == k_name, cohort, src,
                   f"block/{om} 녹아웃 {k_name}")
        rows.append({"panel": "C", "cohort": cohort, "method": "delta",
                     "omics": om, "metric": "z_rmse",
                     "value": float(kno["z_rmse"]) - float(rep["z_rmse"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biology", nargs="+", required=True,
                    help="COHORT=<eval_biology 출력의 biology.tsv> (여러 개)")
    ap.add_argument("--ablation", nargs="+", required=True,
                    help="COHORT=<eval_ablation 출력의 eval_summary.tsv> (여러 개)")
    ap.add_argument("--panel_a_cohort", default="BRCA",
                    help="Panel A에 쓸 코호트 (기본 BRCA)")
    ap.add_argument("--split", default="test", help="Panel C에 쓸 분할 (기본 test)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "source_pathway_knockout.csv"))
    args = ap.parse_args()

    bio_paths = _parse_kv(args.biology, "biology")
    abl_paths = _parse_kv(args.ablation, "ablation")
    if args.panel_a_cohort not in bio_paths:
        raise SystemExit(f"--panel_a_cohort={args.panel_a_cohort}가 --biology에 없습니다: "
                         f"{sorted(bio_paths)}")

    rows = []
    a_src = bio_paths[args.panel_a_cohort]
    rows += build_panel_a(pd.read_csv(a_src, sep="\t"), args.panel_a_cohort, a_src)
    for cohort, p in bio_paths.items():
        rows += build_panel_b(pd.read_csv(p, sep="\t"), cohort, p)
    for cohort, p in abl_paths.items():
        rows += build_panel_c(pd.read_csv(p, sep="\t"), cohort, p, split=args.split)

    df = pd.DataFrame(rows, columns=["panel", "cohort", "method", "omics", "metric", "value"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")

    prov = out.with_name("source_provenance.txt")
    with open(prov, "w") as f:
        f.write(f"생성 시각: {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Panel A 코호트: {args.panel_a_cohort}\n")
        f.write(f"Panel C 분할: {args.split}\n\n")
        for label, paths in (("biology", bio_paths), ("ablation", abl_paths)):
            for cohort, p in paths.items():
                st = p.stat()
                f.write(f"{label:9s} {cohort:6s} {p}  "
                        f"({st.st_size} bytes, mtime "
                        f"{_dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')})\n")

    print(df.to_string(index=False))
    print(f"\nwrote {out}  ({len(df)} rows)")
    print(f"wrote {prov}")


if __name__ == "__main__":
    main()
