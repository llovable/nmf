#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""그림 1. 경로 진폭·순서 보존과 저랭크 녹아웃.

원자료: source_pathway_knockout.csv (make_source_data.py가 생성)
  A — BRCA RNA 블록, Hallmark 경로 표준편차 비 **와 상관** (표 10)
  B — 세 암종 RNA 경로 표준편차 비 **와 상관**, 저랭크 녹아웃 (표 11)
  C — 녹아웃 − 보고 모형의 블록 z-RMSE (양수 = 저랭크가 오차를 줄임)

A·B는 두 지표를 나란히 그린다. 진폭(SD 비)만 보이면 "유리한 지표를
앞세웠다"로 읽히고, 실제로 MOCHI는 진폭에서 앞서고 상관에서 뒤진다.
그 트레이드오프가 그림에서 바로 보여야 한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
CSV = HERE / "source_pathway_knockout.csv"
PDF = HERE / "fig1_pathway_knockout.pdf"
PNG = HERE / "fig1_pathway_knockout.png"

COL = {
    # Mean은 값이 0인 기준선이라 색으로 식별할 필요가 없다. 중립 잉크로 둔다.
    # 나머지 세 방법의 색은 색각이상 대비 검증을 통과한다
    # (#0072B2/#E69F00/#009E73: 최악 인접쌍 ΔE 11.4 protan, 24.2 normal).
    "mean": "#6E6E6E",
    "ridge": "#0072B2",
    "mimir": "#E69F00",
    "mochi": "#009E73",
    "knock": "#D55E00",
    "rna": "#0072B2",
    "methyl": "#56B4E9",
    "protein": "#CC79A7",
    "line": "#222222",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def panel_label(ax, letter):
    ax.text(
        0.0, 1.08, letter, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="bottom", ha="right",
    )


def _need(sel, what):
    """없는 값은 만들어내지 않는다. 무엇이 없는지 말하고 멈춘다."""
    if len(sel) == 0:
        raise SystemExit(
            f"source_pathway_knockout.csv에 '{what}' 행이 없습니다.\n"
            "  이 그림은 진폭(SD 비)과 순서(r)를 함께 그립니다. CSV를 다시 만드세요:\n"
            "    python paper/figures/make_source_data.py \\\n"
            "        --biology  BRCA=... LUAD=... KIRC=... \\\n"
            "        --ablation BRCA=... LUAD=... KIRC=...\n"
            "  eval_biology.py / eval_ablation.py를 녹아웃 항목(=gamma0)을 포함해\n"
            "  먼저 실행해야 합니다."
        )
    return float(sel.value.iloc[0])


def draw_a(ax, df):
    """진폭(SD 비)과 순서(상관)를 나란히 그린다.

    두 지표 모두 1이 진실이므로 축 하나를 공유한다 — 이중 축이 아니다.
    색은 방법(개체)을 나타내고 두 지표는 질감으로 구분한다. 색만으로
    식별하게 두지 않으므로 인쇄·색각이상에서도 읽힌다.
    """
    order = ["Mean", "Ridge 2→1", "MIMIR", "MOCHI"]
    colors = [COL["mean"], COL["ridge"], COL["mimir"], COL["mochi"]]
    pa = df[df.panel == "A"]
    vals = [_need(pa[(pa.metric == "pathway_sd_ratio") & (pa.method == m)],
                  f"A / {m} / pathway_sd_ratio") for m in order]
    rs = [_need(pa[(pa.metric == "pathway_r") & (pa.method == m)],
                f"A / {m} / pathway_r") for m in order]

    y = np.arange(len(order))
    h, gap = 0.30, 0.03          # gap = 인접 막대 사이 배경색 간격
    for i, c in enumerate(colors):
        ax.barh(y[i] - (h + gap) / 2, vals[i], height=h, color=c,
                edgecolor="none", zorder=2)
        ax.barh(y[i] + (h + gap) / 2, rs[i], height=h, facecolor="none",
                edgecolor=c, linewidth=0.7, hatch="////", zorder=2)
    ax.axvline(1.0, color=COL["line"], ls="--", lw=0.7, zorder=1)

    for i in range(len(order)):
        for v, off in ((vals[i], -(h + gap) / 2), (rs[i], (h + gap) / 2)):
            ax.text(max(v, 0.0) + 0.02, y[i] + off, f"{v:.3f}",
                    va="center", ha="left", fontsize=6.6, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.26)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlabel("Fraction of truth recovered  (1 = truth)")
    ax.set_title("BRCA RNA block-missing", loc="left", pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.legend(
        handles=[
            Patch(facecolor="#8A8A8A", edgecolor="none", label="Pathway SD ratio"),
            Patch(facecolor="none", edgecolor="#8A8A8A", hatch="////", linewidth=0.7,
                  label="Pathway correlation $r$"),
        ],
        frameon=False, loc="upper right", fontsize=6.6, handlelength=1.3,
        handletextpad=0.4, borderaxespad=0.15, labelspacing=0.25,
    )
    panel_label(ax, "A")


def draw_b(ax, df):
    """저랭크 녹아웃의 paired 비교. 진폭과 순서를 모두 그린다.

    같은 체크포인트에서 추론 시에만 저랭크를 끄므로 학습 시드 변동이
    상쇄된다. 따로 학습한 ablation보다 강한 증거다.
    행은 코호트 × 지표. SD 비만 그리면 저랭크가 순서(r)에 무슨 일을
    하는지 알 수 없다.
    """
    cohorts = ["BRCA", "LUAD", "KIRC"]
    metrics = [("pathway_sd_ratio", "SD ratio"), ("pathway_r", "$r$")]
    rows, labels = [], []
    for c in cohorts:
        for key, mlabel in metrics:
            sel = df[(df.panel == "B") & (df.cohort == c) & (df.metric == key)]
            a = _need(sel[sel.method == "MOCHI"], f"B / {c} / MOCHI / {key}")
            b = _need(sel[sel.method == "Knockout"], f"B / {c} / Knockout / {key}")
            rows.append((a, b))
            labels.append(f"{c}  {mlabel}")

    y = np.arange(len(rows))
    for i, (a, b) in enumerate(rows):
        ax.plot([b, a], [i, i], color="#888888", lw=1.2, zorder=1,
                solid_capstyle="butt")
        # 배경색 테두리를 둘러 두 점이 겹쳐도 구분된다.
        ax.plot(b, i, "o", ms=8, color=COL["knock"], zorder=3,
                mec="white", mew=1.2, label="Low-rank off" if i == 0 else None)
        ax.plot(a, i, "o", ms=8, color=COL["mochi"], zorder=3,
                mec="white", mew=1.2, label="MOCHI" if i == 0 else None)
        # 부호의 기준을 하나로 고정한다: 녹아웃이 보고 모형 대비 몇 %인가.
        rel = 100 * (b - a) / a if abs(a) > 1e-9 else float("nan")
        ax.text((a + b) / 2, i - 0.30, f"{rel:+.0f}%", ha="center",
                va="bottom", fontsize=6.4, color="#444444")
        ax.text(max(a, b) + 0.02, i, f"{max(a, b):.3f}", va="center", ha="left",
                fontsize=6.6, color="#333333")
        ax.text(min(a, b) - 0.02, i, f"{min(a, b):.3f}", va="center", ha="right",
                fontsize=6.6, color="#333333")

    ax.axvline(1.0, color=COL["line"], ls="--", lw=0.7, zorder=0)
    for i in range(2, len(rows), 2):     # 코호트 사이 옅은 구분선
        ax.axhline(i - 0.5, color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.10, 1.24)
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_xlabel("Fraction of truth recovered   (% = knockout vs MOCHI)")
    ax.set_title("Low-rank knockout (same weights)", loc="left", pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.legend(frameon=False, loc="lower right", fontsize=6.6, handletextpad=0.3,
              borderaxespad=0.15, labelspacing=0.25)
    panel_label(ax, "B")


def draw_c(ax, df):
    cohorts = ["BRCA", "LUAD", "KIRC"]
    omics = [("rna", "RNA"), ("methyl", "Methylation"), ("protein", "Protein")]
    x = np.arange(len(cohorts))
    width = 0.17
    per_omics = {}
    for j, (key, lab) in enumerate(omics):
        vals = []
        for c in cohorts:
            vals.append(_need(df[(df.panel == "C") & (df.cohort == c) & (df.omics == key)],
                              f"C / {c} / {key}"))
        per_omics[key] = vals
        ax.bar(
            x + (j - 1) * width, vals, width=width, color=COL[key],
            edgecolor="none", label=lab, zorder=2,
        )
    all_vals = np.array([v for vs in per_omics.values() for v in vs], dtype=float)
    ax.axhline(0, color=COL["line"], lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Δ z-RMSE  (knockout − MOCHI)")
    ax.set_title("Reconstruction cost of the residual", loc="left", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    lim = max(0.01, 1.35 * float(np.abs(all_vals).max()))
    ax.set_ylim(-lim, lim)
    ax.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=0.9,
              borderaxespad=0.2)

    # 주석은 **데이터에서 만든다**. 문구를 하드코딩하면 재학습 뒤 수치가
    # 바뀌었을 때 그림이 조용히 거짓말을 한다.
    helps = [lab for key, lab in omics if np.mean(per_omics[key]) > 0]
    hurts = [lab for key, lab in omics if np.mean(per_omics[key]) < 0]
    if helps:
        ax.text(len(cohorts) - 0.55, lim * 0.94,
                f"{' / '.join(helps)}: low-rank lowers error",
                fontsize=6.4, color="#333333", ha="right", va="top")
    if hurts:
        ax.text(len(cohorts) - 0.55, -lim * 0.94,
                f"{' / '.join(hurts)}: low-rank raises error",
                fontsize=6.4, color="#333333", ha="right", va="bottom")
    panel_label(ax, "C")


def main():
    style()
    df = pd.read_csv(CSV)
    # Panel B가 3행 -> 6행(코호트 × 지표)이 되어 위쪽 단을 키운다.
    fig = plt.figure(figsize=(7.2, 5.5), dpi=300)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.40, 0.95], hspace=0.38, wspace=0.40,
        left=0.115, right=0.985, top=0.92, bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_a(ax_a, df)
    draw_b(ax_b, df)
    draw_c(ax_c, df)
    fig.savefig(PDF)
    fig.savefig(PNG, dpi=300)
    plt.close(fig)
    print(f"wrote {PDF}")
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
