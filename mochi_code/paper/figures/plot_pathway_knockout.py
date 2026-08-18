#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""그림 1. 경로 진폭 보존과 저랭크 녹아웃.

원자료: source_pathway_knockout.csv
  A — BRCA RNA 블록, Hallmark 경로 표준편차 비·상관 (표 10)
  B — 세 암종 RNA 경로 표준편차 비, γ=0 녹아웃 (표 11)
  C — 녹아웃 − 보고 모형의 블록 z-RMSE (양수 = 저랭크가 오차를 줄임)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
CSV = HERE / "source_pathway_knockout.csv"
PDF = HERE / "fig1_pathway_knockout.pdf"
PNG = HERE / "fig1_pathway_knockout.png"

COL = {
    "mean": "#BBBBBB",
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


def draw_a(ax, df):
    order = ["Mean", "Ridge 2→1", "MIMIR", "MOCHI"]
    colors = [COL["mean"], COL["ridge"], COL["mimir"], COL["mochi"]]
    sub = df[(df.panel == "A") & (df.metric == "pathway_sd_ratio")].set_index("method")
    rsub = df[(df.panel == "A") & (df.metric == "pathway_r")].set_index("method")
    vals = [float(sub.loc[m, "value"]) for m in order]
    rs = [float(rsub.loc[m, "value"]) for m in order]
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="none", zorder=2)
    ax.axvline(1.0, color=COL["line"], ls="--", lw=0.7, zorder=1)
    for i, (v, r) in enumerate(zip(vals, rs)):
        if v <= 0.02:
            ax.text(0.03, i, "0.00  (r = 0.00)", va="center", ha="left", color="#555555")
        else:
            ax.text(v + 0.02, i, f"{v:.3f}   r = {r:.3f}", va="center", ha="left", color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.38)
    ax.set_xlabel("Hallmark pathway SD ratio  (1 = truth)")
    ax.set_title("BRCA RNA block-missing", loc="left", pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.invert_yaxis()
    panel_label(ax, "A")


def draw_b(ax, df):
    cohorts = ["BRCA", "LUAD", "KIRC"]
    y = np.arange(len(cohorts))
    on = []
    off = []
    for c in cohorts:
        on.append(float(df[(df.panel == "B") & (df.cohort == c) & (df.method == "MOCHI")].value.iloc[0]))
        off.append(float(df[(df.panel == "B") & (df.cohort == c) & (df.method == "Knockout")].value.iloc[0]))
    for i, (a, b) in enumerate(zip(on, off)):
        rel = 100 * (a - b) / a
        ax.plot([b, a], [i, i], color="#888888", lw=1.4, zorder=1)
        ax.plot(b, i, "o", ms=7.5, color=COL["knock"], zorder=3, label="γ = 0" if i == 0 else None)
        ax.plot(a, i, "o", ms=7.5, color=COL["mochi"], zorder=3, label="MOCHI" if i == 0 else None)
        ax.text((a + b) / 2, i + 0.18, f"−{rel:.0f}%", ha="center", va="bottom", fontsize=7, color="#444444")
        ax.text(a + 0.025, i, f"{a:.3f}", va="center", ha="left", fontsize=7)
        ax.text(b - 0.025, i, f"{b:.3f}", va="center", ha="right", fontsize=7)
    ax.axvline(1.0, color=COL["line"], ls="--", lw=0.7, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(cohorts)
    ax.set_xlim(0.25, 1.18)
    ax.set_xlabel("Hallmark pathway SD ratio")
    ax.set_title("Low-rank knockout (same weights)", loc="left", pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4, borderaxespad=0.2)
    panel_label(ax, "B")


def draw_c(ax, df):
    cohorts = ["BRCA", "LUAD", "KIRC"]
    omics = [("rna", "RNA"), ("methyl", "Methylation"), ("protein", "Protein")]
    x = np.arange(len(cohorts))
    width = 0.24
    for j, (key, lab) in enumerate(omics):
        vals = []
        for c in cohorts:
            vals.append(float(df[(df.panel == "C") & (df.cohort == c) & (df.omics == key)].value.iloc[0]))
        ax.bar(
            x + (j - 1) * width, vals, width=width, color=COL[key],
            edgecolor="none", label=lab, zorder=2,
        )
    ax.axhline(0, color=COL["line"], lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Δ z-RMSE  (knockout − MOCHI)")
    ax.set_title("Reconstruction cost of the residual", loc="left", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.set_ylim(-0.042, 0.055)
    ax.legend(frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.2)
    ax.text(
        2.22, 0.048, "RNA / methyl: NMF helps",
        fontsize=6.5, color="#0072B2", ha="right", va="top",
    )
    ax.text(
        2.22, -0.038, "Protein: NMF hurts error",
        fontsize=6.5, color="#CC79A7", ha="right", va="bottom",
    )
    panel_label(ax, "C")


def main():
    style()
    df = pd.read_csv(CSV)
    fig = plt.figure(figsize=(7.2, 4.55), dpi=300)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 0.95], hspace=0.42, wspace=0.38,
        left=0.10, right=0.98, top=0.90, bottom=0.10,
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
