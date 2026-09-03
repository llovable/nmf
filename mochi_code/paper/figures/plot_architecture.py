#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""그림 2. MOCHI 아키텍처.

(A) 인코딩 → 융합 → 디코딩. 융합은 은닉 평균(주 경로)에 Transformer 잔차를 더한다.
    디코딩은 선형 디코더에 저랭크 NMF 잔차 γ(Ŵ − W̄)H 를 더한다.
(B) 두 추론 갈래. 블록 결측은 leave-one-omics-out, 칸 결측은 자기 은닉·계수를 ω=10으로 섞는다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
PDF = HERE / "fig2_architecture.pdf"
PNG = HERE / "fig2_architecture.png"

COL = {
    "rna": "#0072B2",
    "methyl": "#56B4E9",
    "protein": "#CC79A7",
    "fuse": "#009E73",
    "nmf": "#D55E00",
    "neutral": "#4D4D4D",
    "fill": "#F7F7F7",
    "line": "#222222",
    "soft": "#E8E8E8",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#222222", tc="#222222",
        lw=0.8, fs=7.5, weight="normal", rounding=0.08, va="center"):
    if weight == "medium":
        weight = "normal"
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.6,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va=va,
            fontsize=fs, color=tc, fontweight=weight, linespacing=1.25)
    return p


def arrow(ax, x1, y1, x2, y2, color="#222222", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=8, lw=lw, color=color,
        shrinkA=0.4, shrinkB=0.4,
    ))


def panel_label(ax, letter, x, y):
    ax.text(x, y, letter, fontsize=12, fontweight="bold", va="center", ha="left")


def draw_a(ax):
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.15, 8.35)
    ax.axis("off")
    panel_label(ax, "A", 0.05, 8.1)
    ax.text(0.45, 8.08, "Encoding, fusion, and decoding", fontsize=9, va="center")

    # column headers
    ax.text(2.15, 7.55, "Inputs", ha="center", fontsize=8, color="#555555")
    ax.text(5.55, 7.55, "Modality AE  +  frozen NMF", ha="center", fontsize=8, color="#555555")
    ax.text(10.15, 7.55, "Fusion  z", ha="center", fontsize=8, color="#555555")
    ax.text(15.15, 7.55, "Target decode", ha="center", fontsize=8, color="#555555")

    # three input rows: protein, rna, methyl (top to bottom, matching MODS visual)
    rows = [
        dict(name="Protein", dim="487", h="128", col=COL["protein"], y=5.85),
        dict(name="RNA", dim="2000", h="512", col=COL["rna"], y=3.85),
        dict(name="Methyl", dim="5000", h="256", col=COL["methyl"], y=1.85),
    ]

    for r in rows:
        y = r["y"]
        box(ax, 0.9, y + 0.15, 2.5, 0.95,
            f"{r['name']}\n$x$  ({r['dim']})",
            fc="#FFFFFF", ec=r["col"], tc=r["col"], lw=1.15, fs=7.5, weight="medium")
        arrow(ax, 3.45, y + 0.62, 4.15, y + 0.62, color=r["col"])
        box(ax, 4.2, y + 0.42, 1.55, 0.42, f"$E$ → $h$ ({r['h']})",
            fc="#FFFFFF", ec=r["col"], tc=r["col"], fs=6.6)
        box(ax, 4.2, y, 1.55, 0.38, "NMF  $W$ (20)",
            fc="#FFF4EC", ec=COL["nmf"], tc=COL["nmf"], fs=6.6)
        # to mean / tokens
        arrow(ax, 5.8, y + 0.63, 7.05, 4.95, color=r["col"], lw=0.7)
        arrow(ax, 5.8, y + 0.19, 7.05, 3.15, color=COL["nmf"], lw=0.7)

    # fusion stack
    box(ax, 7.1, 4.55, 3.9, 1.15,
        "Mean of present $h$\n$\\bar{z}$  (main path)",
        fc="#EAF7F1", ec=COL["fuse"], tc=COL["fuse"], lw=1.1, fs=7.4, weight="medium")
    box(ax, 7.1, 2.35, 3.9, 1.85,
        "Tokens: $h$ + $W$  (k = 20 each)\n"
        "Transformer encoder  (2 layers)\n"
        "Query attention  →  $\\delta$\n"
        "$\\delta$ starts at zero",
        fc="#FFF8F2", ec=COL["nmf"], tc="#5A2A00", lw=1.0, fs=6.8)

    arrow(ax, 9.05, 4.55, 9.05, 4.25, color=COL["neutral"])
    box(ax, 7.55, 1.35, 3.0, 0.85,
        "$z = \\mathrm{LN}(\\bar{z}+\\delta)$",
        fc="#EAF7F1", ec=COL["fuse"], tc=COL["fuse"], lw=1.15, fs=8, weight="medium")
    arrow(ax, 9.05, 2.35, 9.05, 2.22, color=COL["fuse"])

    # decode
    arrow(ax, 10.55, 1.78, 12.15, 5.55, color=COL["fuse"], lw=0.8)
    arrow(ax, 10.55, 1.78, 12.15, 3.25, color=COL["nmf"], lw=0.8)

    box(ax, 12.2, 5.15, 5.3, 1.05,
        "Linear map  $T_t$   →   target hidden $h_t$\n"
        "Decoder  $D_t(h_t)$",
        fc="#FFFFFF", ec=COL["fuse"], tc="#1A4D38", fs=7.2)
    box(ax, 12.2, 2.85, 5.3, 1.45,
        "Coefficient head  $\\hat{W}_t=\\mathrm{softplus}(A_t z+b_t)$\n"
        "Low-rank residual  (trained $\\gamma_t$)\n"
        "$\\gamma_t\\,(\\hat{W}_t-\\bar{W}_t)\\,H_t$",
        fc="#FFF4EC", ec=COL["nmf"], tc=COL["nmf"], lw=1.2, fs=7.2, weight="medium")

    arrow(ax, 14.85, 5.15, 14.85, 4.55, color=COL["neutral"])
    arrow(ax, 14.85, 2.85, 14.85, 2.15, color=COL["nmf"])
    box(ax, 12.2, 1.25, 5.3, 0.85,
        r"$\hat{x}_t = D_t(h_t) + \gamma_t(\hat{W}_t-\bar{W}_t)H_t$",
        fc="#FFF4EC", ec=COL["nmf"], tc="#8A3000", lw=1.25, fs=8, weight="medium")

    # stage brackets
    ax.annotate("", xy=(6.85, 0.55), xytext=(0.7, 0.55),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
    ax.annotate("", xy=(11.95, 0.55), xytext=(7.05, 0.55),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
    ax.annotate("", xy=(17.6, 0.55), xytext=(12.1, 0.55),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
    ax.text(3.75, 0.38, "encode", ha="center", fontsize=7, color="#888888")
    ax.text(9.5, 0.38, "fuse", ha="center", fontsize=7, color="#888888")
    ax.text(14.85, 0.38, "decode  (NMF enters here)", ha="center", fontsize=7, color=COL["nmf"])


def draw_b(ax):
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.1, 3.55)
    ax.axis("off")
    panel_label(ax, "B", 0.05, 3.32)
    ax.text(0.45, 3.3, "Two inference paths share the same weights", fontsize=9, va="center")

    # Block missing
    box(ax, 0.7, 0.35, 8.15, 2.55, "", fc="#F3F8FC", ec=COL["rna"], lw=1.0, rounding=0.12)
    ax.text(4.78, 2.58, "Block missing  (target absent)", ha="center",
            fontsize=8, fontweight="bold", color=COL["rna"])
    box(ax, 1.0, 1.45, 2.15, 0.85, "Present\nRNA + Methyl",
        fc="#FFFFFF", ec=COL["rna"], tc=COL["rna"], fs=7)
    arrow(ax, 3.2, 1.88, 3.7, 1.88, color=COL["rna"])
    box(ax, 3.75, 1.45, 2.15, 0.85, "Fuse\nskip target",
        fc="#FFFFFF", ec=COL["fuse"], tc=COL["fuse"], fs=7)
    arrow(ax, 5.95, 1.88, 6.45, 1.88, color=COL["fuse"])
    box(ax, 6.5, 1.35, 2.1, 1.05, "LOO only\n$h^{\\mathrm{LOO}},\\hat{W}$",
        fc="#FFFFFF", ec=COL["nmf"], tc=COL["nmf"], fs=7)
    ax.text(4.78, 0.7, r"$\hat{x}_t = D_t(h_t^{\mathrm{LOO}}) + \gamma_t(\hat{W}_t-\bar{W}_t)H_t$",
            ha="center", fontsize=7, color="#333333")

    # Cell missing
    box(ax, 9.35, 0.35, 8.15, 2.55, "", fc="#F4FBF7", ec=COL["fuse"], lw=1.0, rounding=0.12)
    ax.text(13.42, 2.58, "Cell missing  (partial target remains)", ha="center",
            fontsize=8, fontweight="bold", color=COL["fuse"])
    box(ax, 9.65, 1.45, 2.05, 0.85, "Own $h, W$\nfrom $x_t$",
        fc="#FFFFFF", ec=COL["fuse"], tc=COL["fuse"], fs=7)
    box(ax, 12.0, 1.45, 2.05, 0.85, "LOO $h, W$\nfrom others",
        fc="#FFFFFF", ec=COL["nmf"], tc=COL["nmf"], fs=7)
    arrow(ax, 11.75, 1.88, 12.0, 1.88, color=COL["neutral"])
    arrow(ax, 14.1, 1.88, 14.55, 1.88, color=COL["neutral"])
    box(ax, 14.6, 1.35, 2.6, 1.05, r"mix $\omega=10$" + "\n" + r"$(10\cdot\mathrm{own}+\mathrm{LOO})/11$",
        fc="#FFFFFF", ec=COL["line"], tc="#222222", fs=6.8)
    ax.text(13.42, 0.7, "Same decoder + low-rank residual as in A",
            ha="center", fontsize=7, color="#333333")


def main():
    style()
    fig = plt.figure(figsize=(7.4, 6.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0], hspace=0.08)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    draw_a(ax_a)
    draw_b(ax_b)
    fig.savefig(PDF)
    fig.savefig(PNG, dpi=300)
    plt.close(fig)
    print(f"saved {PDF}")
    print(f"saved {PNG}")


if __name__ == "__main__":
    main()
