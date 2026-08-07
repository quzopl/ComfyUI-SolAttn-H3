"""Builds the README charts from measurements taken on an RTX 4070 Ti (SM89 / Triton).

Every number comes from a real run:
  * panel A — selftest.py on synthetic QKV, 56 heads, head_dim 128
  * panel B — MiniMax-H3 in ComfyUI, 864x480x125 (17504-row sequence), 8 steps

Colour is bound to the entity rather than to a position within the group:
Sol-Attn is always blue, SDPA orange, SageAttention aqua — in both panels.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = pathlib.Path(__file__).resolve().parent / "images"

# The dataviz reference palette, slots 1-3. Validated with validate_palette.js:
# light - CVD dE 9.2, normal 27.6; dark - CVD dE 9.4, normal 26.5.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "grid": "#dededa",
        "series": {"Sol-Attn": "#2a78d6", "SDPA": "#eb6834", "SageAttention": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "grid": "#3a3a37",
        "series": {"Sol-Attn": "#3987e5", "SDPA": "#d95926", "SageAttention": "#199e70"},
    },
}

# Panel A: ms per attention call, synthetic QKV (selftest.py)
KERNEL_X = ["8 192", "16 384", "30 976"]
KERNEL = {
    "Sol-Attn": [10.2, 31.7, 99.6],
    "SDPA": [26.4, 105.4, 380.0],
    "SageAttention": [11.2, 40.0, 133.6],
}
# Panel B: seconds, MiniMax-H3 in ComfyUI, 17504-row sequence, 8 steps
REAL_X = ["End-to-end", "Attention only"]
REAL = {
    "SDPA": [116.1, 45.5],     # node disabled - ComfyUI default attention
    "Sol-Attn": [84.2, 26.8],  # node enabled
}


def _style(ax, theme, *, ylabel):
    ax.set_facecolor(theme["surface"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=theme["secondary"], length=0, labelsize=9)
    ax.set_ylabel(ylabel, color=theme["secondary"], fontsize=9, labelpad=8)


def _grouped(ax, theme, categories, series, *, label_series, fmt, speedup_vs=None):
    """Grouped bars with a 2px gap between neighbours, labels on one chosen series."""
    n = len(series)
    span = 0.74
    width = span / n
    for index, (name, values) in enumerate(series.items()):
        offset = -span / 2 + width * (index + 0.5)
        positions = [x + offset for x in range(len(categories))]
        ax.bar(positions, values, width=width * 0.94, label=name,
               color=theme["series"][name], linewidth=0)
        if name != label_series:
            continue
        for x, value in zip(positions, values):
            ax.annotate(fmt(value), (x, value), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8.5,
                        color=theme["primary"], fontweight="medium")
        if speedup_vs is None:
            continue
        for x, value, other in zip(positions, values, series[speedup_vs]):
            ax.annotate(f"{other / value:.2f}x", (x, value), textcoords="offset points",
                        xytext=(0, 17), ha="center", fontsize=9, fontweight="semibold",
                        color=theme["primary"])
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, color=theme["secondary"], fontsize=9)


def build(mode: str) -> pathlib.Path:
    theme = THEMES[mode]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=200,
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(theme["surface"])

    _grouped(left, theme, KERNEL_X, KERNEL, label_series="Sol-Attn",
             fmt=lambda v: f"{v:.0f}", speedup_vs="SDPA")
    _style(left, theme, ylabel="ms per attention call")
    left.set_title("Kernel, synthetic QKV — 56 heads, head_dim 128",
                   color=theme["primary"], fontsize=10.5, fontweight="semibold",
                   loc="left", pad=12)
    left.set_xlabel("sequence length (rows)", color=theme["secondary"], fontsize=9,
                    labelpad=6)
    left.yaxis.set_major_locator(MultipleLocator(100))
    left.set_ylim(0, 440)

    _grouped(right, theme, REAL_X, REAL, label_series="Sol-Attn",
             fmt=lambda v: f"{v:.1f} s")
    _style(right, theme, ylabel="seconds")
    right.set_title("MiniMax-H3 in ComfyUI — 864x480, 125 frames, 8 steps",
                    color=theme["primary"], fontsize=10.5, fontweight="semibold",
                    loc="left", pad=12)
    right.set_xlabel("node disabled (SDPA)  vs  node enabled (Sol-Attn)",
                     color=theme["secondary"], fontsize=9, labelpad=6)
    right.set_ylim(0, 140)
    # The speedup ratio above each pair - that is the panel's actual message.
    for index, (off, on) in enumerate(zip(REAL["SDPA"], REAL["Sol-Attn"])):
        right.annotate(f"{off / on:.2f}x faster", (index, max(off, on)),
                       textcoords="offset points", xytext=(0, 22), ha="center",
                       fontsize=10, fontweight="semibold", color=theme["primary"])

    handles, labels = left.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                        fontsize=9.5, bbox_to_anchor=(0.5, -0.005),
                        handlelength=1.1, handleheight=1.1, columnspacing=2.2)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    fig.text(0.008, 0.965,
             "Measured on RTX 4070 Ti (SM89, Triton backend) — the slowest supported path",
             color=theme["secondary"], fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"benchmark-{mode}.png"
    fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print("written:", build(mode))
