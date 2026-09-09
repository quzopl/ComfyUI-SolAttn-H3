"""Builds the README charts from selftest.py runs on four GPUs.

Every panel is the same measurement at the same sequence lengths, so they are
directly comparable: kernel-only time per attention call, 56 heads, head_dim 128,
tau=1.0, thresh_type=diag, on the CuTe DSL backend each card resolves to.

The grid is architecture x segment, so either axis can be read on its own:

              consumer                 datacentre
  Ada         RTX 4070 Ti  cute_sm89   L40                    cute_sm89
  Blackwell   RTX 5080     cute_sm120  RTX PRO 6000 Blackwell cute_sm120

Reading down a column isolates the architecture at a fixed segment; reading
across a row isolates the memory system at a fixed kernel. That matters because
the two move independently: the Blackwell row is faster in absolute terms
everywhere, but the *ratio* to SageAttention is larger on the datacentre part
than on the consumer one, so neither axis alone predicts what a card will give.

The L40 was not measured at 30 976 rows — 48 GB could not hold the tensors — and
that gap is drawn as a gap, with a label. Dropping the category instead would
misalign the axes and quietly imply the run was comparable when it was shorter.

Colour is bound to the entity rather than to a position within the group:
Sol-Attn is always blue, SDPA orange, SageAttention aqua — in every panel.
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

LENGTHS = ["8 192", "16 384", "30 976"]
# Panel A: RTX 4070 Ti, backend cute_sm89
SM89 = {
    "Sol-Attn": [8.70, 27.42, 79.55],
    "SDPA": [26.06, 106.49, 373.51],
    "SageAttention": [11.40, 39.81, 131.57],
}
# Panel B: L40, backend cute_sm89. None = not measured, drawn as a gap.
L40 = {
    "Sol-Attn": [3.93, 13.40, None],
    "SDPA": [13.02, 50.00, None],
    "SageAttention": [5.77, 24.63, None],
}
# Panel C: RTX 5080, 16 GB, backend cute_sm120
SM120_CONSUMER = {
    "Sol-Attn": [5.47, 17.14, 52.85],
    "SDPA": [18.84, 75.39, 267.33],
    "SageAttention": [6.94, 25.66, 87.62],
}
# Panel D: RTX PRO 6000 Blackwell, backend cute_sm120
SM120 = {
    "Sol-Attn": [1.97, 5.53, 16.58],
    "SDPA": [5.88, 21.99, 77.45],
    "SageAttention": [3.63, 12.50, 42.23],
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
        # A missing measurement is drawn as nothing, not as zero: a zero-height
        # bar reads as "instant", which is the opposite of "not run".
        drawn = [(x, v) for x, v in zip(positions, values) if v is not None]
        ax.bar([x for x, _ in drawn], [v for _, v in drawn], width=width * 0.94,
               label=name, color=theme["series"][name], linewidth=0)
        if name != label_series:
            continue
        for x, value in drawn:
            ax.annotate(fmt(value), (x, value), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8.5,
                        color=theme["primary"], fontweight="medium")
        for slot, (x, value) in enumerate(zip(positions, values)):
            if value is None:
                ax.annotate("not\nmeasured", (slot, 0), textcoords="offset points",
                            xytext=(0, 14), ha="center", va="bottom", fontsize=8,
                            color=theme["secondary"], style="italic")
        if speedup_vs is None:
            continue
        for x, value, other in zip(positions, values, series[speedup_vs]):
            if value is None or other is None:
                continue
            ax.annotate(f"{other / value:.2f}x", (x, value), textcoords="offset points",
                        xytext=(0, 17), ha="center", fontsize=9, fontweight="semibold",
                        color=theme["primary"])
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, color=theme["secondary"], fontsize=9)


def build(mode: str) -> pathlib.Path:
    theme = THEMES[mode]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    (top_left, top_right), (bottom_left, bottom_right) = axes

    for ax, data, title, top, tick in (
        (top_left, SM89, "RTX 4070 Ti — backend cute_sm89", 430, 100),
        (top_right, L40, "L40 — backend cute_sm89", 60, 20),
        (bottom_left, SM120_CONSUMER, "RTX 5080 — backend cute_sm120", 300, 100),
        (bottom_right, SM120, "RTX PRO 6000 Blackwell — backend cute_sm120", 90, 20),
    ):
        _grouped(ax, theme, LENGTHS, data, label_series="Sol-Attn",
                 fmt=lambda v: f"{v:.1f}", speedup_vs="SageAttention")
        _style(ax, theme, ylabel="ms per attention call")
        ax.set_title(title, color=theme["primary"], fontsize=10.5,
                     fontweight="semibold", loc="left", pad=12)
        ax.set_xlabel("sequence length (rows)", color=theme["secondary"],
                      fontsize=9, labelpad=6)
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.set_ylim(0, top)

    handles, labels = top_left.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                        fontsize=9.5, bbox_to_anchor=(0.5, -0.005),
                        handlelength=1.1, handleheight=1.1, columnspacing=2.2)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    fig.text(0.008, 0.975,
             "selftest.py, kernel only — 56 heads, head_dim 128, tau=1.0, diag. "
             "Labels on the Sol-Attn bars are the speedup against SageAttention. "
             "Top row Ada, bottom row Blackwell; left column consumer, right column datacentre.",
             color=theme["secondary"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"benchmark-{mode}.png"
    fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print("written:", build(mode))
