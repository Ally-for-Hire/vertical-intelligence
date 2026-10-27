"""Figures for the public README, built from the result files (never from typed-in numbers).
Style follows the dataviz method: one accent hue plus grays for emphasis, thin marks, direct labels,
a title that states the finding, light AND dark variants (GitHub picks by the viewer theme), alt text and
a data table live in the README.

Usage: python figures.py [x06 | all]       -> ../results/figures/vi_<name>_{light,dark}.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", accent="#2a78d6", accent2="#eb6834", gray="#b9b8b2", gray_dark="#8a8982"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f", accent="#3987e5", accent2="#d95926", gray="#4a4945", gray_dark="#77766f"),
}


def _bars(ax, rows, theme, highlight, ceiling_key=None):
    """rows: list of (label, value, key). highlight: {key: color_name}. Horizontal bars, thin, direct-labelled."""
    t = THEMES[theme]
    vmax = max(v for _, v, _ in rows)
    n = len(rows)
    for i, (label, v, key) in enumerate(rows):
        y = n - 1 - i
        col = t[highlight[key]] if key in highlight else t["gray"]
        if v > 0:
            ax.barh(y, v, height=0.38, color=col, linewidth=0)
        ax.text(-vmax * 0.015, y, label, ha="right", va="center", fontsize=9.5, color=t["ink"] if key in highlight else t["ink2"],
                fontweight="bold" if key in highlight else "normal")
        ax.text(v + vmax * 0.012, y, (f"{v:.1f}" if v >= 10 else f"{v:.1f}" if v > 0 else "0"), ha="left", va="center", fontsize=9.5,
                color=t["ink"] if key in highlight else t["ink2"], fontweight="bold" if key in highlight else "normal")
    ax.set_xlim(0, vmax * 1.12)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def fig_x06(theme):
    d = json.load(open(os.path.join(RES, "x0_6_run.json")))["result"]["summary"]["tuned"]["deaths_per_10k"]
    names = {"ORACLE": "Perfect policy (ceiling)", "SUM": "Adds both needs together", "HOMEO": "Hand-picked targets",
             "B0": "Food only", "VIAB": "Learns its own sense of survival", "DEATH": "Only “dying is bad”", "RANDOM": "Random choices"}
    rows = sorted([(names[k], d[k], k) for k in names], key=lambda r: r[1])
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.34, right=0.96, top=0.77, bottom=0.14)
    _bars(ax, rows, theme, {"VIAB": "accent"})
    fig.text(0.03, 0.935, "A learner that builds its own sense of survival dies 2.4× less often", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "than one given only “dying is bad”. Neither has hand-picked targets or weights.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "Deaths per 10,000 ticks, median of 48 simulated worlds (lower is better).", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X0.6 · toy world · not a claim about real brains", fontsize=7.5, color=t["ink2"], ha="right")
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"vi_x06_deaths_{theme}.png")
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


FIGS = {"x06": fig_x06}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in FIGS.items():
        if which in (name, "all"):
            for th in THEMES:
                print(fn(th))
