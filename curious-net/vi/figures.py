"""Figures for the public README, built from the result files (never from typed-in numbers).
Style follows the dataviz method: one accent hue plus grays for emphasis, thin marks, direct labels,
a title that states the finding, light AND dark variants (GitHub picks by the viewer theme); alt text and
a data table live in the README.

Usage: python figures.py [x06 | x07 | all]       -> ../results/figures/vi_<name>_{light,dark}.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", accent="#2a78d6", accent2="#eb6834", gray="#b9b8b2"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f", accent="#3987e5", accent2="#d95926", gray="#4a4945"),
}


def _fmt(v):
    return "0" if v == 0 else f"{v:.1f}"


def _bars(ax, rows, theme, highlight, labels=True):
    """rows: list of (label, value, key). highlight: {key: color_name}. Horizontal bars, thin, direct-labelled."""
    t = THEMES[theme]
    vmax = max(v for _, v, _ in rows)
    n = len(rows)
    for i, (label, v, key) in enumerate(rows):
        y = n - 1 - i
        hl = key in highlight
        if v > 0:
            ax.barh(y, v, height=0.38, color=t[highlight[key]] if hl else t["gray"], linewidth=0)
        if labels:
            ax.text(-vmax * 0.02, y, label, ha="right", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        ax.text(v + vmax * 0.015, y, _fmt(v), ha="left", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
    ax.set_xlim(0, vmax * 1.14)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _save(fig, name, theme):
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"vi_{name}_{theme}.png")
    fig.savefig(out, facecolor=THEMES[theme]["surface"])
    plt.close(fig)
    return out


def fig_x06(theme):
    d = json.load(open(os.path.join(RES, "x0_6_run.json")))["result"]["summary"]["tuned"]["deaths_per_10k"]
    names = {"ORACLE": "Perfect policy (ceiling)", "SUM": "Adds both needs together", "HOMEO": "Hand-picked targets",
             "B0": "Food only", "VIAB": "Learns its own sense of survival", "DEATH": "Only “dying is bad”", "RANDOM": "Random choices"}
    rows = sorted([(names[k], d[k], k) for k in names], key=lambda r: r[1])
    t = THEMES[theme]
    ratio = d["DEATH"] / d["VIAB"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.34, right=0.96, top=0.77, bottom=0.14)
    _bars(ax, rows, theme, {"VIAB": "accent"})
    fig.text(0.03, 0.935, f"A learner that builds its own sense of survival dies {ratio:.1f}× less often", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "than one given only “dying is bad”. Neither has hand-picked targets or weights.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "Deaths per 10,000 ticks, median of 48 simulated worlds (lower is better).", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X0.6 · toy world · not a claim about real brains", fontsize=7.5, color=t["ink2"], ha="right")
    return _save(fig, "x06_deaths", theme)


def fig_x07(theme):
    S = json.load(open(os.path.join(RES, "x0_7_run.json")))["result"]["summary"]
    orth, trade = S["orth/tuned"]["deaths_per_10k"], S["trade/tuned"]["deaths_per_10k"]
    names = {"ORACLE": "Perfect policy (ceiling)", "SUM": "Adds both needs together", "MARGIN": "Weakest need first (no set numbers)",
             "HOMEO": "Hand-picked targets", "B0": "Food only (ignores one need)", "VIAB": "Learns from deaths alone", "RANDOM": "Random choices"}
    order = sorted(names, key=lambda k: orth[k])
    ratios = [trade["B0"] / trade[k] for k in ("SUM", "HOMEO", "MARGIN")]
    b0_default = S["trade/default"]["deaths_per_10k"]["B0"]
    t = THEMES[theme]
    fig = plt.figure(figsize=(11.2, 4.9), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    axl = fig.add_axes([0.255, 0.13, 0.30, 0.60]); axr = fig.add_axes([0.655, 0.13, 0.30, 0.60])
    for ax in (axl, axr):
        ax.set_facecolor(t["surface"])
    _bars(axl, [(names[k], orth[k], k) for k in order], theme, {"B0": "accent2"}, labels=True)
    _bars(axr, [(names[k], trade[k], k) for k in order], theme, {"B0": "accent2"}, labels=False)
    axl.set_title("Gentle world: needs rarely conflict", loc="left", fontsize=10, color=t["ink2"], pad=8)
    axr.set_title("Trade-off world: needs often conflict", loc="left", fontsize=10, color=t["ink2"], pad=8)
    fig.text(0.02, 0.945, f"A learner that ignores one of its needs dies {min(ratios):.0f}–{max(ratios):.0f}× more often when needs conflict", fontsize=14, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.02, 0.885, "How the needs are combined (sum, hand-picked targets, weakest first) barely matters; watching all of them does.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.02, 0.85, "Deaths per 10,000 ticks, median of 48 simulated worlds per panel (lower is better).", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.98, 0.025, f"experiment X0.7 · toy worlds · food-only learner tuned on food, with default settings it dies {b0_default:.0f} per 10,000 in the trade-off world · not a claim about real brains",
             fontsize=7.2, color=t["ink2"], ha="right")
    return _save(fig, "x07_needs", theme)


FIGS = {"x06": fig_x06, "x07": fig_x07}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in FIGS.items():
        if which in (name, "all"):
            for th in THEMES:
                print(fn(th))
