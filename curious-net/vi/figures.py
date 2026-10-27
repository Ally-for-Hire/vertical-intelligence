"""Figures for the public README, built from the result files (never from typed-in numbers).
Style follows the dataviz method: one accent hue plus grays for emphasis, thin marks, direct labels,
a title that states the finding, light AND dark variants (GitHub picks by the viewer theme); alt text and
a data table live in the README.

Usage: python figures.py [x12 | x11 | x02 | x06 | x07 | x66 | x67 | x68 | x69 | all]       -> ../results/figures/vi_<name>_{light,dark}.png"""
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


def fig_x66(theme):
    d = json.load(open(os.path.join(HERE, "..", "..", "neuroblock-research", "emergent", "runs", "x66_machinery.json")))["summary_judge"]
    names = {"b": "Keep going the parent's way", "c": "Mix two parents", "d": "Grow by copying a unit", "e": "All three together"}
    sub = {"b": "heritable direction", "c": "heritable recombination", "d": "heritable duplication", "e": "all three heritable"}
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.6, 4.9), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.30, right=0.97, top=0.76, bottom=0.14)
    arms = ["b", "c", "d", "e"]
    n = len(arms)
    for i, k in enumerate(arms):
        o = d[k]; y = n - 1 - i; v = o["cheaper_than_a"]
        hl = k == "d"
        ax.barh(y, v, height=0.36, color=t["accent"] if hl else t["gray"], linewidth=0)
        ax.text(-0.02, y + 0.12, names[k], ha="right", va="center", fontsize=10, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        ax.text(-0.02, y - 0.16, sub[k], ha="right", va="center", fontsize=8, color=t["ink2"])
        ax.text(1.42, y + 0.1, f"{v:.2f}× cheaper · solved {o['solved']}/{o['total']}", ha="left", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        gtxt = ", ".join(f"{g} {x['p']:.2g}" + ("↓" if x["median_shift"] < 0 else "↑") if x["p"] < 0.01 else f"{g} {x['p']:.2g}" for g, x in o["genes"].items())
        ax.text(1.42, y - 0.18, "gene sign test p (needs < 0.01): " + gtxt, ha="left", va="center", fontsize=7.8, color=t["ink2"])
    for x, lab, ha, dx in ((1.0, "plain baseline", "right", -0.02), (1.25, "pass line", "left", 0.02)):
        ax.axvline(x, color=t["ink2"], lw=0.9, ls=(0, (4, 3)))
        ax.text(x + dx, n - 0.45, lab, ha=ha, va="bottom", fontsize=8, color=t["ink2"])
    ax.set_xlim(0, 2.75); ax.set_ylim(-0.6, n - 0.25); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.935, "None of nature's three search tricks improved itself", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "Each trick was made inheritable and blind; the genes behind them mostly drifted instead of being selected.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "How many times cheaper than the plain evolutionary search (higher is better), 70 runs per arm.", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X6.6 · toy benchmark · pass needs 1.25× cheaper, as many solved, and genes under selection (p < 0.01)", fontsize=7.4, color=t["ink2"], ha="right")
    return _save(fig, "x66_machinery", theme)


def fig_x67(theme):
    d = json.load(open(os.path.join(HERE, "..", "..", "neuroblock-research", "emergent", "runs", "x67_energy.json")))["summary"]
    t = THEMES[theme]
    solved = [("Tuned gradient descent", d["adam_tuned"]["median_cost"], d["adam_tuned"]["solved"], False),
              ("Graded cull (a grader ranks and culls)", d["a"]["median_cost"], d["a"]["solved"], False),
              ("Default gradient descent", d["adam0.001"]["median_cost"], d["adam0.001"]["solved"], False),
              ("Energy world (nothing ranks anything)", d["e"]["median_cost"], d["e"]["solved"], True)]
    unsolved = [("Energy world, literal units", f"not solved: the population went extinct in {d['b']['extinct']}/{d['b']['total']} runs"),
                ("Unlimited food", f"not solved: the population exploded in {d['c']['exploded']}/{d['c']['total']} runs"),
                ("Random search", f"not solved in {d['d']['total']}/{d['d']['total']} runs within 3,000,000")]
    rows = solved + [(a, None, None, False) for a, _ in unsolved]
    n = len(rows); lo, hi = 1e3, 6e7
    ratio = d["e"]["median_cost"] / d["a"]["median_cost"]
    fig, ax = plt.subplots(figsize=(10.4, 5.2), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.33, right=0.97, top=0.75, bottom=0.14)
    ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(-0.7, n - 0.3); ax.set_yticks([])
    for i, (label, v, k, hl) in enumerate(rows):
        y = n - 1 - i
        ax.text(lo * 0.82, y, label, ha="right", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal", clip_on=False)
        if v is not None:
            ax.barh(y, v - lo, left=lo, height=0.38, color=t["accent"] if hl else t["gray"], linewidth=0)
            ax.text(v * 1.12, y, f"{v:,.0f} · solved {k}/20", ha="left", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal", zorder=5, bbox=dict(facecolor=t["surface"], edgecolor="none", pad=2.0))
        else:
            ax.text(lo * 1.15, y, dict(unsolved)[label], ha="left", va="center", fontsize=9, color=t["ink2"], style="italic")
    ax.axvline(3e6, color=t["ink2"], lw=0.9, ls=(0, (4, 3))); ax.text(3e6 * 0.9, n - 0.4, "run budget", ha="right", va="bottom", fontsize=8, color=t["ink2"])
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7]); ax.minorticks_off()
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0, which="both")
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, which="major"); ax.set_axisbelow(True)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.935, f"Selection with no grader works, but is about {ratio:.0f}× costlier than ranking", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "In the energy world nothing ranks the learners: they split when they have energy to spare and die without it.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "Cost to reach 95% accuracy on a noisy XOR, median of 20 runs (log scale; lower is better).", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X6.7 · toy problem · the energy world misses the registered 18/20 by one run", fontsize=7.6, color=t["ink2"], ha="right")
    return _save(fig, "x67_nograder", theme)


def fig_x68(theme):
    d = json.load(open(os.path.join(HERE, "..", "..", "neuroblock-research", "emergent", "runs", "x68_stones.json")))["summary"]
    names = {"a": "Direct (the full problem only)", "b": "Fixed ramp (easy stages on a timer)", "c": "Self-paced ramp (advance when solved)", "d": "Mixed (all stages at once)"}
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10.4, 4.9), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.37, right=0.97, top=0.76, bottom=0.14)
    arms = ["a", "b", "c", "d"]; n = len(arms)
    for i, k in enumerate(arms):
        o = d[k]; y = n - 1 - i; hl = k == "c"
        if o["solved"] > 0:
            ax.barh(y, o["solved"], height=0.36, color=t["accent"] if hl else t["gray"], linewidth=0)
        ax.text(-0.4, y, names[k], ha="right", va="center", fontsize=10, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        ax.text(max(o["solved"], 0) + 0.5, y + 0.1, f"solved {o['solved']}/{o['total']}", ha="left", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        sp, ck = o["solved_by_problem"]["spirals"], o["solved_by_problem"]["checker3"]
        med = o["median_cost"]["spirals"]
        tail = f"spiral median cost {med:,.0f}" if sp >= 5 else "spirals mostly unsolved at 10,000,000"
        ax.text(max(o["solved"], 0) + 0.5, y - 0.18, f"spirals {sp}/10 · checkerboard {ck}/10 · {tail}", ha="left", va="center", fontsize=7.8, color=t["ink2"])
    ax.set_xlim(0, 32); ax.set_ylim(-0.6, n - 0.3); ax.set_yticks([]); ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp_ in ("top", "right", "left", "bottom"):
        ax.spines[sp_].set_visible(False)
    fig.text(0.03, 0.935, "Stepping stones did not help, and the checkerboard stayed unsolved", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "Easier versions solved first, with their cost counted: two of three ramps made things worse.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "Runs solved out of 20 (10 spirals, 10 checkerboard) within a budget of 10,000,000 (higher is better).", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X6.8 · toy problems · no ramp arm passed the registered rule", fontsize=7.6, color=t["ink2"], ha="right")
    return _save(fig, "x68_stones", theme)


def fig_x69(theme):
    d = json.load(open(os.path.join(HERE, "..", "..", "neuroblock-research", "emergent", "runs", "x69_growth.json")))["summary_judge"]
    names = {"g10": "Copy a unit in 10% of children", "g30": "Copy a unit in 30% of children", "g50": "Copy a unit in 50% of children", "g100": "Copy a unit in every child"}
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10.0, 4.9), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.30, right=0.97, top=0.76, bottom=0.14)
    arms = ["g10", "g30", "g50", "g100"]; n = len(arms)
    best = max(arms, key=lambda k: d[k]["cheaper_than_a"])
    for i, k in enumerate(arms):
        o = d[k]; y = n - 1 - i; hl = k == best; v = o["cheaper_than_a"]
        ax.barh(y, v, height=0.36, color=t["accent"] if hl else t["gray"], linewidth=0)
        ax.text(-0.02, y, names[k], ha="right", va="center", fontsize=10, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        ax.text(1.42, y + 0.1, f"{v:.2f}× cheaper overall · solved {o['solved']}/{o['total']}", ha="left", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal")
        ax.text(1.42, y - 0.18, f"spirals {o['spirals_vs_a']:.2f}× cheaper · net width grew to {o['width']:.0f}", ha="left", va="center", fontsize=7.8, color=t["ink2"])
    for x, lab, ha, dx in ((1.0, "plain baseline", "right", -0.02), (1.25, "pass line", "left", 0.02)):
        ax.axvline(x, color=t["ink2"], lw=0.9, ls=(0, (4, 3))); ax.text(x + dx, n - 0.45, lab, ha=ha, va="bottom", fontsize=8, color=t["ink2"])
    ax.set_xlim(0, 2.75); ax.set_ylim(-0.6, n - 0.25); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0); ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.935, "Letting the search grow its own parts helps a little, and none reach the pass line", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "Copying a hidden unit (without changing what the network computes) at a fixed rate, with no gene to evolve.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "How many times cheaper than the plain evolutionary search (higher is better), 70 runs per arm.", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X6.9 · toy benchmark · the best arm missed the registered 1.25× line by 0.02 · the checkerboard stayed unsolved", fontsize=7.4, color=t["ink2"], ha="right")
    return _save(fig, "x69_growth", theme)


def fig_x02(theme):
    d = json.load(open(os.path.join(RES, "x0_2_run.json")))["result"]["summary"]
    worlds = ["0.01", "0.05", "0.25"]; xl = ["Mild threat" + chr(10) + "(staying out still pays)", "Medium threat", "Severe threat" + chr(10) + "(only home is sensible)"]
    series = [("TWOF", "Two drives, no switch"), ("TWOS", "Two drives, hand-built switch"), ("B0", "Random exploring"), ("ONE", "One signal (no weight, no switch)")]
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10.0, 4.9), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.09, right=0.70, top=0.76, bottom=0.17)
    for k, lab in series:
        ys = [100 * d[w]["explore_threat"][k] for w in worlds]; hl = k == "ONE"
        ax.plot([0, 1, 2], ys, color=t["accent"] if hl else t["gray"], lw=3.0 if hl else 1.8, marker="o", ms=7 if hl else 5, zorder=3 if hl else 2, solid_capstyle="round")
        ly = {"B0": 2.6, "TWOF": 1.4, "ONE": 0.2, "TWOS": -1.0}[k]
        ax.annotate(lab, xy=(2.0, ys[-1]), xytext=(2.14, ly), ha="left", va="center", fontsize=9.5, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal",
                    arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.7, shrinkA=2, shrinkB=4), annotation_clip=False)
    ax.set_xlim(-0.2, 2.2); ax.set_ylim(-1.5, 11.0); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(xl, fontsize=9, color=t["ink2"])
    ax.set_ylabel("share of threat ticks spent exploring (%)", fontsize=9, color=t["ink2"])
    ax.tick_params(axis="y", colors=t["ink2"], labelsize=8.5, length=0); ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.935, "One signal explores under threat only when staying out pays", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "It retreats in proportion to the stakes with no weight and no switch; hand-built versions need constants.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "Share of threat ticks spent visiting a patch other than the one believed best, median of 48 simulated worlds.", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X0.2 · toy worlds · a tuned two-drive agent still had slightly lower regret in the medium and severe worlds", fontsize=7.4, color=t["ink2"], ha="right")
    return _save(fig, "x02_secure_base", theme)


def fig_x11(theme):
    S = json.load(open(os.path.join(RES, "x1_1_run.json")))["result"]["summary"]
    t = THEMES[theme]
    arms = [("UR", "Minimize surprise"), ("NOV", "Seek surprise"), ("LP", "Chase learning progress"), ("RANDOM", "Choose at random")]
    fig, ax = plt.subplots(figsize=(10.0, 4.9), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.24, right=0.97, top=0.70, bottom=0.14)
    n = len(arms); cols = (("Dark room (nothing to learn)", "dark_b", t["gray"]), ("Noise (unlearnable)", "noise_b", t["accent2"]), ("Learnable regions", None, t["accent"]))
    for i, (k, lab) in enumerate(arms):
        y = n - 1 - i; left = 0.0
        parts = [S["dark_b"][k], S["noise_b"][k], S["l1_b"][k] + S["l2_b"][k]]
        tot = sum(parts); parts = [100 * p / tot for p in parts]
        for (name, _, col), w in zip(cols, parts):
            ax.barh(y, w, left=left, height=0.5, color=col, edgecolor=t["surface"], linewidth=1.5)
            if w >= 7:
                ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=9.5, color="#ffffff" if col != t["gray"] else t["ink"], fontweight="bold")
            left += w
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=10, color=t["ink"] if k == "LP" else t["ink2"], fontweight="bold" if k == "LP" else "normal")
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    x0 = 0.24
    for name, _, col in cols:
        fig.patches.append(plt.Rectangle((x0, 0.755), 0.018, 0.028, transform=fig.transFigure, facecolor=col, edgecolor="none"))
        fig.text(x0 + 0.024, 0.769, name, fontsize=9, color=t["ink2"], va="center"); x0 += 0.25
    fig.text(0.03, 0.935, "Only learning progress avoids both the dark room and the noisy TV", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.885, "Where each drive spends its time once a second learnable region has opened (median of 48 simulated worlds).", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, f"experiment X1.1 · toy world · learning progress took about {S['t_l2']['LP'] / S['t_l2']['RANDOM']:.0f}× longer than random to learn the newly opened region", fontsize=7.6, color=t["ink2"], ha="right")
    return _save(fig, "x11_dark_room", theme)


def fig_x12(theme):
    S = json.load(open(os.path.join(RES, "x1_2_run.json")))["summary"]
    names = {"ORACLE": "Perfect learner (ceiling)", "LPO": "Learning progress + optimism", "NOVO": "Seek surprise + optimism", "RANDOM": "Choose at random",
             "LP": "Learning progress (plain)", "UR": "Minimize surprise", "NOV": "Seek surprise (plain)"}
    order = sorted(names, key=lambda k: S["t_l2"][k]); t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.14)
    n = len(order); lo, hi = 5.0, 40000.0
    ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([]); ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1,000"]); ax.minorticks_off()
    for i, k in enumerate(order):
        y = n - 1 - i; v = S["t_l2"][k]; hl = k == "LPO"
        ax.barh(y, v - lo, left=lo, height=0.4, color=t["accent"] if hl else t["gray"], linewidth=0)
        ax.text(lo * 0.9, y, names[k], ha="right", va="center", fontsize=9.8, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal", clip_on=False)
        trap = 100 * S["trap_a"][k]
        ax.text(v * 1.15, y, (f"{v:,.0f} ticks (rests in the dark room once everything is learned)" if k == "ORACLE" else f"{v:,.0f} ticks · {trap:.0f}% of the time in the traps"), ha="left", va="center", fontsize=8.8, color=t["ink"] if hl else t["ink2"], fontweight="bold" if hl else "normal", zorder=5, bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.5))
    ax.tick_params(axis="x", colors=t["ink2"], labelsize=8.5, length=0); ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.text(0.03, 0.935, f"An optimistic start fixes learning progress: {S['t_l2']['LP'] / S['t_l2']['LPO']:.0f}× faster at finding a new thing to learn", fontsize=13.5, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.03, 0.875, "Untried regions start at the best reward the drive could pay, with nothing tuned; the trap avoidance is kept.", fontsize=10, color=t["ink2"], ha="left")
    fig.text(0.03, 0.84, "Ticks to learn a newly opened learnable region (median of 48 worlds, log scale; lower is better) and time spent in the traps.", fontsize=9, color=t["ink2"], ha="left")
    fig.text(0.97, 0.025, "experiment X1.2 · toy world · traps = the perfectly predictable dark room and the unlearnable noise, first half of the run", fontsize=7.6, color=t["ink2"], ha="right")
    return _save(fig, "x12_optimism", theme)


FIGS = {"x12": fig_x12, "x11": fig_x11, "x02": fig_x02, "x06": fig_x06, "x07": fig_x07, "x66": fig_x66, "x67": fig_x67, "x68": fig_x68, "x69": fig_x69}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in FIGS.items():
        if which in (name, "all"):
            for th in THEMES:
                print(fn(th))
