"""Where every file of the two old repos goes in the unified vertical-intelligence repo.

dest(repo, rel) -> new relative path, or None to skip (bytecode/caches).
"""
import re

LEG = "experiments/self-gate-and-calibration"
SIM = "experiments/energy-brain-simulator"
EMG = "experiments/emergent-optimizers"
FAM = {
    "value": "experiments/value-and-self",
    "bored": "experiments/boredom-and-prediction",
    "self": "experiments/self-and-doubt",
    "l2l": "experiments/learning-to-learn",
    "deep": "experiments/deep-discovery",
}
DEEP_MODULES = {"wrec.py", "cell.py"}
SKIP_PARTS = {"__pycache__"}
SKIP_EXT = {".pyc"}


def family_of_xid(a, b):
    a, b = int(a), int(b)
    if a == 0: return "value"
    if a in (7, 8): return "self"
    if a == 1:
        if b <= 2: return "value"
        if b <= 4: return "bored"
        if b <= 28: return "l2l"
        return "deep"
    return None


def vi_family(name):
    """family key for a file in curious-net/vi, or None for shared infrastructure."""
    stem = name.rsplit(".", 1)[0]
    m = re.match(r"^x(\d+)_(\d+)", stem)
    if m: return family_of_xid(m.group(1), m.group(2))
    if name in DEEP_MODULES: return "deep"
    m = re.match(r"^_(?:pilot|smoke|t|tune)(\d+)", stem)      # scratch scripts named after an X1.<n> experiment
    if m: return family_of_xid(1, m.group(1))
    return None


def vi_result(name):
    """True when a curious-net/results entry belongs to the vi runs (stays in the shared root results/)."""
    return bool(re.match(r"^(x\d+_|_|decide_log|info_history|progress|wandb_local|vi_)", name))


def dest(repo, rel):
    p = rel.split("/")
    if SKIP_PARTS & set(p) or ("." + p[-1].rsplit(".", 1)[-1]) in SKIP_EXT: return None
    if repo == "curious-net":
        top = p[0]
        if top == "vi" and len(p) == 2:
            if p[1] == "README.md": return "core/README.md"
            f = vi_family(p[1])
            return f"{FAM[f]}/{p[1]}" if f else f"core/{p[1]}"
        if top in ("LOG.md", "HANDOFF.md", "CONSTANTS.md", ".gitignore") and len(p) == 1: return rel
        if top in ("paper", "notes", "worlds"): return rel
        if top in ("src", "experiments", "data"): return f"{LEG}/{rel}"
        if top == "results":
            if len(p) == 1: return "results"
            if p[1] == "figures":
                return f"results/{'/'.join(p[1:])}" if len(p) > 2 and p[2].startswith("vi_") else f"{LEG}/results/{'/'.join(p[1:])}"
            return f"results/{'/'.join(p[1:])}" if vi_result(p[1]) else f"{LEG}/results/{'/'.join(p[1:])}"
        return f"{LEG}/{rel}"                                  # anything unforeseen stays with the older family
    if repo == "neuroblock-research":
        if p[0] == "emergent": return f"{EMG}/{rel}"
        return f"{SIM}/{rel}"
    raise ValueError(repo)


if __name__ == "__main__":
    import subprocess, collections, sys
    c = collections.Counter()
    for repo, root in (("curious-net", "_in/curious-net"), ("neuroblock-research", "_in/neuroblock-research")):
        out = subprocess.run(["git", "ls-files", root], capture_output=True, text=True, cwd=sys.argv[1]).stdout.split("\n")
        for f in filter(None, out):
            d = dest(repo, f[len(root) + 1:])
            c["/".join(d.split("/")[:2]) if d else "SKIP"] += 1
    for k, v in sorted(c.items()): print(v, k)
