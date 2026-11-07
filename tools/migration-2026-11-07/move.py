"""Step 1: git mv every tracked file of _in/<repo> to its family path. Step 2: carry working-tree-only files.
Usage: python move.py tracked | carry
"""
import os, subprocess, sys, shutil
sys.path.insert(0, os.path.dirname(__file__))
from mapping import dest

NEW = r"~/vertical-intelligence"
OLD = {"curious-net": r"~/curious-net", "neuroblock-research": r"~/neuroblock-research"}


def git(*a, cwd=NEW, inp=None):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, input=inp, check=True).stdout


def tracked():
    n = 0
    for repo in OLD:
        root = f"_in/{repo}"
        for f in filter(None, git("ls-files", "-z", root).split("\0")):
            rel = f[len(root) + 1:]
            d = dest(repo, rel)
            if d is None: continue
            os.makedirs(os.path.dirname(os.path.join(NEW, d)) or NEW, exist_ok=True)
            git("mv", f, d); n += 1
    print("moved", n)


def carry():
    """copy every file that exists in the old working tree (untracked, ignored, or locally modified) but differs/missing in NEW."""
    added = changed = 0
    for repo, old in OLD.items():
        for dp, dn, fn in os.walk(old):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".git")]
            for f in fn:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, old).replace("\\", "/")
                d = dest(repo, rel)
                if d is None: continue
                tgt = os.path.join(NEW, d)
                if os.path.exists(tgt):
                    if os.path.getsize(tgt) == os.path.getsize(full) and open(tgt, "rb").read() == open(full, "rb").read(): continue
                    changed += 1
                else: added += 1
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                shutil.copy2(full, tgt)
    print("added", added, "changed", changed)


if __name__ == "__main__":
    {"tracked": tracked, "carry": carry}[sys.argv[1]]()
