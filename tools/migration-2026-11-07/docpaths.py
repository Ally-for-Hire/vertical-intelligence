"""Translate old-layout paths inside a text file: curious-net/<x> and neuroblock-research/<x> -> unified layout.
Also bare `vi/<script>.py` (no repo prefix) -> its family folder. Usage: python docpaths.py file [file...]   (in place)
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mapping import dest, vi_family, FAM

DIR_FIX = {  # directory-style mentions
    "curious-net/vi/": "experiments/ and core/", "curious-net/vi": "experiments/ and core/",
    "curious-net/": "./", "neuroblock-research/": "experiments/energy-brain-simulator/",
    "curious-net": "the repo root", "neuroblock-research": "experiments/energy-brain-simulator",
}


def one(repo, rest):
    d = dest(repo, rest)
    return d


def tr_text(s):
    def prefixed(m):
        repo, rest = m.group(1), m.group(2)
        tok = m.group(0)
        if rest == "" or rest.endswith("/") or "." not in os.path.basename(rest) and rest in ("vi", "paper", "results", "worlds", "src", "experiments", "data", "emergent", "notes"):
            if repo == "curious-net":
                return {"vi": "experiments/ and core/", "paper": "paper/", "results": "results/", "worlds": "worlds/", "notes": "notes/",
                        "src": "experiments/self-gate-and-calibration/src/", "experiments": "experiments/self-gate-and-calibration/experiments/",
                        "data": "experiments/self-gate-and-calibration/data/"}.get(rest.rstrip("/"), tok)
            if rest.rstrip("/") == "emergent": return "experiments/emergent-optimizers/emergent/"
            return tok
        d = one(repo, rest)
        return d if d else tok
    s = re.sub(r"(?<![\w/.-])(curious-net|neuroblock-research)/((?:[\w.\-]+/)*[\w.\-]*)", prefixed, s)
    # bare vi/<script>.py
    def bare(m):
        name = m.group(1)
        f = vi_family(name)
        return f"{FAM[f]}/{name}" if f else f"core/{name}"
    s = re.sub(r"(?<![\w/.-])vi/([\w]+\.(?:py|sh|md))", bare, s)
    return s


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = open(p, encoding="utf-8", newline="").read()
        t = tr_text(s)
        if t != s: open(p, "w", encoding="utf-8", newline="").write(t); print("translated", p)
