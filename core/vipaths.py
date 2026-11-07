"""Where things are in the unified vertical-intelligence repo, and the import path for every experiment script.

Every script in core/ and in the code families starts with a two-line bootstrap that finds this file and imports it;
importing it puts core/ and every code family on sys.path, so `import x0_1`, `from constants import C` and
`import budget` work from any script exactly as they did when all of it sat in one folder.

    ROOT      repo root
    CORE      shared code (constants, worlds, learners, run control)
    EXP       experiments/<family>/ folders
    RESULTS   one shared results/ folder for the vi runs (cross-experiment reads need one place); each family README lists its files
    WORLDS, PAPER   saved worlds, the prediction register and write-ups
    LEGACY_SRC, LEGACY_EXP   the older self-gate family's library and scripts (x8_1* import from them)
    CODE_DIRS all folders searched for modules; find(name) -> path of the script called <name>.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
EXP = ROOT / "experiments"
RESULTS = ROOT / "results"
WORLDS = ROOT / "worlds"
PAPER = ROOT / "paper"
CODE_FAMILIES = ["value-and-self", "boredom-and-prediction", "self-and-doubt", "learning-to-learn", "deep-discovery"]
LEGACY = EXP / "self-gate-and-calibration"
LEGACY_SRC = LEGACY / "src"
LEGACY_EXP = LEGACY / "experiments"
SIM = EXP / "energy-brain-simulator"            # holds car/, probnet/, selfscore/ ... (a package root for `import car.env`)
EMERGENT = EXP / "emergent-optimizers" / "emergent"
CODE_DIRS = [CORE] + [EXP / f for f in CODE_FAMILIES]

for _p in reversed([str(p) for p in CODE_DIRS + [SIM]]):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def find(name):
    """Path of <name>.py among the code folders, or None."""
    for d in CODE_DIRS:
        p = d / f"{name}.py"
        if p.exists():
            return p
    return None
