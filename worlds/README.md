# Saved worlds (for future experiments)

Each file is one realized world: the full concept library (every concept's operation, parents, parameters and standardisation), every problem
(which concepts, mixing weights, stage, whether it carries a new concept) and the expansion points. Example points are not stored; each problem
regenerates its own points from its seed, bit-exactly.

```python
import sys; sys.path.insert(0, "vi")
import wstream
data, info = wstream.load("worlds/x1_19/world_870000.json")   # list of problems in the same format every vi/ learner takes
```

| Folder | Made by | What | Notes |
|---|---|---|---|
| `x1_19/` | `vi/x1_19.py run` (2026-11-02) | 48 train-and-expand loops (seeds 870000-870047), 240 problems each, 24 concepts at the start, +12 per expansion (5-7 expansions, up to 108 concepts) | expansion times were set by the flat.OPT learner's own stagnation signal, so these worlds replay that learner's curriculum. `world_871000.json` is the unscored timing world |
| `x1_20/` | `vi/x1_20.py run` (2026-11-02) | 48 loops (seeds 880000-880047), 400 problems each, stagnation window 30 (stages ~60), 4-6 expansions, up to ~96 concepts | same format |
| `x1_21/` | `vi/x1_21.py run` (2026-11-02) | 48 loops (seeds 890000-890047), 400 problems, stages set by a fixed internal probe (40-150 problems), 2-7 expansions | same format |

Generator: `vi/wstream.py` (grammar from `vi/winf.py`). A world is finite at every moment; it grows only at its recorded expansion points.
