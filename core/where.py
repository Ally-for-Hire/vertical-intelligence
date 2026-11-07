"""Find where an experiment lives now. Older logs, the register and the wiki say `vi/x1_47.py` or `curious-net/vi/...`;
those scripts moved to experiments/<family>/ and core/ in the 2026-11-07 reorganization.

    python core/where.py x1_47        -> path of the script, plus its result files
    python core/where.py              -> every family and its scripts
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import vipaths  # vi-bootstrap

if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        for d in vipaths.CODE_DIRS:
            print(f"{d.relative_to(vipaths.ROOT)}: {len(list(d.glob('*.py')))} scripts")
    for n in names:
        p = vipaths.find(n)
        print(f"{n}: {p.relative_to(vipaths.ROOT) if p else 'not found'}")
        for r in sorted(vipaths.RESULTS.glob(f"{n}_*")):
            print(f"   {r.relative_to(vipaths.ROOT)}")
