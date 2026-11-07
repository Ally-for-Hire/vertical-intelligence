"""Patch the moved code so paths resolve in the unified layout. Idempotent. Usage: python patch.py"""
import ast, os, re, sys
NEW = "~/vertical-intelligence"
FAMS = ["core", "experiments/value-and-self", "experiments/boredom-and-prediction", "experiments/self-and-doubt",
        "experiments/learning-to-learn", "experiments/deep-discovery", "experiments/emergent-optimizers/emergent"]
MARK = "# vi-bootstrap"
BOOT = ("import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents "
        "if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  " + MARK + "\n")


def read(p): return open(p, encoding="utf-8", newline="").read()
def write(p, s): open(p, "w", encoding="utf-8", newline="").write(s)


def insert_boot(src):
    if MARK in src: return src
    nl = "\r\n" if "\r\n" in src else "\n"
    tree = ast.parse(src)
    at = 0                                                      # line index (0-based) after the docstring / __future__ imports
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str)) \
           or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            at = node.end_lineno
        else: break
    lines = src.split(nl)
    lines.insert(at, BOOT.rstrip("\n"))
    return nl.join(lines)


def fix(src, rel):
    # results / worlds / paper joins relative to the script folder
    for what, const in (("results", "RESULTS"), ("worlds", "WORLDS"), ("paper", "PAPER")):
        src = re.sub(r'os\.path\.join\((?:os\.path\.dirname\(os\.path\.abspath\(__file__\)\)|HERE), "\.\.", "%s"' % what,
                     'os.path.join(str(vipaths.%s)' % const, src)
        src = re.sub(r'os\.path\.join\((?:os\.path\.dirname\(os\.path\.abspath\(__file__\)\)|HERE), "\.\.", "%s"\)' % what, 'str(vipaths.%s)' % const, src)
    src = src.replace('str(vipaths.RESULTS))', 'str(vipaths.RESULTS))')
    # x8_1*: ROOT idiom -> legacy folders
    src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)\nfor p in (os.path.join(ROOT, "experiments"), os.path.join(ROOT, "src"), HERE):\n    sys.path.insert(0, p)',
                      'HERE = os.path.dirname(os.path.abspath(__file__))\nfor p in (str(vipaths.LEGACY_EXP), str(vipaths.LEGACY_SRC), HERE):\n    sys.path.insert(0, p)')
    src = src.replace('OUT = os.path.join(ROOT, "results")', 'OUT = str(vipaths.RESULTS)')
    # emergent: hard-coded vi folder
    src = re.sub(r"sys\.path\.insert\(0, r?['\"]~/curious-net/vi['\"]\)", "pass", src)
    # figures: emergent runs
    src = src.replace('os.path.join(HERE, "..", "..", "neuroblock-research", "emergent", "runs", ', 'os.path.join(str(vipaths.EMERGENT), "runs", ')
    return src


def main():
    n = 0
    for fam in FAMS:
        d = os.path.join(NEW, fam)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or f == "vipaths.py": continue
            p = os.path.join(d, f); s = read(p); o = s
            s = fix(s, f)
            try: s = insert_boot(s)
            except SyntaxError as e: print("SYNTAX", p, e); continue
            if s != o: write(p, s); n += 1
    print("patched", n)

main()
