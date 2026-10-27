"""Audit: every numeric literal in the agent code must be (a) trivial (0, 1, 2, -1, 0.5), (b) read through the registry C("name"),
or (c) on a line tagged  # CONST: name  where `name` is declared in constants.py. env.py (the world, not the agent) is exempt as a file.
Prints undeclared literals (= problems) and the registry's ARBITRARY entries (= known problems)."""
import io, re, sys, tokenize, pathlib
import constants

TRIVIAL = {"0", "1", "2", "0.5", "1.0", "0.0", "2.0", "3", "4", "6", "8", "1e9", "30", "100"}   # 2.0: log base; 1e9/30/100: numerical sentinels and percent scale   # structural counts (tuple sizes, 6 actions, 3 action bits); see constants.STRUCTURAL
EXEMPT_FILES = {"w0.py", "constants.py", "audit.py", "figures.py"}   # w0.py is the WORLD (like src/env.py)
here = pathlib.Path(__file__).parent
bad = []
for f in sorted(here.glob("*.py")):
    if f.name in EXEMPT_FILES:
        continue
    src = f.read_text(encoding="utf-8")
    lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NUMBER and tok.string not in TRIVIAL and tok.string.lower() not in ("1e-9", "1e-12", "1e-6", "1e-3", "1e-300"):
            ln = lines[tok.start[0] - 1]
            m = re.search(r"#\s*CONST:\s*([\w,]+)", ln)
            if m and all(n in constants.REG for n in m.group(1).split(",")):
                continue
            if 'C("' in ln and re.search(r'C\("(\w+)"\)', ln) and all(n in constants.REG for n in re.findall(r'C\("(\w+)"\)', ln)):
                continue
            bad.append((f.name, tok.start[0], tok.string, ln.strip()[:90]))
print(f"{len(bad)} undeclared numeric literals")
for b in bad:
    print("  ", *b)
arb = constants.problems()
print(f"\n{len(arb)} declared ARBITRARY constants (each is an open problem):")
for n, e in arb:
    print(f"  {n} = {e['value']}   [{e['where']}]  {e['note']}")
sys.exit(1 if bad else 0)
