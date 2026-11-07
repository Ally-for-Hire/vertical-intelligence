"""X1.33 on the one-rule world (vi/wrec.py): extremely simple in nature, extremely profound in depth. Same ladder and grades as x1_33.py. Usage: python x1_33r.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import x1_33

if __name__ == "__main__":
    x1_33.run("rec")
