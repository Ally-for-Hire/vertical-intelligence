"""wrec.py: a world that is EXTREMELY SIMPLE in nature and EXTREMELY PROFOUND in depth (author 2026-11-03).
ONE rule, two constants, no random parameters: concept n = sin(g * (concept n-1 + x[n mod m] + e * (mean of all older concepts + all variables))), concept -1 = 0; the e-term (THIN) ties every concept thinly to everything older and to every variable, so nothing is unconnected. Concept n has depth n and is a nest of n+1 sines
over the environment's variables; the gain g = x_wrec_gain > 1 makes the function's oscillation grow with depth, so deep concepts cannot be written down by
any net of bounded size (the physical limit). Every concept is computable exactly (gradable). Same interface as winf.Library, so wstream.Stream uses it with
cfg['lib'] = 'rec' (the stream picks which levels appear in each problem, with the same reuse-weighted choice as winf)."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import numpy as np
import winf

GAIN = 2.0   # CONST: x_wrec_gain
THIN = 0.1   # CONST: x_wrec_thin: every concept is also thinly tied to EVERY earlier concept and EVERY environment variable (author 2026-11-03: a deep connection between everything, possibly thin, no unconnected nodes); 0 = the earlier chain-only world


class RecLibrary(winf.Library):
    def _raw(self, k, x, vals):
        prev = vals[:, k - 1] if k else 0.0
        older = vals[:, :k - 1].mean(1) if k >= 2 else 0.0
        return np.sin(GAIN * (prev + x[:, k % x.shape[1]] + THIN * (older + x.sum(1) / np.sqrt(x.shape[1]))))

    def pick(self, n, exclude=()):
        ok = [i for i in range(len(self.c)) if i not in exclude]      # the stream shows every level equally often: no reuse bias
        return list(self.rng.choice(ok, min(n, len(ok)), replace=False)) if ok else []

    def new(self, base=False):
        k = len(self.c)
        self.c.append(dict(op="rec", par=(k - 1,) if k else (), p={}, mu=0.0, sd=1.0, depth=k)); self.uses.append(0)
        raw = self._raw(k, self.ref, self.ref_vals)
        self.c[k]["mu"], self.c[k]["sd"] = float(raw.mean()), float(raw.std() + 1e-9)
        self.ref_vals = np.hstack([self.ref_vals, ((raw - self.c[k]["mu"]) / self.c[k]["sd"])[:, None]])
        return k
