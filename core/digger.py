"""digger.py: a learner that CHOOSES its inputs, builds functions from anything it has, and is trained on TOTAL MODEL SIZE in one of three readings
(author, 2026-11-02, inbox I22: 'dig until stone'; 'try all the types and record findings').

Functions (nodes): ('f', u, b, ()) flat tanh(u.x + b); ('s', theta, par) concept node = tiny 2-layer net on [x, outputs of nodes `par`] (functions of
the environment's variables AND of its own functions). Per problem the learner sees a POOL of candidate inputs and picks which to label: x128_rand at
random, then x128_chosen where two halves of its current model disagree most (its own experiment). Decision and test points are never chosen.
Readings of 'training on total model size' (cfg['size']):
  "mdl"    keep an addition only if it SHRINKS total description length: bits gained on decision points minus its model bits
           (new node: 0.5 log2(n) per weight; link: log2(#nodes + 1) + 0.5 log2(n));
  "grow"   bigger is better: keep any addition that helps on BOTH halves of the decision points (confirmed), no size charge;
  "budget" a total weight budget (x128_budget0) that only grows (x1.5) after x128_stall problems in a row hit it; additions judged by flat.OPT's bar.
API: Learner(seed, cfg).step(problem) -> dict(r2 on test, bits); .recovered(lib, probe) -> per-concept best |corr|"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import math
import numpy as np
import flat
from flat import vbits, adjusted, B1, B2, EPS, FLOOR

HID = 6                                                                # CONST: x127_hidden
DEFAULT = dict(flat.OPT, size="mdl", hid=HID, mdl_w=0.5, n_sub=2, n_par=2, rand=60, chosen=0, mix_rand=20, mix_chosen=40, pick="rand", node_bits=0.0, cap=0, flat_off=False, clone=False, bar=False, max_templates=16, parent_pick="corr", reuse_first=False, rule="none", rule_min=2, rule_fid=0.0, prune_tol=0.0, budget0=600, stall_budget=10, ft_patience=30)   # CONST: x128_cfg


def n_in(nd, m): return m + len(nd[2]) if nd[0] == "s" else m


def nparams(nd, m): return (m + 1) if nd[0] == "f" else len(nd[1])


def unpack(th, k):
    H = (len(th) - 1) // (k + 2)                                       # hidden size read from the node itself (cfg["hid"] at creation)
    W1 = th[:k * H].reshape(k, H); b1 = th[k * H:k * H + H]; w2 = th[k * H + H:k * H + 2 * H]; return W1, b1, w2, th[-1]


def forward(nd, z):
    if nd[0] == "f": return np.tanh(z @ nd[1] + nd[2])
    W1, b1, w2, b2 = unpack(nd[1], z.shape[1]); return np.tanh(np.tanh(z @ W1 + b1) @ w2 + b2)


def node_input(nd, x, H):
    return x if (nd[0] == "f" or not nd[2]) else np.hstack([x, H[:, list(nd[2])]])


def grad(nd, z, dh):
    if nd[0] == "f":
        h = np.tanh(z @ nd[1] + nd[2]); dz = dh * (1 - h * h); return np.concatenate([z.T @ dz, [dz.sum()]])
    W1, b1, w2, b2 = unpack(nd[1], z.shape[1]); a = np.tanh(z @ W1 + b1); h = np.tanh(a @ w2 + b2); dz = dh * (1 - h * h)
    da = np.outer(dz, w2) * (1 - a * a); return np.concatenate([(z.T @ da).ravel(), da.sum(0), a.T @ dz, [dz.sum()]])


def pvec(nd): return np.concatenate([nd[1], [nd[2]]]) if nd[0] == "f" else nd[1]


def rebuild(nd, p): return ("f", p[:-1].copy(), float(p[-1]), ()) if nd[0] == "f" else ("s", p.copy(), nd[2])


def all_act(units, x):
    H = np.zeros((len(x), len(units)))
    for i, nd in enumerate(units): H[:, i] = forward(nd, node_input(nd, x, H))
    return H


class Cache:
    def __init__(self, xtr, ytr, xva, yva, units):
        self.xtr, self.ytr, self.xva, self.yva = xtr, ytr, xva, yva; self.tr = all_act(units, xtr); self.va = all_act(units, xva)

    def add(self, nd):
        self.tr = np.hstack([self.tr, forward(nd, node_input(nd, self.xtr, self.tr))[:, None]])
        self.va = np.hstack([self.va, forward(nd, node_input(nd, self.xva, self.va))[:, None]])

    def refresh(self, units, first):
        for i in range(first, len(units)):
            self.tr[:, i] = forward(units[i], node_input(units[i], self.xtr, self.tr)); self.va[:, i] = forward(units[i], node_input(units[i], self.xva, self.va))


def train_new(rng, c, r, units, cfg, kind):
    """Candidates fitted to the residual: flat nodes on x, or concept nodes on [x, the n_par existing nodes best matching the residual]."""
    if kind == "f": return [("f", u, b, ()) for u, b in flat.train_units(rng, c.xtr, r, cfg)]
    out, n = [], len(units)
    if cfg["parent_pick"] == "all": sets, reps = [()] + [(i,) for i in range(n)], 1      # ask of EVERY known function: is this problem a simple function of it and the inputs?
    elif cfg["parent_pick"] == "none": sets, reps = [()], cfg["n_sub"]
    else:
        par = ()
        if n:
            rc = np.abs((c.tr - c.tr.mean(0)).T @ (r - r.mean())) / (c.tr.std(0) * r.std() * len(r) + 1e-9)
            par = tuple(sorted(int(i) for i in np.argsort(-rc)[:cfg["n_par"]]))
        sets, reps = [par], cfg["n_sub"]
    for par in sets:
        z = np.hstack([c.xtr, c.tr[:, list(par)]]) if par else c.xtr; k = z.shape[1]
        for _ in range(reps):
            H = cfg["hid"]; th = np.concatenate([rng.standard_normal(k * H) / math.sqrt(k), np.zeros(H), rng.standard_normal(H) / math.sqrt(H), [0.0]])
            P = [th, np.array([cfg["a0"]])]; mo = [np.zeros_like(p) for p in P]; ve = [np.zeros_like(p) for p in P]; last = float(np.mean(r ** 2))
            for s in range(1, cfg["unit_steps"] + 1):
                nd = ("s", P[0], par); h = forward(nd, z); e = P[1][0] * h - r
                g = [grad(nd, z, e * P[1][0] * (2.0 / len(r))), np.array([2.0 * np.mean(e * h)])]
                for i in range(2):
                    mo[i] = B1 * mo[i] + (1 - B1) * g[i]; ve[i] = B2 * ve[i] + (1 - B2) * g[i] * g[i]
                    P[i] = P[i] - cfg["unit_lr"] * (mo[i] / (1 - B1 ** s)) / (np.sqrt(ve[i] / (1 - B2 ** s)) + EPS)
                if s % cfg["check"] == 0:
                    cur = float(e @ e) / len(r)
                    if cur > last * (1 - cfg["stall"]): break
                    last = cur
            out.append(("s", P[0].copy(), par))
    if cfg.get("clone_eager") and n:      # (old eager version, slow; kept for reference) PATTERNS IN PATTERNS: reuse a learned computation with a different parent and a different input wiring (no training)
        m = c.xtr.shape[1]; H = cfg["hid"]
        for tq in [nd for nd in units if nd[0] == "s" and len(nd[2]) == 1]:
            W1, b1, w2, b2 = unpack(tq[1], m + 1); W1 = W1.copy()
            for sft in range(m):
                Wp = W1.copy(); Wp[:m] = np.roll(W1[:m], -sft, axis=0)
                th = np.concatenate([Wp.ravel(), b1, w2, [b2]])
                for p in range(n):
                    if sft == 0 and p == tq[2][0]: continue
                    out.append(("s", th.copy(), (p,)))
    return out


class Learner:
    def __init__(self, seed, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {})); self.rng = np.random.default_rng([seed, flat.STREAM])
        self.units, self.deg, self.heads, self.caches = [], [], [], []; self.budget = self.cfg["budget0"]; self.hits = 0; self.m = None

    def size(self): return sum(nparams(nd, self.m) for nd in self.units)

    def link_step(self, c, E, t, allow_new, new_from=0):
        cfg, units = self.cfg, self.units; nva = len(c.yva); rid = cfg["ridge"]; h = nva // 2; created = []; hit = False
        for _ in range(cfg["max_steps"]):
            F, Fv, Gi, beta = flat.solve(c, E, rid); r = c.ytr - F @ beta; pv = Fv @ beta
            idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)
            props = (([] if cfg["flat_off"] else train_new(self.rng, c, r, units, cfg, "f")) + train_new(self.rng, c, r, units, cfg, "s")) if allow_new else []
            Z = np.hstack([c.tr[:, idx]] + [forward(nd, node_input(nd, c.xtr, c.tr))[:, None] for nd in props]) if (len(idx) or props) else np.zeros((len(c.ytr), 0))
            Zv = np.hstack([c.va[:, idx]] + [forward(nd, node_input(nd, c.xva, c.va))[:, None] for nd in props]) if (len(idx) or props) else np.zeros((nva, 0))
            specs, csz = [], 0
            if allow_new and cfg["clone"] and units:
                specs, Zc, Zcv, csz = self.clone_cols(c)
                if specs: Z = np.hstack([Z, Zc]); Zv = np.hstack([Zv, Zcv])
            K = Z.shape[1]
            if K == 0: break
            Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + rid - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / np.maximum(sch, 1e-12)
            err = c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc))
            help_ = vbits(err, nva) - float(vbits(c.yva - pv, nva))
            isnew = np.zeros(K, bool); isnew[len(idx):] = True
            sizes = np.array([0] * len(idx) + [nparams(nd, self.m) for nd in props] + [csz] * len(specs), float)
            if cfg["size"] == "mdl":
                model_bits = np.where(isnew, sizes * cfg["mdl_w"] * math.log2(len(c.ytr)) + cfg["node_bits"], 0.0) + math.log2(len(units) + K + 1) + 0.5 * math.log2(len(c.ytr))
                v = help_ - model_bits; ok = v > 0
            elif cfg["size"] == "grow":
                hA = vbits(err[:h], h) - float(vbits(c.yva[:h] - pv[:h], h)); hB = vbits(err[h:], nva - h) - float(vbits(c.yva[h:] - pv[h:], nva - h))
                v = hA; ok = (hA > 0) & (hB > 0)
                if cfg["bar"]: ok = ok & (help_ > math.log2(K + 1))     # many candidates -> some help by chance: the evidence bar grows with the number tried
                if cfg["cap"]: ok = ok & ~(isnew & (self.size() + sizes > cfg["cap"]))       # hard capacity: the net cannot grow past cap weights
            else:
                dg = np.zeros(K)
                if len(idx): dg[:len(idx)] = np.asarray(self.deg, float)[idx]
                v = adjusted(help_, isnew, dg, t, cfg); ok = v > cfg["cost_scale"] * math.log2(K + 1)
                over = isnew & (self.size() + sizes > self.budget)
                if (ok & over).any() and not (ok & ~over).any(): hit = True
                ok = ok & ~over
            if not ok.any(): break
            k = int(np.argmax(np.where(ok, v, -np.inf)))
            if isnew[k]:
                kk = k - len(idx); nd = props[kk] if kk < len(props) else self.make_clone(specs[kk - len(props)]); units.append(nd); self.deg.append(0); j = len(units) - 1; created.append(j)
                for cx in self.caches:
                    if cx.tr.shape[1] < len(units): cx.add(nd)
            else: j = int(idx[k])
            E.append(j); self.deg[j] += 1
        return E, created, hit

    def templates(self):
        """Learned computations worth reusing: concept nodes with exactly one parent, the most used first (at most x135_max_templates)."""
        T = [i for i, nd in enumerate(self.units) if nd[0] == "s" and len(nd[2]) == 1]
        return sorted(T, key=lambda i: -self.deg[i])[:self.cfg["max_templates"]]

    def clone_cols(self, c):
        """PATTERNS IN PATTERNS: every (template, input rotation, parent) clone evaluated at once; only the winner is ever built."""
        m = c.xtr.shape[1]; specs, A, Av = [], [], []; T = self.templates(); U = len(self.units)
        for ti in T:
            tq = self.units[ti]; W1, b1, w2, b2 = unpack(tq[1], m + 1); W1x, wp = W1[:m], W1[m]
            for sft in range(m):
                Wx = np.roll(W1x, -sft, axis=0)
                for X, Hc, store in ((c.xtr, c.tr, A), (c.xva, c.va, Av)):
                    pre = (X @ Wx)[:, None, :] + Hc[:, :U, None] * wp + b1
                    store.append(np.tanh(np.tanh(pre) @ w2 + b2))
                specs += [(ti, sft, p) for p in range(U)]
        if not specs: return [], None, None, 0
        Zc, Zcv = np.hstack(A), np.hstack(Av); keep = np.array([not (sf == 0 and p == self.units[ti][2][0]) for ti, sf, p in specs])
        return [sp for sp, kp in zip(specs, keep) if kp], Zc[:, keep], Zcv[:, keep], len(self.units[T[0]][1])

    def make_clone(self, spec):
        ti, sft, p = spec; tq = self.units[ti]; m = self.m; W1, b1, w2, b2 = unpack(tq[1], m + 1); Wp = W1.copy(); Wp[:m] = np.roll(W1[:m], -sft, axis=0)
        return ("s", np.concatenate([Wp.ravel(), b1, w2, [b2]]), (p,))

    def choose(self, d, cfg=None):
        """Pick which pool points to label: cfg rand random, then cfg chosen where two half-models disagree most."""
        cfg = cfg or self.cfg; xp, yp = d["tr"]; n = len(yp); rng = self.rng
        pick = list(rng.choice(n, cfg["rand"], replace=False))
        if cfg["chosen"]:
            rest = np.array([i for i in range(n) if i not in set(pick)])
            if self.units:
                Hp = all_act(self.units, xp); Hs = Hp[pick]; ys = yp[pick]
                rc = np.abs((Hs - Hs.mean(0)).T @ (ys - ys.mean())) / (Hs.std(0) * ys.std() * len(ys) + 1e-9); top = np.argsort(-rc)[:cfg["max_steps"]]
                F = np.hstack([np.ones((len(pick), 1)), Hs[:, top]]); Fr = np.hstack([np.ones((len(rest), 1)), Hp[rest][:, top]])
                hh = len(pick) // 2
                b1 = np.linalg.solve(F[:hh].T @ F[:hh] + cfg["ridge"] * np.eye(F.shape[1]), F[:hh].T @ ys[:hh])
                b2 = np.linalg.solve(F[hh:].T @ F[hh:] + cfg["ridge"] * np.eye(F.shape[1]), F[hh:].T @ ys[hh:])
                dis = np.abs(Fr @ b1 - Fr @ b2); pick += list(rest[np.argsort(-dis)[:cfg["chosen"]]])
            else: pick += list(rng.choice(rest, cfg["chosen"], replace=False))
        return np.array(pick)

    def step(self, d):
        if self.cfg["pick"] == "auto":    # optional evolution: learn both ways, keep the better
            # random is one of the learner's choices: learn the problem both ways on copies, keep the one with lower decision-point error
            import copy
            best = None
            for mode in ("rand", "mix"):
                L = copy.deepcopy(self); L.cfg = dict(L.cfg, pick=mode); res = L._step(d)
                if best is None or res[1] < best[1]: best = (L, res[1], res[0], mode)
            self.__dict__.update(best[0].__dict__); self.cfg = dict(best[0].cfg, pick="auto"); self.chose = getattr(self, "chose", []) + [best[3]]
            return best[2]
        return self._step(d)[0]

    def _step(self, d):
        cfg = self.cfg; self.m = d["tr"][0].shape[1]; t = len(self.heads)
        if cfg["pick"] == "rand": pick = np.array(self.rng.choice(len(d["tr"][1]), cfg["rand"] + cfg["chosen"], replace=False))     # DEFAULT: all random
        else: pick = self.choose(d, dict(cfg, rand=cfg["mix_rand"], chosen=cfg["mix_chosen"]))                                      # optional: some chosen
        xtr, ytr = d["tr"][0][pick], d["tr"][1][pick]
        c = Cache(xtr, ytr, d["va"][0], d["va"][1], self.units); self.caches.append(c)
        n0 = len(self.units)
        if cfg["reuse_first"]:      # simplest = reuse what is already known; build new functions only for what is left over
            E, created, hit = self.link_step(c, [], t, False); E, created, hit = self.link_step(c, E, t, True)
        else: E, created, hit = self.link_step(c, [], t, True)
        self.hits = self.hits + 1 if hit else 0
        if cfg["size"] == "budget" and self.hits >= cfg["stall_budget"]: self.budget *= 1.5; self.hits = 0     # CONST: x128_cfg
        if E: self.train(E, c)
        if cfg["prune_tol"] and len(E) > 1: E = self.prune(E, c, cfg["prune_tol"])
        if cfg["rule"] != "none" and len(E) >= cfg["rule_min"]: self.make_rule(d, E, c, t)
        if cfg["sweep"] and created:
            for s in range(t): self.heads[s] = self.link_step(self.caches[s], self.heads[s], t, False, new_from=n0)[0]
        self.heads.append(E)
        F = np.hstack([np.ones((len(c.ytr), 1)), c.tr[:, E]]) if E else np.ones((len(c.ytr), 1))
        Fv = np.hstack([np.ones((len(c.yva), 1)), c.va[:, E]]) if E else np.ones((len(c.yva), 1))
        beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(F.shape[1]), F.T @ c.ytr)
        return self.score(E, d, xtr, ytr), float(np.mean((c.yva - Fv @ beta) ** 2))

    def prune(self, E, c, tol):
        """Smallest set that still solves the problem: drop members while the decision-point error rises by less than tol (relative)."""
        rid = self.cfg["ridge"]
        def val(S):
            F = np.hstack([np.ones((len(c.ytr), 1)), c.tr[:, S]]); Fv = np.hstack([np.ones((len(c.yva), 1)), c.va[:, S]])
            return float(np.mean((c.yva - Fv @ np.linalg.solve(F.T @ F + rid * np.eye(F.shape[1]), F.T @ c.ytr)) ** 2))
        E = list(E); base = val(E)
        while len(E) > 1:
            cand = min(((val([x for x in E if x != j]), j) for j in E)); 
            if cand[0] <= base * (1 + tol): E.remove(cand[1])
            else: break
        return E

    def make_rule(self, d, E, c, t):
        """After solving a problem, distil the solution into ONE reusable function of the inputs (and known functions): fitted to the solution's own
        predictions on all pool inputs (no extra labels). kind: flat / sub / best (higher fidelity of the two). Kept if fidelity (corr^2 on the pool) >= rule_fid."""
        cfg, units = self.cfg, self.units; xp = d["tr"][0]
        xf = np.vstack([c.xtr, c.xva]); yf = np.concatenate([c.ytr, c.yva]); Af = all_act(units, xf)[:, E]; Ap = all_act(units, xp)[:, E]
        F = np.hstack([np.ones((len(yf), 1)), Af]); beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(F.shape[1]), F.T @ yf)
        target = beta[0] + Ap @ beta[1:]; pc = Cache(xp, target, c.xva, c.yva, units); r = target - target.mean(); cands = []
        for kind in (("f", "s") if cfg["rule"] == "best" else (cfg["rule"][0],)):
            for nd in train_new(self.rng, pc, r, units, cfg, kind):
                o = forward(nd, node_input(nd, pc.xtr, pc.tr)); cands.append((float(np.corrcoef(o, target)[0, 1] ** 2) if o.std() > 0 else 0.0, nd))
        if not cands: return
        fid, nd = max(cands, key=lambda z: z[0])
        if fid < cfg["rule_fid"]: return
        units.append(nd); self.deg.append(0); j = len(units) - 1; self.rules = getattr(self, "rules", {}); self.rules[j] = (t, fid)
        for cx in self.caches:
            if cx.tr.shape[1] < len(units): cx.add(nd)

    def train(self, E, c):
        """Linked nodes train together (parents' outputs from the cache held as inputs), best decision-error step kept; caches refreshed after."""
        cfg, units = self.cfg, self.units
        Z = {i: node_input(units[i], c.xtr, c.tr) for i in E}; Zv = {i: node_input(units[i], c.xva, c.va) for i in E}
        P = {i: pvec(units[i]) for i in E}; nd = lambda i: rebuild(units[i], P[i])
        H = np.column_stack([forward(nd(i), Z[i]) for i in E]); F = np.hstack([np.ones((len(c.ytr), 1)), H])
        beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(len(E) + 1), F.T @ c.ytr)
        mo = {i: np.zeros_like(p) for i, p in P.items()}; ve = {i: np.zeros_like(p) for i, p in P.items()}; mb = np.zeros_like(beta); vb = np.zeros_like(beta)
        val = lambda: float(np.mean((beta[0] + np.column_stack([forward(nd(i), Zv[i]) for i in E]) @ beta[1:] - c.yva) ** 2))
        best, bestP, since = val(), {i: p.copy() for i, p in P.items()}, 0
        for s in range(1, cfg["ft_steps"] + 1):
            H = np.column_stack([forward(nd(i), Z[i]) for i in E]); e = (beta[0] + H @ beta[1:] - c.ytr) * (2.0 / len(c.ytr))
            gb = np.concatenate([[e.sum()], H.T @ e])
            for k, i in enumerate(E):
                g = grad(nd(i), Z[i], e * beta[1 + k]); mo[i] = B1 * mo[i] + (1 - B1) * g; ve[i] = B2 * ve[i] + (1 - B2) * g * g
                P[i] = P[i] - cfg["ft_lr"] * (mo[i] / (1 - B1 ** s)) / (np.sqrt(ve[i] / (1 - B2 ** s)) + EPS)
            mb = B1 * mb + (1 - B1) * gb; vb = B2 * vb + (1 - B2) * gb * gb
            beta = beta - cfg["ft_lr"] * (mb / (1 - B1 ** s)) / (np.sqrt(vb / (1 - B2 ** s)) + EPS)
            v = val()
            if v < best: best, bestP, since = v, {i: p.copy() for i, p in P.items()}, 0
            else:
                since += 1
                if since >= cfg["ft_patience"]: break
        for i, p in bestP.items(): units[i] = rebuild(units[i], p)
        first = min(E)
        for cx in self.caches: cx.refresh(units, first)

    def score(self, E, d, xtr, ytr):
        x, y = d["te"]
        xf = np.vstack([xtr, d["va"][0]]) if self.cfg["refit_all"] else xtr; yf = np.concatenate([ytr, d["va"][1]]) if self.cfg["refit_all"] else ytr
        Hf, Ht = all_act(self.units, xf)[:, E], all_act(self.units, x)[:, E]
        F = np.hstack([np.ones((len(yf), 1)), Hf]); Ft = np.hstack([np.ones((len(y), 1)), Ht])
        beta = np.linalg.solve(F.T @ F + self.cfg["ridge"] * np.eye(F.shape[1]), F.T @ yf)
        mse = float(np.mean((y - Ft @ beta) ** 2)); return dict(r2=1.0 - mse / float(np.var(y)), bits=0.5 * math.log2(1.0 / max(mse, FLOOR)))

    def recovered(self, xp, V, cols):
        """For each true concept column j in cols: best |corr| between V[:, j] and any of the learner's functions on probe inputs xp."""
        if not self.units: return {j: 0.0 for j in cols}
        H = all_act(self.units, xp); Hz = (H - H.mean(0)) / (H.std(0) + 1e-9)
        return {j: float(np.max(np.abs(Hz.T @ ((V[:, j] - V[:, j].mean()) / (V[:, j].std() + 1e-9))) / len(xp))) for j in cols}
