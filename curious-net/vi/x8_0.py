"""X8.0: proof loop. Do the closed forms of the combined model (paper/MODEL.md T9-T16) agree with simulation of the model they were derived for?
Registered in paper/PREDICTIONS.md (P332-P339) before this was run. Usage: python x8_0.py selftest | run | log   (writes ../results/x8_0_run.json)
The simulations use the model's own assumptions: agreement proves the algebra, NOT that real models obey assumptions A1-A7 (MODEL.md s5)."""
import sys, os, json, math
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import beta as BETA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_3 import util_loss

OUT = X.OUT
LAM = C("x73_wrong_cost")
OFF = dict(zip(("t9", "t11", "n", "t13", "t14", "self", "t12", "stab", "t16"), C("x80_off")))   # seed offsets, one per check


def rngs(o):
    return np.random.default_rng(C("x80_seed0") + o)


def se(a):
    return float(a.std(ddof=1) / math.sqrt(len(a)))


def delta_of(cS, cA, q, lam):
    """T11: how much worse than the slow path an automatic path may be (error rate)."""
    return cS / lam - cA / (lam * (1 - q))


def n_min(delta):
    """T11: practice length with no disagreement, the first n with 1/(n+2) < delta."""
    return int(math.floor(1 / delta - 2)) + 1


# ---------------------------------------------------------------------------------------------------- T9
def t9():
    a, b = C("x80_beta"); N = C("x80_mc_n"); r = rngs(OFF["t9"]); p = r.beta(a, b, N); right = r.random(N) < p
    worst_beaten, worst_curve, bite = -1e9, 0.0, None
    for lam in C("x80_lam"):
        for ab in C("x80_abst"):
            tau = (lam - ab) / (1 + lam)

            def u(t):
                return np.where(p >= t, np.where(right, 1.0, -lam), -ab)

            def closed(t):
                ans = (1 + lam) * a / (a + b) * (1 - BETA.cdf(t, a + 1, b)) - lam * (1 - BETA.cdf(t, a, b))
                return ans - ab * BETA.cdf(t, a, b)
            ut = u(tau)
            for t in C("x80_tgrid"):
                ug = u(t); d = ut - ug; s = se(d)
                beaten = (-d.mean() / s) if s > 0 else 0.0          # how many standard errors t beats tau by
                worst_beaten = max(worst_beaten, beaten)
                worst_curve = max(worst_curve, abs(ug.mean() - closed(t)) / se(ug))
                if lam == LAM and ab == 0 and t == 0.5:
                    bite = float(d.mean() / s)                       # tau beats the naive 0.5 by this many standard errors
    ok = worst_beaten <= C("x80_se_k") and worst_curve <= C("x80_se_k")
    return dict(ok=ok, worst_beaten_se=worst_beaten, worst_curve_se=worst_curve, bite_se=bite, bite_ok=bite is not None and bite >= C("x80_bite_se"))


# ---------------------------------------------------------------------------------------------------- T11
def gain_sim(r, N, cS, cA, q, lam, eS, eA):
    """Simulated cost saved per case by the automatic path (audited cases use the slow answer)."""
    wS = r.random(N) < eS; wA = r.random(N) < eA; aud = r.random(N) < q
    return (cS + lam * wS) - (cA + aud * cS + lam * np.where(aud, wS, wA))


def t11():
    cS, eS, N = C("x80_cS"), C("x80_eS"), C("x80_mc_n"); r = rngs(OFF["t11"]); worst, bad_sign, n_pts = 0.0, 0, 0
    for q in C("x80_qgrid"):
        for cr in C("x80_cAgrid"):
            for g in C("x80_ggrid"):
                cA = cr * cS; gs = gain_sim(r, N, cS, cA, q, LAM, eS, eS + g); m, s = gs.mean(), se(gs)
                closed = (1 - q) * cS - cA - LAM * (1 - q) * g
                worst = max(worst, abs(m - closed) / s); n_pts += 1
                if abs(m) / s > C("x80_se_k") and (m > 0) != (g < delta_of(cS, cA, q, LAM)):
                    bad_sign += 1
    # practice length of the Bayes controller
    r = rngs(OFF["n"]); exact = []
    for dl in C("x80_deltas"):
        n = 1
        while not (1 / (n + 2) < dl):
            n += 1
        exact.append((dl, n, n_min(dl)))
    dref = delta_of(cS, C("x80_cA"), C("x80_q"), LAM); nm = n_min(dref); nmax = C("x80_nmax"); brk = []
    for d in C("x80_dis"):
        k = np.cumsum(r.random((C("x80_seqs"), nmax)) < d, axis=1); n = np.arange(1, nmax + 1)
        hit = (k + 1) / (n + 2) < dref; first = np.where(hit.any(1), hit.argmax(1) + 1, nmax)
        med = float(np.median(first)); det = (1 - 2 * dref) / (dref - d)
        brk.append(dict(d=d, median_n=med, n_min=nm, n_det=det, ok=nm <= med <= C("x80_nbracket") * det))
    exact_ok = all(a == b for _, a, b in exact)
    ok = worst <= C("x80_se_k") and bad_sign == 0 and exact_ok and all(b["ok"] for b in brk)
    return dict(ok=ok, worst_gain_se=worst, sign_violations=bad_sign, points=n_pts, exact_n=exact, exact_ok=exact_ok, bracket=brk, delta_ref=dref)


# ---------------------------------------------------------------------------------------------------- T12
def cycles(r, rho, m, kappa, q, cR, ncyc):
    """Exact sampler of the shift cycle: a healthy stretch, then cases until the m-th audited disagreement (a negative binomial); returns (cost, cases)."""
    cS, dE = C("x80_cS"), C("x80_shift_err"); dD = dE / kappa; pd = q * dD
    healthy = r.geometric(rho, ncyc); n_o = r.negative_binomial(m, pd, ncyc)
    pa = q * (1 - dD) / (1 - pd); pe = (1 - q) * dE / (1 - pd)
    aud_o = r.binomial(n_o, pa); err = r.binomial(n_o - aud_o, pe / (1 - pa)); aud_h = r.binomial(healthy, q)
    return float(((aud_h + aud_o + m) * cS + LAM * err + cR).sum()), float((healthy + n_o + m).sum())


def phi_sim(seed, rho, m, kappa, q, cR):
    r = rngs(seed); vals = []
    for _ in range(C("x80_stream_reps")):
        c, n = cycles(r, rho, m, kappa, q, cR, C("x80_cycles")); vals.append(c / n)
    v = np.array(vals); return float(v.mean()), se(v)


def phi_closed(rho, m, kappa, q, cR):
    return q * C("x80_cS") + rho * (LAM * m * kappa * (1 - q) / q + cR)


def t12():
    cS, cA, cR = C("x80_cS"), C("x80_cA"), C("x80_cR"); worst_a, worst_flat, min_bite = 0.0, 0.0, 1e9; rows = []; seed = OFF["t12"]
    for rho in C("x80_rhos"):
        for m in C("x80_ms"):
            for kappa in C("x80_kappas"):
                seed += 1; qs = math.sqrt(rho * LAM * m * kappa / cS)
                at = {f: phi_sim(seed, rho, m, kappa, min(1.0, f * qs), cR)[0] for f in (0.5, 1.0, 2.0)}
                for f, v in at.items():
                    worst_a = max(worst_a, abs(v - phi_closed(rho, m, kappa, min(1.0, f * qs), cR)) / phi_closed(rho, m, kappa, min(1.0, f * qs), cR))
                min_bite = min(min_bite, at[0.5] / at[1.0] - 1, at[2.0] / at[1.0] - 1)
                grid = [qs * C("x80_qratio") ** k for k in range(-C("x80_qsteps"), C("x80_qsteps") + 1)]
                best = min(phi_sim(seed, rho, m, kappa, g, cR)[0] for g in grid if g <= 1)
                flat = at[1.0] / best - 1; worst_flat = max(worst_flat, flat)
                rows.append(dict(rho=rho, m=m, kappa=kappa, q_star=qs, phi_closed=phi_closed(rho, m, kappa, qs, cR), phi_sim=at[1.0], flat_excess=flat))
    # stability limit (c_R = 0, no error gap)
    stab = []; ok_stab = True
    for mult in C("x80_stab"):
        rho = mult * (math.sqrt(cS) - math.sqrt(cA)) ** 2 / LAM; qs = math.sqrt(rho * LAM / cS); v, s = phi_sim(OFF["stab"], rho, 1, 1.0, qs, 0.0)
        total = cA + v; dD = C("x80_shift_err")
        good = (total + C("x80_se_k") * s < cS) if mult < 1 else (total - C("x80_se_k") * s > cS)
        ok_stab &= good; stab.append(dict(mult=mult, rho=rho, q_star=qs, auto_total=total, slow=cS, regime_ratio=rho / (qs * dD), ok=bool(good)))
    ok = worst_a <= C("x80_phi_tol") and worst_flat <= C("x80_flat_tol") and ok_stab
    return dict(ok=ok, worst_closed_error=worst_a, worst_flat_excess=worst_flat, min_bite_share=min_bite, bite_ok=min_bite >= C("x80_bite_share"), stability=stab, rows=rows)


# ---------------------------------------------------------------------------------------------------- T13
def t13():
    cS, cA, q, eS, er, ew, N = C("x80_cS"), C("x80_cA"), C("x80_q"), C("x80_eS"), C("x80_er"), C("x80_ew"), C("x80_mc_n"); r = rngs(OFF["t13"])
    dl = delta_of(cS, cA, q, LAM); astar = 1 - (dl + eS - er) / (ew - er)

    def gain(a):
        routed = r.random(N) < a; eA = np.where(routed, er, ew); wS = r.random(N) < eS; wA = r.random(N) < eA; aud = r.random(N) < q
        g = (cS + LAM * wS) - (cA + aud * cS + LAM * np.where(aud, wS, wA)); return g.mean(), se(g)
    xs = [astar + o for o in C("x80_agrid")]; ys = [gain(a)[0] for a in xs]; slope, icpt = np.polyfit(xs, ys, 1); cross = -icpt / slope
    cslope = LAM * (1 - q) * (ew - er); up, dn = gain(astar + C("x80_a_bite")), gain(astar - C("x80_a_bite"))
    ok = abs(cross - astar) <= C("x80_a_tol") and abs(slope / cslope - 1) <= C("x80_slope_tol")
    return dict(ok=ok, a_star=astar, crossing=float(cross), slope=float(slope), slope_closed=cslope, bite_ok=bool(up[0] > C("x80_bite_se") * up[1] and dn[0] < -C("x80_bite_se") * dn[1]),
                gain_plus=up[0], gain_minus=dn[0])


# ---------------------------------------------------------------------------------------------------- T14
def t14():
    cS, cA, q, eS, cT, N = C("x80_cS"), C("x80_cA"), C("x80_q"), C("x80_eS"), C("x80_cT"), C("x80_mc_n"); r = rngs(OFF["t14"]); ms = []
    for f in C("x80_fs"):
        auto = r.random(N) < f; aud = r.random(N) < q; comp = np.where(auto, cA + aud * cS, cS)
        closed = cS / (f * (cA + q * cS) + (1 - f) * cS); ms.append(dict(f=f, M_sim=float(cS / comp.mean()), M_closed=closed, err=abs(cS / comp.mean() / closed - 1)))
    Z, R = C("x80_classes"), C("x80_Rmax"); dl = delta_of(cS, cA, q, LAM); n0 = n_min(dl); G = (1 - q) * cS - cA
    wS = r.random((Z, R)) < eS; wA = r.random((Z, R)) < eS; aud = r.random((Z, R)) < q
    s = (cS + LAM * wS) - (cA + aud * cS + LAM * np.where(aud, wS, wA)); s[:, :n0] = 0.0; s[:, n0] -= cT; cum = s.cumsum(1).mean(0)
    pos = np.nonzero(cum[n0:] >= 0)[0] + n0; i = int(pos[0]); rbe = i + 1 - cum[i] / (cum[i] - cum[i - 1]) if i > 0 else float(i + 1)
    closed = n0 + cT / G
    ok = all(m["err"] <= C("x80_M_tol") for m in ms) and abs(rbe / closed - 1) <= C("x80_Rbe_tol")
    return dict(ok=ok, capacity=ms, n_star=n0, G=G, Rbe_sim=float(rbe), Rbe_closed=closed)


# ---------------------------------------------------------------------------------------------------- T15
def t15():
    torch.manual_seed(C("x80_seed0")); N, R = C("x80_free_cases"), C("x71_regimes"); res = {}
    y0 = torch.randint(0, R, (N,))
    for name, refresh in (("refresh", True), ("fixed", False)):
        o = torch.nn.Parameter(torch.zeros(N, R + 1)); opt = torch.optim.Adam([o], lr=C("x71_lr"))
        for _ in range(C("x80_free_iters")):
            y = torch.randint(0, R, (N,)) if refresh else y0
            loss = util_loss(o, y); opt.zero_grad(); loss.backward(); opt.step()
        res[name] = float((o.argmax(1) == R).float().mean())
    return dict(ok=res["refresh"] >= C("x80_refresh_hi") and res["fixed"] <= C("x80_fixed_lo"), abstain_refresh=res["refresh"], abstain_fixed=res["fixed"])


# ---------------------------------------------------------------------------------------------------- T16
def t16():
    torch.manual_seed(C("x80_seed0")); R, d = C("x71_regimes"), C("x80_dim"); worst = 0.0
    for _ in range(C("x80_ident_n")):
        W = torch.randn(R, d, dtype=torch.float64, requires_grad=True); x = torch.randn(d, dtype=torch.float64); t = int(torch.randint(0, R, (1,)))
        F.cross_entropy((W @ x)[None], torch.tensor([t])).backward(); p = F.softmax(W @ x, 0).detach(); tv = F.one_hot(torch.tensor(t), R).double()
        worst = max(worst, float((W.grad - torch.outer(p - tv, x)).abs().max()))
    diffs = []
    for k in range(C("x80_tasks")):
        r = rngs(OFF["t16"] + k); cen = r.normal(size=(R, d)) * C("x80_sep")
        def draw(n):
            y = r.integers(0, R, n); return torch.tensor(cen[y] + r.normal(size=(n, d)), dtype=torch.float32), y
        x, y = draw(C("x80_stream_n")); xt, yt = draw(C("x80_stream_n") // 4)
        flip = r.random(len(y)) < C("x80_teacher_err"); teach = np.where(flip, (y + r.integers(1, R, len(y))) % R, y)
        aud = np.nonzero(r.random(len(y)) < C("x80_q"))[0]
        Wa = torch.zeros(R, d, requires_grad=True); Wl = torch.zeros(R, d)
        for i in aud:
            la = F.cross_entropy((Wa @ x[i])[None], torch.tensor([teach[i]])); g, = torch.autograd.grad(la, Wa)
            with torch.no_grad():
                Wa -= C("x80_lr_local") * g
                pl = F.softmax(Wl @ x[i], 0)
                if int(pl.argmax()) != teach[i]:                               # the third factor: an audited disagreement
                    Wl -= C("x80_lr_local") * torch.outer(pl - F.one_hot(torch.tensor(int(teach[i])), R).float(), x[i])
        acc = lambda W: float(((xt @ W.detach().T).argmax(1).numpy() == yt).mean())
        diffs.append(acc(Wa) - acc(Wl))
    med = float(np.median(diffs))
    return dict(ok=worst <= C("x80_ident_tol") and abs(med) <= C("x80_acc_tol"), max_identity_error=worst, median_acc_gap_autograd_minus_local=med, gaps=diffs)


# ---------------------------------------------------------------------------------------------------- driver
def phi_ren(rho, m, kappa, q, cR):
    """T12 refined: time spent shifted lowers the effective shift rate by 1/(1+rho D), D = m/(q Delta_d) (renewal-reward)."""
    D = m / (q * C("x80_shift_err") / kappa)
    return q * C("x80_cS") + rho * (LAM * m * kappa * (1 - q) / q + cR) / (1 + rho * D)


def t12b():
    """P340 on FRESH seeds: the original closed form versus the renewal-corrected one in every (rho, m, kappa, 0.5q*/q*/2q*) cell."""
    cS, cR = C("x80_cS"), C("x80_cR"); seed = OFF["t12"] + C("x80_fresh_off"); worst_o, worst_r, rows = 0.0, 0.0, []
    for rho in C("x80_rhos"):
        for m in C("x80_ms"):
            for kappa in C("x80_kappas"):
                seed += 1; qs = math.sqrt(rho * LAM * m * kappa / cS)
                for f in (0.5, 1.0, 2.0):
                    q = min(1.0, f * qs); v = phi_sim(seed, rho, m, kappa, q, cR)[0]; eo = abs(v / phi_closed(rho, m, kappa, q, cR) - 1); er = abs(v / phi_ren(rho, m, kappa, q, cR) - 1)
                    worst_o, worst_r = max(worst_o, eo), max(worst_r, er); rows.append(dict(rho=rho, m=m, kappa=kappa, f=f, err_original=eo, err_renewal=er))
    return dict(ok=worst_r <= C("x80_ren_tol"), worst_original=worst_o, worst_renewal=worst_r, rows=rows)


def rerun():
    res = {"T14": t14(), "T12b": t12b()}
    for k, v in res.items():
        print(k, "ok" if v["ok"] else "FAIL", {a: b for a, b in v.items() if isinstance(b, (int, float, bool))}, flush=True)
    with open(os.path.join(OUT, "x8_0_rerun.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


def selftest():
    """Sampler versus a per-case loop on a small setting, and the closed forms' arithmetic."""
    for dl in C("x80_deltas"):
        n = 1
        while not (1 / (n + 2) < dl):
            n += 1
        assert n == n_min(dl), ("n_min", dl)
    dl = delta_of(C("x80_cS"), C("x80_cA"), C("x80_q"), LAM); assert abs(dl - (C("x80_cS") / LAM - C("x80_cA") / (LAM * (1 - C("x80_q"))))) < C("x80_ident_tol"), dl
    rho, m, kappa, q, cR = C("x80_rhos")[-1], 1, 1.0, 0.3, C("x80_cR"); r = rngs(OFF["self"]); cS, dE = C("x80_cS"), C("x80_shift_err"); tot, cases = 0.0, 0
    for _ in range(C("x80_cycles")):
        healthy = True; det = 0
        while det < m:
            cases += 1
            if healthy:
                if r.random() < rho: healthy = False
                tot += cS * (r.random() < q)
            else:
                if r.random() < q:
                    tot += cS
                    if r.random() < dE / kappa: det += 1
                elif r.random() < dE: tot += LAM
        tot += cR
    v = phi_sim(OFF["self"], rho, m, kappa, q, cR)[0]; loop = tot / cases
    print(f"sampler {v:.5f}  per-case loop {loop:.5f}  closed {phi_closed(rho, m, kappa, q, cR):.5f}")
    assert abs(v / loop - 1) < C("x80_phi_tol"), "sampler disagrees with the per-case loop"
    print("selftest ok")


def run():
    res = {}
    for k, f in (("T9", t9), ("T11", t11), ("T12", t12), ("T13", t13), ("T14", t14), ("T15", t15), ("T16", t16)):
        res[k] = f(); print(k, "ok" if res[k]["ok"] else "FAIL", {a: b for a, b in res[k].items() if isinstance(b, (int, float, bool))}, flush=True)
    res["P332"] = dict(ok=bool(res["T9"]["bite_ok"] and res["T12"]["bite_ok"] and res["T13"]["bite_ok"]))
    verdicts = {"P332": res["P332"]["ok"], "P333": res["T9"]["ok"], "P334": res["T11"]["ok"], "P335": res["T12"]["ok"], "P336": res["T13"]["ok"], "P337": res["T14"]["ok"], "P338": res["T15"]["ok"], "P339": res["T16"]["ok"]}
    res["verdicts"] = verdicts; os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "x8_0_run.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(verdicts)


def log():
    with open(os.path.join(OUT, "x8_0_run.json")) as fh:
        print(json.dumps(json.load(fh)["verdicts"], indent=1))


if __name__ == "__main__":
    {"selftest": selftest, "run": run, "rerun": rerun, "log": log}[sys.argv[1]]()
