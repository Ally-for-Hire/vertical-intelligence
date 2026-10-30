"""X7.1 Mirror: a minimal higher layer that takes ANY nn.Module, learns a generative model of its internals, and dreams them with the environment cut off.
Registered in paper/PREDICTIONS.md (P294-P299) before this was run. Mirror never names a layer type: it hooks every leaf module, randomly projects each
output to a fixed width, standardises, and fits a mixture-density next-step model (GRU or memoryless). Dreaming = feed the model its own samples.
Usage: python x7_1.py selftest | run | log        (writes ../results/x7_1_<mode>.json)"""
import sys, os, json, time, inspect
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X

OUT = X.OUT
FAMS = ("MLP", "CONV1D", "GRU", "TRANSFORMER", "RANDOM")
ARMS = ("MIRROR_GRU", "MIRROR_AR1", "GAUSS_IID", "SHUFFLE_IID", "HOSTDREAM")


# ---------------------------------------------------------------- Mirror: the whole layer (host-agnostic) -------------------------------
class Mirror:
    """Attach to any nn.Module: forward hooks on every leaf module; read(x) returns the host's whole internal state as one vector per sample."""
    def __init__(self, host, seed):
        self.host, self.g, self.P, self.cur, self.order = host, torch.Generator().manual_seed(seed), {}, {}, []
        host.train()                                                       # train mode keeps fused fast paths (which skip hooks) off; no dropout/BN is used
        for i, m in enumerate([m for m in host.modules() if not list(m.children())]): m.register_forward_hook(self._tap(i))

    def _tap(self, i):
        def f(m, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o): self.cur[i] = o.flatten(1)
        return f

    @torch.no_grad()
    def read(self, x):
        self.cur = {}; self.host(x)
        keys = list(self.cur)
        if not self.order: self.order = keys
        out = []
        for k in self.order:
            o = self.cur[k]
            if k not in self.P: self.P[k] = torch.randn(o.shape[1], C("x71_proj"), generator=self.g) / o.shape[1] ** 0.5
            out.append(o @ self.P[k])
        self.last = slice(C("x71_proj") * (len(self.order) - 1), C("x71_proj") * len(self.order))      # the layer that fired last
        return torch.cat(out, 1)


class Gen(nn.Module):
    """Mixture-density next-step model of a vector stream; recurrent (GRU) or memoryless (MLP). The model is the same for every host."""
    def __init__(s, D, recurrent):
        super().__init__(); H, s.K, s.D, s.rec = C("x71_hidden"), C("x71_mix"), D, recurrent
        s.core = nn.GRU(D, H, batch_first=True) if recurrent else nn.Sequential(nn.Linear(D, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh())
        s.head = nn.Linear(H, s.K * (2 * D + 1))

    def forward(s, x, h=None):
        z, h = s.core(x, h) if s.rec else (s.core(x), None)
        o = s.head(z); K, D = s.K, s.D
        return o[..., :K], o[..., K:K + K * D].reshape(*o.shape[:-1], K, D), o[..., K + K * D:].reshape(*o.shape[:-1], K, D).clamp(C("x71_logsd_min"), C("x71_logsd_max")), h

    def nll(s, x, y):
        pi, mu, ls, _ = s(x)
        lp = (-0.5 * ((y[..., None, :] - mu) / ls.exp()) ** 2 - ls).sum(-1) + F.log_softmax(pi, -1)
        return -torch.logsumexp(lp, -1)                                   # constant 0.5*D*log(2*pi) dropped: same for every arm

    @torch.no_grad()
    def dream(s, warm, L, rng):
        """Prime with real steps, then free-run on own samples. warm (B,W,D) -> (B,L,D). No environment input after the cut-off."""
        pi, mu, ls, h = s(warm); out = []
        for _ in range(L):
            p = F.softmax(pi[:, -1], -1); k = torch.multinomial(p, 1, generator=rng).squeeze(1); ar = torch.arange(len(k))
            x = mu[:, -1][ar, k] + ls[:, -1][ar, k].exp() * torch.randn(len(k), s.D, generator=rng)
            out.append(x); pi, mu, ls, h = s(x[:, None], h)
        return torch.stack(out, 1)


def fit_gen(data, recurrent, seed):
    torch.manual_seed(seed); g = Gen(data.shape[1], recurrent); opt = torch.optim.Adam(g.parameters(), lr=C("x71_lr")); W = C("x71_win"); rng = np.random.default_rng(seed)
    for _ in range(C("x71_train_iters")):
        st = rng.integers(0, len(data) - W - 1, C("x71_batch")); w = torch.stack([data[a:a + W + 1] for a in st])
        loss = g.nll(w[:, :-1], w[:, 1:]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return g


# ---------------------------------------------------------------- world and hosts ------------------------------------------------------
class Env:
    def __init__(s, seed, sd_means=1.0):
        s.rng = np.random.default_rng(seed); s.means = s.rng.normal(0, sd_means, (C("x71_regimes"), C("x71_in_dim")))

    def stream(s, rng, T):
        R, r, out = C("x71_regimes"), int(rng.integers(C("x71_regimes"))), []
        for _ in range(T):
            if rng.random() < 1 / C("x71_dwell"): r = int((r + 1 + rng.integers(R - 1)) % R)
            out.append(r)
        r = np.array(out); return torch.tensor(s.means[r] + C("x71_regime_sd") * rng.normal(size=(T, C("x71_in_dim"))), dtype=torch.float32), torch.tensor(r)


class Conv(nn.Module):
    def __init__(s): super().__init__(); s.c = nn.Conv1d(1, 8, 3, padding=1); s.a = nn.ReLU(); s.f = nn.Linear(8 * C("x71_in_dim"), C("x71_regimes"))
    def forward(s, x): return s.f(s.a(s.c(x[:, None, :])).flatten(1))


class GRUHost(nn.Module):
    def __init__(s): super().__init__(); s.g = nn.GRU(1, C("x71_host_w"), batch_first=True); s.f = nn.Linear(C("x71_host_w"), C("x71_regimes"))
    def forward(s, x): return s.f(s.g(x[:, :, None])[0][:, -1])


class TfHost(nn.Module):
    def __init__(s):
        super().__init__(); w = C("x71_host_w"); s.e = nn.Linear(1, w); s.enc = nn.TransformerEncoderLayer(w, 2, 2 * w, dropout=0.0, batch_first=True); s.f = nn.Linear(w, C("x71_regimes"))
    def forward(s, x): return s.f(s.enc(s.e(x[:, :, None])).mean(1))


def make_host(fam, seed, env):
    torch.manual_seed(seed); I, R, H = C("x71_in_dim"), C("x71_regimes"), C("x71_hidden")
    h = {"MLP": lambda: nn.Sequential(nn.Linear(I, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, R)), "CONV1D": Conv, "GRU": GRUHost, "TRANSFORMER": TfHost}
    host = h["MLP" if fam == "RANDOM" else fam]()
    if fam != "RANDOM":
        opt = torch.optim.Adam(host.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed + 1)
        for _ in range(C("x71_host_steps")):
            r = torch.tensor(rng.integers(0, R, C("x71_host_batch"))); x = torch.tensor(env.means[r.numpy()] + C("x71_regime_sd") * rng.normal(size=(len(r), I)), dtype=torch.float32)
            loss = F.cross_entropy(host(x), r); opt.zero_grad(); loss.backward(); opt.step()
    return host


# ---------------------------------------------------------------- metrics ---------------------------------------------------------------
def windows(seq, n, rng, lo=0, hi=None):
    """seq (T,D) or (B,T,D); n windows of x71_win steps with start in [lo, hi-win]."""
    W = C("x71_win")
    if seq.dim() == 2:
        st = rng.integers(lo, (hi or len(seq)) - W, n); return torch.stack([seq[a:a + W] for a in st])
    b = rng.integers(0, len(seq), n); st = rng.integers(lo, (hi or seq.shape[1]) - W, n); return torch.stack([seq[i, a:a + W] for i, a in zip(b, st)])


def mmd2(A, B, bw):
    """Unbiased MMD^2 with an RBF kernel on flattened windows."""
    A, B = A.flatten(1), B.flatten(1); k = lambda U, V: torch.exp(-torch.cdist(U, V) ** 2 / (2 * bw ** 2)); n, m = len(A), len(B)
    kaa, kbb = k(A, A), k(B, B)
    return float((kaa.sum() - kaa.trace()) / (n * (n - 1)) + (kbb.sum() - kbb.trace()) / (m * (m - 1)) - 2 * k(A, B).mean())


def one(seed, fam):
    torch.set_num_threads(1); rs = lambda o: np.random.default_rng(seed * 10 + o); tg = torch.Generator().manual_seed(seed)
    env = Env(seed); host = make_host(fam, seed, env); mir = Mirror(host, seed)
    x_obs, _ = env.stream(rs(1), C("x71_obs_steps")); s_obs = mir.read(x_obs); mu, sd = s_obs.mean(0), s_obs.std(0) + 1e-6; z = lambda s: (s - mu) / sd
    obs = z(s_obs); D = obs.shape[1]
    refs = [env.stream(rs(2 + i), C("x71_ref_len"))[0] for i in range(C("x71_streams"))]                                    # A reference, B floor, C priming
    sA, sB, sC = [z(mir.read(r)) for r in refs]; nw, W = C("x71_nwin"), C("x71_win")
    wA, wB = windows(sA, nw, rs(5)), windows(sB, nw, rs(6)); bw = float(torch.cdist(wA.flatten(1), wA.flatten(1)).median())
    out = {"fam": fam, "D": D, "floor": mmd2(wB, wA, bw)}
    g_gru, g_ar1 = fit_gen(obs, True, seed), fit_gen(obs, False, seed)
    warm = windows(sC, nw, rs(7))[:, :C("x71_warm")]
    L, E_, L0 = C("x71_dream_len"), C("x71_early_end"), C("x71_late_start")
    dreams = {"MIRROR_GRU": g_gru.dream(warm, L, tg), "MIRROR_AR1": g_ar1.dream(warm, L, tg)}
    cov = torch.cov(obs.T) + 1e-4 * torch.eye(D); gs = torch.distributions.MultivariateNormal(obs.mean(0), cov)
    dreams["GAUSS_IID"] = gs.sample((nw, L)); dreams["SHUFFLE_IID"] = obs[torch.tensor(rs(8).integers(0, len(obs), (nw, L)))]
    xm, xs = x_obs.mean(0), x_obs.std(0) + 1e-6; g_env = fit_gen((x_obs - xm) / xs, True, seed)                                  # REFERENCE: models the environment, uses the real host
    wenv = ((refs[2] - xm) / xs)[:, :]; wi = windows(wenv, nw, rs(7))[:, :C("x71_warm")]
    xd = (g_env.dream(wi, L, tg) * xs + xm).reshape(-1, C("x71_in_dim"))                                                      # dreamed ENVIRONMENT, then the real host reads it
    dreams["HOSTDREAM"] = z(torch.cat([mir.read(c) for c in xd.split(C("x71_ref_len"))])).reshape(nw, L, D)
    out["mmd_late"], out["mmd_early"] = {}, {}
    for a in ARMS:
        out["mmd_late"][a] = mmd2(windows(dreams[a], nw, rs(9), L0, L), wA, bw); out["mmd_early"][a] = mmd2(windows(dreams[a], nw, rs(10), 0, E_), wA, bw)
    sh = obs[torch.tensor(rs(12).permutation(len(obs)))]; out["mmd_shuffle_real"] = mmd2(windows(sh, nw, rs(11)), wA, bw)
    # consistency: nonlinear probe (real data only) predicting the last-firing layer from all other layers
    ls = mir.last; keep = torch.ones(D, dtype=torch.bool); keep[ls] = False
    torch.manual_seed(seed); pr = nn.Sequential(nn.Linear(int(keep.sum()), C("x71_hidden")), nn.Tanh(), nn.Linear(C("x71_hidden"), C("x71_proj"))); opt = torch.optim.Adam(pr.parameters(), lr=C("x71_lr"))
    for _ in range(C("x71_probe_steps")):
        i = torch.tensor(rs(13).integers(0, len(obs), C("x71_host_batch"))); l = F.mse_loss(pr(obs[i][:, keep]), obs[i][:, ls]); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        perr = lambda s: float(F.mse_loss(pr(s[..., keep]), s[..., ls]))
        out["probe"] = {"REAL": perr(sA), **{a: perr(dreams[a][:, L0:]) for a in ARMS}}
    # limits: surprise on a shifted environment
    env2 = Env(seed + C("x71_shift_seed") * 1000, C("x71_shift_sd")); sS = z(mir.read(env2.stream(rs(14), C("x71_ref_len"))[0]))
    with torch.no_grad():
        wu, wsf = windows(sB, nw, rs(15)), windows(sS, nw, rs(15)); n_u, n_s = g_gru.nll(wu[:, :-1], wu[:, 1:]).mean(1), g_gru.nll(wsf[:, :-1], wsf[:, 1:]).mean(1)
    out["surprise_z"] = float((n_s.mean() - n_u.mean()) / (n_u.std() + 1e-9)); out["nll_unshifted"], out["nll_shifted"] = float(n_u.mean()), float(n_s.mean())
    out["dream_std_ratio"] = {a: float(dreams[a][:, L0:].std() / sA.std()) for a in ARMS}
    return out


# ---------------------------------------------------------------- summary and verdicts ---------------------------------------------------
def summarize(per):
    fams = sorted({r["fam"] for r in per.values()}, key=FAMS.index); seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float)
    famof = np.array([per[s]["fam"] for s in seeds]); ci = lambda d: X.boot_ci(d, C("x71_seed0"))
    late = lambda a: g(lambda r: r["mmd_late"][a]); S = {"fams": fams, "median_mmd_late": {a: float(np.median(late(a))) for a in ARMS}, "median_floor": float(np.median(g(lambda r: r["floor"]))), "paired": {}, "by_family": {}}
    for b in ("GAUSS_IID", "SHUFFLE_IID"):
        S["paired"][f"late MMD MIRROR_GRU - {b}"] = ci(late("MIRROR_GRU") - late(b))
        for f in fams: S["by_family"][f"{f}: MIRROR_GRU - {b}"] = X.boot_ci((late("MIRROR_GRU") - late(b))[famof == f], C("x71_seed0"))
    S["manip_ratio_ok"] = float(np.mean(g(lambda r: r["mmd_shuffle_real"]) >= C("x71_manip_ratio") * g(lambda r: max(r["floor"], 1e-6))))
    S["late_over_early"] = float(np.median(g(lambda r: r["mmd_late"]["MIRROR_GRU"]) / np.maximum(g(lambda r: r["mmd_early"]["MIRROR_GRU"]), 1e-9)))
    pe = lambda a: g(lambda r: r["probe"][a]); S["paired"]["probe error MIRROR_GRU - GAUSS_IID"] = ci(pe("MIRROR_GRU") - pe("GAUSS_IID"))
    S["probe_over_floor"] = float(np.median(pe("MIRROR_GRU") / np.maximum(pe("REAL"), 1e-9))); S["median_probe"] = {a: float(np.median(pe(a))) for a in ("REAL",) + ARMS}
    S["ar1_over_gru"] = float(np.median(late("MIRROR_AR1") / np.maximum(late("MIRROR_GRU"), 1e-9)))
    S["surprise_ok"] = float(np.mean(g(lambda r: r["surprise_z"]) > C("x71_shift_z"))); S["median_surprise_z"] = float(np.median(g(lambda r: r["surprise_z"])))
    S["median_dream_std_ratio"] = {a: float(np.median(g(lambda r: r["dream_std_ratio"][a]))) for a in ARMS}
    return S


def judge(S):
    below = lambda k: S["paired"][k][2] < 0                                                    # upper end of the 95% range below 0
    fam_ok = all(S["by_family"][f"{f}: MIRROR_GRU - {b}"][2] < 0 for f in S["fams"] for b in ("GAUSS_IID", "SHUFFLE_IID"))
    v = []
    v.append(("P294", "Time matters in this world (a model without time cannot match) and the same code ran on all 5 hosts", "PASSED" if S["manip_ratio_ok"] >= 0.9 and len(S["fams"]) == len(FAMS) else "VOID", f"iid-shuffled real windows were at least {C('x71_manip_ratio'):g}x the real-vs-real floor in {S['manip_ratio_ok']:.0%} of worlds; families run: {S['fams']}"))
    p = S["paired"]; v.append(("P295", "Dreams after the cut-off match real internals better than both iid baselines (pooled and in every family)", "PASSED" if below("late MMD MIRROR_GRU - GAUSS_IID") and below("late MMD MIRROR_GRU - SHUFFLE_IID") and fam_ok else "FAILED",
                              f"pooled median MMD^2 differences {[round(x, 4) for x in p['late MMD MIRROR_GRU - GAUSS_IID']]} vs Gaussian, {[round(x, 4) for x in p['late MMD MIRROR_GRU - SHUFFLE_IID']]} vs shuffle; every family below 0: {fam_ok}; median MMD^2 {S['median_mmd_late']}, floor {S['median_floor']:.4f}"))
    v.append(("P296", "No collapse: late MMD / early MMD <= 1.5", "PASSED" if S["late_over_early"] <= C("x71_collapse_ratio") else "FAILED", f"median ratio {S['late_over_early']:.2f}"))
    v.append(("P297", "Dreams are internally consistent (probe error below Gaussian dreams and <= 2x the real floor)", "PASSED" if below("probe error MIRROR_GRU - GAUSS_IID") and S["probe_over_floor"] <= C("x71_floor_ratio") else "FAILED", f"paired difference {[round(x, 4) for x in p['probe error MIRROR_GRU - GAUSS_IID']]}; Mirror/real-floor {S['probe_over_floor']:.2f}; median probe errors {S['median_probe']}"))
    v.append(("P298", "Memory is not needed (AR1 / GRU MMD <= 1.1)", "PASSED" if S["ar1_over_gru"] <= C("x71_memless_ratio") else "FAILED", f"median ratio {S['ar1_over_gru']:.2f}"))
    v.append(("P299", "The Mirror notices when the environment no longer matches its model (>= 90% of worlds, > 2 sd)", "PASSED" if S["surprise_ok"] >= 0.9 else "FAILED", f"{S['surprise_ok']:.0%} of worlds; median z {S['median_surprise_z']:.1f}"))
    return v


ARM_TEXT = {"MIRROR_GRU": ("Mirror with memory", "Watches only the network's internals, then imagines them with the world cut off (GRU model)."),
            "MIRROR_AR1": ("Mirror without memory", "Same, but the model sees only the previous internal state."),
            "GAUSS_IID": ("Blind bell-curve", "Draws internal states independently from a Gaussian fitted to the same observations: right on average, no sense of time."),
            "SHUFFLE_IID": ("Shuffled real states", "Real internal states drawn independently in random order: perfect on average, no sense of time."),
            "HOSTDREAM": ("Reference: model the world, run the real network", "Not a rival: uses the real network, which Mirror may not. Shows what modelling the environment would give.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.1", "Mirror: a minimal layer that models any network from the inside", "Can one unchanged layer, attached by hooks to any neural network, learn the network's whole internal state and keep producing realistic internal states ('visions') after the environment is cut off?",
                 "A noisy world switches between 4 hidden regimes. Five kinds of network (MLP, 1-D conv, GRU, transformer, untrained MLP) look at it. Mirror only sees the networks' internal numbers.", ARM_TEXT,
                 "Fidelity = MMD^2 between imagined and real windows of 32 steps (lower is better; 'floor' = two real streams against each other). Probe error = how well the network's own layer-to-layer relation holds inside the dreams.", dict(worlds=len(per), families=list(FAMS)), tags=["mirror", "any-network"])
    tk.table("Summary (median over worlds)", ["Arm", "Late-dream MMD^2 (lower is better)", "Layer-consistency error (lower is better)", "Dream spread / real spread (1 is ideal)"], [[tk.name(a), round(S["median_mmd_late"][a], 4), round(S["median_probe"][a], 4), round(S["median_dream_std_ratio"][a], 2)] for a in ARMS])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in {**S["paired"], **S["by_family"]}.items()])
    tk.table("Results per world", ["World seed", "Network", "Floor MMD^2"] + [f"{tk.name(a)} MMD^2 (late)" for a in ARMS] + ["Surprise on shifted world (sd)"], [[int(s), r["fam"], round(r["floor"], 4)] + [round(r["mmd_late"][a], 4) for a in ARMS] + [round(r["surprise_z"], 1)] for s, r in per.items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def selftest():
    src = inspect.getsource(Mirror)
    assert not any(n in src for n in ("nn.Linear(", "Conv", "GRUHost", "TfHost")), "Mirror must not name host layer types"
    seeds = [C("x71_seed0") + i for i in range(C("x71_worlds_per_host") * len(FAMS))]; assert len(set(seeds)) == len(seeds)
    r = one(C("x71_seed0"), "TRANSFORMER"); print(json.dumps(r, default=float)[:400]); print("selftest ok")


def main(mode):
    t0 = time.time()
    if mode == "selftest": selftest(); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_1_run.json")))["result"]); return
    from budget import pmap
    jobs = [(C("x71_seed0") + i * len(FAMS) + j, f) for i in range(C("x71_worlds_per_host")) for j, f in enumerate(FAMS)]
    res = pmap(one, jobs); per = {s: r for (s, _), r in zip(jobs, res)}
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_1_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3])
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
