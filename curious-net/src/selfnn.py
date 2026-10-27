"""Self-learning network, v0. Base = per-output online MLPs (plain, Adam, batch 1). The ONLY experience-dependent part is a learned prior over
'does input i affect output j' from scenario-independent pair descriptors measured on the first W ticks. With no experience the prior keeps every
input, so the network is EXACTLY the plain per-output network (parity by construction). Labels for learning the prior come from the agent's own
trained network (permutation importance with a paired z-test at Bonferroni level 1/n_inputs), never from the generator's ground truth.
Baseline settings (hidden width, lr, ridge strength, W) are experiment settings, declared in experiments/expL_selfnn.py."""
import numpy as np
from scipy.stats import norm, rankdata
from nn import AdamMLP, RunStd

FEATS = 8


def _corr(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return (B.T @ A) / len(A)                                           # (nB, nA)


def pair_descriptors(X, D, out_pos, n_act):
    """X: (W, n_in) standardised inputs at t; D: (W, n_out) standardised next-step changes; out_pos: input column of each output's own variable.
    returns (n_out, n_in, FEATS) scenario-independent descriptors (permutation-equivariant: no feature depends on an input's index or name)."""
    W, n_in = X.shape
    n_out = D.shape[1]
    sq = np.sqrt(W)
    f1 = np.abs(_corr(X, D)) * sq                                       # linear association with the output's next change
    Rx = np.apply_along_axis(rankdata, 0, X)
    Rd = np.apply_along_axis(rankdata, 0, D)
    f2 = np.abs(_corr(Rx, Rd)) * sq                                     # monotone association
    dX = np.vstack([np.zeros((1, n_in)), np.diff(X, axis=0)])
    f3 = np.abs(_corr(dX, D)) * sq                                      # association of the input's last change with the output's next change
    f4 = np.zeros((n_out, n_in))                                        # association left after removing the output's own state
    for j in range(n_out):
        xo = X[:, out_pos[j]]
        b = (xo - xo.mean()) @ (D[:, j] - D[:, j].mean()) / ((xo - xo.mean()) @ (xo - xo.mean()) + 1e-9)
        res = D[:, j] - b * xo
        f4[j] = np.abs(_corr(X, res[:, None])[0]) * sq
    q = np.stack([np.searchsorted(np.quantile(X[:, i], [.25, .5, .75]), X[:, i]) for i in range(n_in)], 1)
    sg = (D > 0).astype(int)
    f5 = np.zeros((n_out, n_in))                                        # binned mutual information (4 quantile bins of the input vs sign of the output change)
    for j in range(n_out):
        pg = (sg[:, j].sum() + .5) / (W + 1)
        for b in range(4):
            m = (q == b).astype(float)
            n1 = m.T @ sg[:, j]
            n = m.sum(0)
            p1 = (n1 + .5) / (n + 1)
            f5[j] += n / W * (p1 * np.log(p1 / pg) + (1 - p1) * np.log((1 - p1) / (1 - pg)))
    f5 = f5 * W
    ac = np.array([abs(np.corrcoef(X[1:, i], X[:-1, i])[0, 1]) if X[:, i].std() > 0 else 0.0 for i in range(n_in)])
    out = np.zeros((n_out, n_in, FEATS))
    for k, f in enumerate((f1, f2, f3, f4, f5)):
        out[:, :, k] = np.log1p(np.nan_to_num(f))
    out[:, :, 5] = np.tile(np.nan_to_num(ac), (n_out, 1))
    for j in range(n_out):
        out[j, out_pos[j], 6] = 1.0                                     # this input is the output's own variable
    out[:, n_in - n_act:, 7] = 1.0                                      # this input is an action channel
    return out


class PriorModel:
    """logistic regression on pair descriptors (ridge), fitted on pooled own-experience labels; w=None (no experience) keeps every input"""
    def __init__(self, ridge, rule="hard", tau=0.5):
        self.w = None
        self.ridge = ridge
        self.rule, self.tau = rule, tau                 # decision rule for using the prior: hard threshold, 'zeromiss' (smallest p of any past positive), or 'soft' (scale input by p)
        self.y_min_p = None

    def fit(self, F, y):
        n, d = F.shape
        A = np.hstack([F, np.ones((n, 1))])
        w = np.zeros(d + 1)
        for _ in range(25):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) + self.ridge * np.r_[w[:-1], 0]
            H = (A * (p * (1 - p))[:, None]).T @ A + self.ridge * np.eye(d + 1)
            w -= np.linalg.solve(H, g)
        self.w = w
        pp = self.prob(F)
        self.y_min_p = float(pp[y > 0.5].min()) if (y > 0.5).any() else 0.0

    def prob(self, F):
        if self.w is None:
            return np.ones(F.shape[0])
        return 1 / (1 + np.exp(-(np.hstack([F, np.ones((len(F), 1))]) @ self.w)))


def run_scenario(X, Y, out_pos, n_act, T, hidden, lr, W, seed, mode, prior=None, truth_mask=None, gl=0.0, drop=0.0):
    """one scenario. mode: 'joint' (one MLP, all inputs) | 'plain' (per-output MLPs, all inputs) | 'oracle' (true parents) | 'screen' | 'prior' (learned prior).
    Targets = change of the output variables; everything standardised online; returns prequential squared errors, masks, descriptors, nets."""
    ns = Y.shape[1]
    n_in = X.shape[1]
    D = Y - X[:, :ns]
    sx, sy = RunStd(n_in), RunStd(len(out_pos))
    n_out = len(out_pos)
    masks = np.ones((n_out, n_in))
    if mode == "oracle":
        masks = truth_mask.astype(float)
    if mode == "joint":
        nets = [AdamMLP(n_in, n_out, hidden, lr, seed, None)]
    else:
        nets = [AdamMLP(n_in, 1, hidden, lr, seed + j, masks[j], gl, drop) for j in range(n_out)]
    err = np.zeros(T)
    desc = None
    Xs = np.zeros((T, n_in))
    Ds = np.zeros((T, n_out))
    for t in range(T):
        x = X[t]
        y = D[t][out_pos]
        sx.update(x)
        sy.update(y)
        xs, ys = sx(x), sy(y)
        Xs[t], Ds[t] = xs, ys
        if t == W:
            desc = pair_descriptors(Xs[:W], Ds[:W], out_pos, n_act)
            if mode == "screen":
                zc = norm.isf(1.0 / (n_in * n_out))
                masks = (np.expm1(desc[:, :, 0]) > zc).astype(float)
            elif mode == "prior":
                if getattr(prior, 'windows', False):
                    p = prior.prob_windows(Xs[:W], Ds[:W], out_pos, n_act)
                else:
                    p = prior.prob(desc.reshape(-1, FEATS)).reshape(n_out, n_in)
                if prior.rule == "soft":
                    masks = p
                elif prior.rule == "zeromiss":
                    masks = (p >= prior.y_min_p).astype(float)
                else:
                    masks = (p >= prior.tau).astype(float)
            if mode in ("screen", "prior"):
                for j in range(n_out):
                    if masks[j].sum() == 0:
                        masks[j] = 1.0                      # no basis to exclude anything: keep every input (parity)
                    nets[j].mask = masks[j]
        if mode == "joint":
            p = nets[0].step(xs, ys)
            err[t] = np.mean((p - ys) ** 2)
        else:
            e = 0.0
            for j in range(n_out):
                pj = nets[j].step(xs, ys[j:j + 1])
                e += (pj[0] - ys[j]) ** 2
            err[t] = e / n_out
    return dict(err=err, masks=masks, desc=desc, nets=nets, Xs=Xs, Ds=Ds)


def perm_labels(nets, Xs, Ds, last=300, seed=0):
    """own-experience relevance labels: shuffle one input over the last `last` samples; label 1 if the paired error increase is significant at 1/n_in"""
    rng = np.random.default_rng(seed)
    n_out = len(nets)
    n_in = Xs.shape[1]
    Xb, Db = Xs[-last:], Ds[-last:]
    zc = norm.isf(1.0 / n_in)
    lab = np.zeros((n_out, n_in))
    for j, net in enumerate(nets):
        mk = net.mask if net.mask is not None else 1.0
        base = ((np.tanh((Xb * mk) @ net.W1 + net.b1) @ net.W2 + net.b2)[:, 0] - Db[:, j]) ** 2
        for i in range(n_in):
            Xp = Xb.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            e = ((np.tanh((Xp * mk) @ net.W1 + net.b1) @ net.W2 + net.b2)[:, 0] - Db[:, j]) ** 2
            d = e - base
            lab[j, i] = float(d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12) > zc)
    return lab


class NNPrior:
    """small neural prior over pair descriptors (torch, CPU): descriptors -> P(input affects output). Used with soft gating (input scaled by p).
    Same interface as PriorModel. Training settings are experiment settings (expM)."""
    def __init__(self, hidden=32, steps=400, lr=1e-2, seed=0):
        self.hidden, self.steps, self.lr, self.seed = hidden, steps, lr, seed
        self.net = None
        self.rule, self.tau, self.y_min_p, self.w = "soft", 0.5, 0.0, None

    def fit(self, F, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        Ft = torch.tensor((F - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], self.hidden), torch.nn.Tanh(), torch.nn.Linear(self.hidden, self.hidden), torch.nn.Tanh(), torch.nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(Ft)[:, 0], yt)
            loss.backward()
            opt.step()
        self.w = True

    def prob(self, F):
        if self.net is None:
            return np.ones(F.shape[0])
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor((F - self.mu) / self.sd, dtype=torch.float32))[:, 0]).numpy().astype(float)



class SeqPrior:
    """learned competitor that reads the RAW first-W-tick series of each (input i, output j) pair instead of hand-built descriptors: 1D CNN over the 2-channel window
    [x_i(t), dx_j(t+1)] (standardised), pooled, plus self/action flags, -> P(i affects j). Permutation-equivariant (shared over pairs). Used as a soft gate like NNPrior."""
    windows = True
    rule, tau, y_min_p, w = "soft", 0.5, 0.0, None

    def __init__(self, steps=300, lr=3e-3, seed=0):
        self.steps, self.lr, self.seed = steps, lr, seed
        self.net = None

    @staticmethod
    def pairs(Xs, Ds, out_pos, n_act):
        W, n_in = Xs.shape
        n_out = Ds.shape[1]
        seq = np.zeros((n_out, n_in, 2, W), np.float32)
        seq[:, :, 0, :] = Xs.T[None, :, :]
        seq[:, :, 1, :] = Ds.T[:, None, :]
        flags = np.zeros((n_out, n_in, 2), np.float32)
        for j in range(n_out):
            flags[j, out_pos[j], 0] = 1.0
        flags[:, n_in - n_act:, 1] = 1.0
        return seq.reshape(-1, 2, W), flags.reshape(-1, 2)

    def fit(self, S, Fl, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        from seqnet import SeqNet
        self.net = SeqNet()
        St, Ft, yt = torch.tensor(S), torch.tensor(Fl), torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(yt)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            idx = torch.randint(0, n, (4096,), generator=g)
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(self.net(St[idx], Ft[idx]), yt[idx]).backward()
            opt.step()
        self.w = True

    def prob_windows(self, Xs, Ds, out_pos, n_act):
        import torch
        S, Fl = self.pairs(Xs, Ds, out_pos, n_act)
        if self.net is None:
            return np.ones((Ds.shape[1], Xs.shape[1]))
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(S), torch.tensor(Fl))).numpy().astype(float)
        return p.reshape(Ds.shape[1], Xs.shape[1])
