"""Variable-token transformer predictor (torch, CPU). Every observed variable and action is a TOKEN: value embedding + a fixed random identity vector (drawn per
scenario, so nothing can be memorised by position or name). Each output queries the input tokens with multi-head cross-attention (+ residual of its own token and an MLP).
Self-learning part = an additive attention BIAS b[j,i] (log of a learned relevance prior). Neutral at init: bias = 0 gives the plain transformer exactly (parity).
Online training: one Adam step per tick on the newest sample, prequential error measured BEFORE the step. Settings are experiment settings (expN)."""
import numpy as np
import torch
import torch.nn as nn
from nn import RunStd
from selfnn import pair_descriptors, FEATS


class VarTF(nn.Module):
    def __init__(self, n_in, out_pos, d=32, heads=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("ident", torch.randn(n_in, d, generator=g))
        self.out_pos = list(out_pos)
        self.h, self.d = heads, d
        self.val = nn.Linear(1, d)
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.qbase = nn.Parameter(torch.zeros(d))
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.Tanh(), nn.Linear(2 * d, 1))
        for m in [self.val, self.Wq, self.Wk, self.Wv] + [x for x in self.mlp if isinstance(x, nn.Linear)]:
            nn.init.normal_(m.weight, 0, 1.0 / np.sqrt(m.in_features))
            nn.init.zeros_(m.bias)

    def forward(self, x, bias):
        """x: (B, n_in) standardised inputs; bias: (n_out, n_in) additive attention logits. returns (B, n_out)"""
        B, n_in = x.shape
        tok = self.ident[None] + self.val(x[..., None])                          # (B,n_in,d)
        q = self.Wq(self.ident[self.out_pos] + self.qbase)                       # (n_out,d)
        k, v = self.Wk(tok), self.Wv(tok)
        H, dh = self.h, self.d // self.h
        qh = q.view(-1, H, dh).permute(1, 0, 2)                                  # (H,n_out,dh)
        kh = k.view(B, n_in, H, dh).permute(0, 2, 1, 3)                          # (B,H,n_in,dh)
        vh = v.view(B, n_in, H, dh).permute(0, 2, 1, 3)
        logits = torch.einsum("hod,bhid->bhoi", qh, kh) / np.sqrt(dh) + bias[None, None]
        a = torch.softmax(logits, -1)
        ctx = torch.einsum("bhoi,bhid->bhod", a, vh).permute(0, 2, 1, 3).reshape(B, -1, self.d)
        h = ctx + tok[:, self.out_pos]
        return self.mlp(h)[..., 0]


def run_scenario_tf(X, Y, out_pos, n_act, T, W, seed, mode, prior=None, truth_mask=None, lr=3e-3, floor=1e-3, gate="soft"):
    """mode: 'plain' | 'oracle' (attention only to true parents) | 'prior' (bias = log of the learned relevance prior, floored) """
    torch.manual_seed(seed)
    ns = Y.shape[1]
    n_in = X.shape[1]
    n_out = len(out_pos)
    D = Y - X[:, :ns]
    sx, sy = RunStd(n_in), RunStd(n_out)
    model = VarTF(n_in, out_pos, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bias = torch.zeros(n_out, n_in)
    if mode == "oracle":
        bias = torch.where(torch.tensor(truth_mask > 0), torch.zeros(()), torch.full((), -1e9))
        bias[~torch.tensor(truth_mask.any(1))] = 0.0
    err = np.zeros(T)
    Xs = np.zeros((T, n_in))
    Ds = np.zeros((T, n_out))
    desc = None
    for t in range(T):
        x = X[t]
        y = D[t][out_pos]
        sx.update(x)
        sy.update(y)
        xs, ys = sx(x), sy(y)
        Xs[t], Ds[t] = xs, ys
        if t == W:
            desc = pair_descriptors(Xs[:W], Ds[:W], out_pos, n_act)
            if mode == "prior":
                p = prior.prob(desc.reshape(-1, FEATS)).reshape(n_out, n_in)
                if gate == "hard":                      # thresholded mask: an excluded input can never be re-examined (stress test)
                    keep = p >= 0.5
                    keep[~keep.any(1)] = True
                    bias = torch.tensor(np.where(keep, 0.0, -1e9), dtype=torch.float32)
                else:
                    bias = torch.tensor(np.log(np.clip(p, floor, 1.0)), dtype=torch.float32)
        xt = torch.tensor(xs, dtype=torch.float32)[None]
        yt = torch.tensor(ys, dtype=torch.float32)[None]
        pred = model(xt, bias)
        loss = ((pred - yt) ** 2).mean()
        err[t] = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dict(err=err, desc=desc if desc is not None else pair_descriptors(Xs[:W], Ds[:W], out_pos, n_act), model=model, Xs=Xs, Ds=Ds, bias=bias, sx=sx, sy=sy)


def perm_labels_tf(model, bias, Xs, Ds, last=300, seed=0):
    """own-experience relevance labels from the trained transformer (same paired z-test at Bonferroni level 1/n_in as for the MLP)"""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    n_in = Xs.shape[1]
    n_out = Ds.shape[1]
    Xb = torch.tensor(Xs[-last:], dtype=torch.float32)
    Db = torch.tensor(Ds[-last:], dtype=torch.float32)
    zc = norm.isf(1.0 / n_in)
    lab = np.zeros((n_out, n_in))
    with torch.no_grad():
        base = (model(Xb, bias) - Db) ** 2                                       # (last,n_out)
        for i in range(n_in):
            Xp = Xb.clone()
            Xp[:, i] = Xp[torch.tensor(rng.permutation(len(Xp))), i]
            d = ((model(Xp, bias) - Db) ** 2 - base).numpy()
            z = d.mean(0) / (d.std(0) / np.sqrt(len(d)) + 1e-12)
            lab[:, i] = (z > zc).astype(float)
    return lab



def synth_labels_tf(model, bias, sx, sy, Xs, Ds, out_pos, last=300, seed=0):
    """GENERATED-DATA labels (the model-collapse setting): roll the trained model forward on its own predictions (plus noise at the real residual scale) for the core
    columns, keep the exogenous columns (actions, distractors) from the real record, then compute permutation-importance labels on this SYNTHETIC data. The labels can only
    reflect what the model already believes; no new information from the world enters."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    n_in = Xs.shape[1]
    n_out = Ds.shape[1]
    T = len(Xs)
    start = T - last - 1
    sdx = np.sqrt(np.maximum(sx.s2, 1e-6)); sdy = np.sqrt(np.maximum(sy.s2, 1e-6))
    with torch.no_grad():
        real_pred = model(torch.tensor(Xs[-last:], dtype=torch.float32), bias).numpy()
        resid = (real_pred - Ds[-last:]).std(0)
        syn = np.zeros((last, n_in)); syn[0] = Xs[start + 1]
        Dsyn = np.zeros((last, n_out))
        for t in range(last):
            pred = model(torch.tensor(syn[t], dtype=torch.float32)[None], bias).numpy()[0]
            Dsyn[t] = pred + resid * rng.normal(size=n_out)
            if t + 1 < last:
                nxt = Xs[start + t + 2].copy()
                raw_core = sx.m[out_pos] + sdx[out_pos] * syn[t][out_pos]
                raw_new = raw_core + (sy.m + sdy * Dsyn[t])
                nxt[out_pos] = (raw_new - sx.m[out_pos]) / sdx[out_pos]
                syn[t + 1] = np.clip(nxt, -8, 8)
    Xb = torch.tensor(syn, dtype=torch.float32); Db = torch.tensor(Dsyn, dtype=torch.float32)
    zc = norm.isf(1.0 / n_in)
    lab = np.zeros((n_out, n_in))
    with torch.no_grad():
        base = (model(Xb, bias) - Db) ** 2
        for i in range(n_in):
            Xp = Xb.clone()
            Xp[:, i] = Xp[torch.tensor(rng.permutation(len(Xp))), i]
            d = ((model(Xp, bias) - Db) ** 2 - base).numpy()
            lab[:, i] = ((d.mean(0) / (d.std(0) / np.sqrt(len(d)) + 1e-12)) > zc).astype(float)
    return lab
