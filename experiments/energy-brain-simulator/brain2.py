"""Brain2 = Brain + (1) hardening neurons + (2) always-try-to-expand with a satisfaction criterion."""
import numpy as np
from neuroblock import Brain


class Brain2(Brain):
    def __init__(s, *a, rigid=False, c0=30.0, a0=5.0, expand=False, support=8, frac=0.5,
                 gain_frac=0.15, buf=400, **kw):
        super().__init__(*a, **kw)
        s.rigid, s.c0, s.a0 = rigid, c0, a0
        s.expand, s.support, s.frac, s.gain_frac, s.bufn = expand, support, frac, gain_frac, buf
        s.bx, s.by = [], []
        s.exp_total = 0

    # ---- hardening: learning rate and movement shrink with experience / age ----
    def lr_of(s, i):
        return s.lr / (1 + s.cnt[i] / s.c0) if s.rigid else s.lr

    def eta_vec(s):
        if not s.rigid:
            return super().eta_vec()
        return s.eta / (1 + s.age / s.a0)

    # ---- recent-experience buffer ----
    def remember(s, x, y):
        s.bx.append(x); s.by.append(y)
        if len(s.bx) > s.bufn:
            s.bx.pop(0); s.by.pop(0)

    # ---- expansion drive: propose a specialist, keep it only if it satisfies enough experience ----
    def _expand(s):
        if s.n == 0 or len(s.bx) < 50:
            return
        X = np.array(s.bx); Y = np.array(s.by)
        dist = np.linalg.norm(X[:, None, :] - s.w[None, 1:, :], axis=2)
        resid = dist.min(1)
        poor = resid > s.frac * s.rho
        if poor.sum() < s.support:
            return
        P, Yp, Rp = X[poor], Y[poor], resid[poor]
        DD = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        mem = DD < s.frac * s.rho
        i = int(mem.sum(1).argmax())
        m = mem[i]
        if m.sum() < s.support:
            return
        c = P[m].mean(0)
        gain = np.clip(Rp[m] - np.linalg.norm(P[m] - c, axis=1), 0, None).sum()
        if gain < s.gain_frac * s.rho * s.support:
            return
        lab = np.bincount(Yp[m], minlength=s.K).astype(float)
        par = 1 + int(np.linalg.norm(s.w[1:] - c, axis=1).argmin())
        ang = s.rng.uniform(0, 2 * np.pi)
        p = s.pos[par] + s.spawn_len * np.array([np.cos(ang), np.sin(ang)])
        s.pos = np.vstack([s.pos, p]); s.w = np.vstack([s.w, c]); s.par = np.append(s.par, par)
        s.d = np.append(s.d, s.d[par] + s.spawn_len * 40.0)
        s.cnt = np.append(s.cnt, float(m.sum())); s.ema = np.append(s.ema, 1.0)
        s.age = np.append(s.age, 0); s.win_use = np.append(s.win_use, 1.0)
        s.lab = np.vstack([s.lab, lab[None, :]])
        s.exp_total += 1

    def condense(s):
        if s.expand:
            s._expand()
        super().condense()
