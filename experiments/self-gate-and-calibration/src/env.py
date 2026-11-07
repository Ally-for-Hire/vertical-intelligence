"""Continuous car on a tile-grid track, with causal variables and classed distractor variables.

The agent sees (a) a car-frame KxK Boolean tile view and (b) a vector of named scalar variables.
Actions are 6 combos: steer in {L,S,R} x throttle in {0,1}. The env never resets on leaving the track
(drag x4 off-track), so the stream is continuous; it only re-seats the car if it leaves the grid.
"""
import numpy as np

DT = 0.1
ACCEL = 4.0
K_FWD, K_LAT = 7, 7
CLASSES = ("noise", "clock", "copy_v", "lamp", "lag_score")   # distractor classes a..e


class Track:
    def __init__(self, seed, size=48, width=4.0, kind=None):
        rng = np.random.default_rng(seed)
        self.size, self.width = size, width
        n = 400
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        R = size * 0.32
        amp = rng.uniform(0.04, 0.13, 2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        k = (2, 3) if kind is None else kind
        r = R * (1 + amp[0] * np.sin(k[0] * th + ph[0]) + amp[1] * np.sin(k[1] * th + ph[1]))
        self.cx = size / 2 + r * np.cos(th)
        self.cy = size / 2 + r * np.sin(th)
        d = np.stack([np.roll(self.cx, -1) - self.cx, np.roll(self.cy, -1) - self.cy], 1)
        self.tang = np.arctan2(d[:, 1], d[:, 0])
        gx, gy = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
        dd = np.hypot(gx[..., None] - self.cx, gy[..., None] - self.cy).min(-1)
        self.mask = dd <= width / 2
        self.start = (self.cx[0], self.cy[0], self.tang[0])

    def on(self, x, y):
        ix, iy = int(x), int(y)
        return 0 <= ix < self.size and 0 <= iy < self.size and bool(self.mask[iy, ix])

    def frame(self, x, y):
        """nearest centerline index, distance, signed lateral offset, tangent angle"""
        dx, dy = self.cx - x, self.cy - y
        i = int(np.argmin(dx * dx + dy * dy))
        t = self.tang[i]
        lat = -np.sin(t) * (x - self.cx[i]) + np.cos(t) * (y - self.cy[i])
        return i, abs(lat), lat, t


_FI = np.repeat(np.arange(K_FWD), K_LAT).astype(float)
_LJ = np.tile(np.arange(K_LAT) - K_LAT // 2, K_FWD).astype(float)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CarEnv:
    """Variables exposed each tick (scalars): causal block, then n_distract distractors."""
    CAUSAL = ("v", "omega", "sin", "cos", "lat", "herr", "ontrack", "drag", "score")

    def __init__(self, seed=0, n_distract=0, drag=None, track=None, time_noise=True, respawn=60, mirror=False, classes=None):
        self.respawn, self.off_ticks = respawn, 0
        self.mirror = mirror
        self.rng = np.random.default_rng(seed + 10_000)
        self.track = track if track is not None else Track(seed)
        self.drag = float(drag if drag is not None else self.rng.uniform(0.6, 1.0))
        self.vmax = ACCEL / self.drag
        self.n_d = n_distract
        cl = classes or CLASSES
        self.dist_class = [cl[i % len(cl)] for i in range(n_distract)]
        self.ar = np.zeros(n_distract)
        self.periods = self.rng.uniform(20, 200, n_distract)
        self.names = list(self.CAUSAL) + [f"d{i}_{c}" for i, c in enumerate(self.dist_class)]
        self.t = 0
        self.lamp = np.zeros(n_distract)
        self.prev_score = 0.0
        self.reset()

    def set_drag(self, d):
        self.drag = float(d)
        self.vmax = ACCEL / self.drag

    def reset(self):
        x, y, th = self.track.start
        self.x, self.y, self.th, self.v, self.om = x, y, th, 0.0, 0.0
        self._refresh()

    def _refresh(self):
        tr = self.track
        i, adl, lat, tg = tr.frame(self.x, self.y)
        self.lat_d, self.lat, self.herr = adl, lat, wrap(self.th - tg)
        self.ontrack = 1.0 if tr.on(self.x, self.y) else 0.0
        hw = tr.width / 2
        self.score = self.ontrack * (self.v / self.vmax) * max(0.0, 1 - adl / hw)

    def view(self):
        c, s = np.cos(self.th), np.sin(self.th)
        px = self.x + c * _FI - s * _LJ
        py = self.y + s * _FI + c * _LJ
        ix, iy = px.astype(int), py.astype(int)
        ok = (ix >= 0) & (ix < self.track.size) & (iy >= 0) & (iy < self.track.size)
        out = np.zeros(len(ix), np.uint8)
        out[ok] = self.track.mask[iy[ok], ix[ok]]
        return out

    def scalars(self):
        base = [self.v, self.om, np.sin(self.th), np.cos(self.th), self.lat, self.herr,
                self.ontrack, self.drag, self.score]
        if self.n_d:
            d = np.empty(self.n_d)
            for i, c in enumerate(self.dist_class):
                if c == "noise":
                    d[i] = self.rng.random()
                elif c == "clock":
                    d[i] = np.sin(2 * np.pi * self.t / self.periods[i])
                elif c == "copy_v":
                    d[i] = self.v + self.rng.normal(0, 0.3)
                elif c == "lamp":
                    d[i] = self.lamp[i]
                elif c == "ar1":
                    d[i] = self.ar[i]
                else:
                    d[i] = self.prev_score
            base = base + list(d)
        return np.asarray(base, float)

    def step(self, a):
        steer = (-1, 0, 1)[a % 3]      # left = +angle in math orientation: use sign only
        if self.mirror:
            steer = -steer
        thr = a // 3
        off = 1.0 if self.ontrack else 4.0
        self.prev_score = self.score
        self.v += (ACCEL * thr - self.drag * off * self.v) * DT
        self.om = 1.5 * steer * (self.v / self.vmax)
        self.th += self.om * DT
        self.x += np.cos(self.th) * self.v * DT
        self.y += np.sin(self.th) * self.v * DT
        for i, c in enumerate(self.dist_class):
            if c == "lamp":
                self.lamp[i] = 0.9 * self.lamp[i] + 0.1 * (steer == -1)
        self.ar = 0.9 * self.ar + self.rng.normal(0, 0.3, self.n_d)
        self.t += 1
        self.off_ticks = 0 if self.track.on(self.x, self.y) else self.off_ticks + 1
        if self.respawn and self.off_ticks >= self.respawn:            # track marshal: put the car back on the track, at rest
            i = int(self.rng.integers(len(self.track.cx)))
            self.x, self.y, self.th = self.track.cx[i], self.track.cy[i], self.track.tang[i]
            self.v = self.om = 0.0
            self.off_ticks = 0
        if not (0 <= self.x < self.track.size and 0 <= self.y < self.track.size):
            self.reset()
        else:
            self._refresh()

    # ground truth (variable-level, lag-1) for structure scoring --------------------------------
    def truth_edges(self):
        """set of (parent_var, child_var); 'a_L','a_R','a_T' are action variables, 'grid' the tile view."""
        E = set()
        st = ["v", "omega", "sin", "cos", "lat", "herr", "ontrack", "score"]
        E |= {("v", "v"), ("a_T", "v"), ("ontrack", "v")}
        E |= {("a_L", "omega"), ("a_R", "omega"), ("v", "omega")}
        E |= {("sin", "sin"), ("cos", "sin"), ("omega", "sin"), ("sin", "cos"), ("cos", "cos"), ("omega", "cos")}
        for p in ("lat", "herr", "v", "sin", "cos", "omega"):
            E |= {(p, "lat"), (p, "herr")}
        E |= {("herr", "lat"), ("lat", "ontrack"), ("herr", "ontrack"), ("v", "ontrack"), ("ontrack", "ontrack")}
        E |= {("drag", "drag")}
        for p in ("v", "lat", "herr", "ontrack", "a_T", "a_L", "a_R", "omega", "score", "sin", "cos"):
            E.add((p, "score"))
        for p in ("grid", "v", "omega", "a_L", "a_R", "a_T", "lat", "herr", "sin", "cos", "ontrack"):
            E.add((p, "grid"))
        for i, c in enumerate(self.dist_class):
            n = f"d{i}_{c}"
            if c == "clock":
                E.add((n, n))
            elif c == "copy_v":
                E |= {("v", n), ("a_T", n), ("ontrack", n)}
            elif c == "lamp":
                E |= {(n, n), ("a_L", n)}
            elif c == "lag_score":
                E |= {("score", n)}
        return E
