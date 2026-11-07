"""Second-generation genome: grow in ANY direction, with evolvable internal mechanisms (user, 2026-10-26). See DESIGN.md, revision 3.

What changed relative to pop.py (kept, unchanged):
  * connections may run between ANY two nodes (any direction, loops, self-connections); the network is recurrent and runs T internal steps (T is a gene)
  * capacity is not fixed: when any individual runs low on free hidden slots the whole population's tensors are enlarged
  * each hidden node carries an evolvable activation TYPE (stochastic binary, tanh, relu, sin, gaussian, identity)
  * each individual carries its own mutation machinery as genes: step size, mutation count, and the probability of every kind of change (self-adaptation)
  * new structural operators: split_edge (function-preserving insertion of a pass-through node), change_type, change_T, add_loop, plus the old ones
Node layout: [0,d) inputs (clamped every step) | [d, d+H) hidden slots | [d+H, N) outputs (probability that the action's feedback will be 1).
"""
import os, time
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dev.type == 'cuda': torch.cuda.set_per_process_memory_fraction(float(os.environ.get('NB_GPU_FRAC', '0.5')))      # never let a run take (nearly) all VRAM of the display GPU: it fails with an error instead of risking the whole PC
OPS = ('weight', 'add_edge', 'rm_edge', 'add_node', 'rm_node', 'dup_node', 'split_edge', 'change_type', 'change_T', 'add_loop')
OP_PRIOR = [0.40, 0.20, 0.10, 0.12, 0.06, 0.12, 0.10, 0.05, 0.05, 0.05]
NTYPES = 6                                   # 0 stochastic binary | 1 tanh | 2 relu(clamped) | 3 sin | 4 gaussian | 5 identity(clamped)
FIELDS = ('W', 'M', 'b', 'alive', 'typ', 'beta', 'T', 'oplog', 'sigma', 'rate')


class Pop2:
    def __init__(s, P, d, K, H=32, stochastic=True, S=4, tensors=None, init_edges=2):
        s.P, s.d, s.K, s.H = P, d, K, H
        s.N = d + H + K; s.o0 = d + H; s.stochastic = stochastic; s.S = S
        if tensors is not None:
            for k in FIELDS: setattr(s, k, tensors[k])
            return
        N = s.N
        s.W = torch.zeros(P, N, N, device=dev); s.M = torch.zeros(P, N, N, dtype=torch.bool, device=dev)
        s.b = torch.zeros(P, N, device=dev); s.alive = torch.zeros(P, H, dtype=torch.bool, device=dev)
        s.typ = torch.zeros(P, N, dtype=torch.long, device=dev); s.beta = torch.ones(P, device=dev)
        s.T = torch.full((P,), 2, dtype=torch.long, device=dev)
        s.oplog = torch.log(torch.tensor(OP_PRIOR, device=dev)).expand(P, -1).clone()
        s.sigma = torch.full((P,), 0.3, device=dev); s.rate = torch.full((P,), 1.0, device=dev)
        ar = torch.arange(P, device=dev)[:, None]
        for k in range(K):
            if init_edges <= 0:
                s.M[:, :d, s.o0 + k] = True; s.W[:, :d, s.o0 + k] = torch.randn(P, d, device=dev) * 0.5; continue
            idx = torch.randint(0, d, (P, init_edges), device=dev)
            s.M[ar, idx, s.o0 + k] = True; s.W[ar, idx, s.o0 + k] = torch.randn(P, init_edges, device=dev) * 0.5

    # ---- bookkeeping ---------------------------------------------------------------------------------------------------------------
    def _t(s):
        return {k: getattr(s, k) for k in FIELDS}

    def take(s, idx):
        return Pop2(len(idx), s.d, s.K, s.H, s.stochastic, s.S, tensors={k: v[idx].clone() for k, v in s._t().items()})

    @staticmethod
    def cat(a, b):
        return Pop2(a.P + b.P, a.d, a.K, a.H, a.stochastic, a.S, tensors={k: torch.cat([x, y]) for (k, x), y in zip(a._t().items(), b._t().values())})

    def ensure_capacity(s, min_free=6, step=32):
        """grow in any direction: enlarge the hidden block for the whole population when some individual is nearly full (no fixed size ceiling)"""
        if int((~s.alive).sum(1).min()) >= min_free: return False
        N2 = s.N + step
        ix = torch.cat([torch.arange(s.o0, device=dev), torch.arange(s.o0 + step, N2, device=dev)])     # where the old nodes live after the outputs move over
        W2 = torch.zeros(s.P, N2, N2, device=dev); W2[:, ix[:, None], ix[None, :]] = s.W
        M2 = torch.zeros(s.P, N2, N2, dtype=torch.bool, device=dev); M2[:, ix[:, None], ix[None, :]] = s.M
        b2 = torch.zeros(s.P, N2, device=dev); b2[:, ix] = s.b
        t2 = torch.zeros(s.P, N2, dtype=torch.long, device=dev); t2[:, ix] = s.typ
        s.W, s.M, s.b, s.typ = W2, M2, b2, t2
        s.alive = torch.cat([s.alive, torch.zeros(s.P, step, dtype=torch.bool, device=dev)], 1)
        s.H += step; s.N = N2; s.o0 = s.d + s.H
        return True

    # ---- forward: recurrent, T internal steps, stochastic or deterministic nodes ----------------------------------------------------
    def forward_ref(s, x):          # [reference implementation, unchanged; forward() now dispatches to forward_fast]
        """x: (B,d) shared by all individuals -> (P,B,K) predicted P(feedback=1 | x, action k)"""
        P, B, d, N, o0 = s.P, x.shape[0], s.d, s.N, s.o0
        S = s.S if s.stochastic else 1
        xr = x.repeat(S, 1) if S > 1 else x; BS = xr.shape[0]
        al = torch.cat([torch.ones(P, d, device=dev), s.alive.float(), torch.ones(P, s.K, device=dev)], 1)
        Wm = s.W * s.M * al[:, :, None] * al[:, None, :]
        beta = s.beta[:, None, None]
        a = torch.zeros(P, BS, N, device=dev); a[:, :, :d] = xr
        out = torch.zeros(P, BS, s.K, device=dev)
        for t in range(1, int(s.T.max()) + 1):
            z = torch.bmm(a, Wm) + s.b[:, None, :]
            zh = z[:, :, d:o0]; ty = s.typ[:, d:o0][:, None, :]
            h = torch.tanh(zh)
            if s.stochastic:
                h = torch.where(ty == 0, (torch.rand_like(zh) < torch.sigmoid(beta * zh)).float() * 2 - 1, h)
            h = torch.where(ty == 2, torch.relu(zh).clamp(max=3.0), h)
            h = torch.where(ty == 3, torch.sin(zh), h)
            h = torch.where(ty == 4, torch.exp(-zh * zh), h)
            h = torch.where(ty == 5, zh.clamp(-3.0, 3.0), h)
            q = torch.sigmoid(beta * z[:, :, o0:])
            a = torch.cat([xr[None].expand(P, -1, -1), h, q], 2) * al[:, None, :]
            out = torch.where((s.T == t)[:, None, None], q, out)
        return out.view(P, S, B, s.K).mean(1) if S > 1 else out

    def forward(s, x):
        return s.forward_fast(x) if s.fast else s.forward_ref(x)

    # ---- effective complexity (cycle-safe: transitive closure by repeated squaring) -------------------------------------------------------------
    def complexity(s, thr=0.05):
        P, N, d, o0 = s.P, s.N, s.d, s.o0
        al = torch.cat([torch.ones(P, d, device=dev), s.alive.float(), torch.ones(P, s.K, device=dev)], 1)
        Mw = ((s.M & (s.W.abs() > thr)).float()) * al[:, :, None] * al[:, None, :]
        R = (Mw + torch.eye(N, device=dev)[None]).clamp(max=1)
        for _ in range(7): R = (torch.bmm(R, R) > 0).float()
        fin = R[:, :d, :].amax(1); rto = R[:, :, o0:].amax(2)
        nodes = (s.alive.float() * fin[:, d:o0] * rto[:, d:o0]).sum(1)
        edges = (Mw * fin[:, :, None] * rto[:, None, :]).sum((1, 2))
        return edges + nodes, nodes, edges

    def stats(s):
        """what the population has evolved: mechanism genes and structure"""
        a = s.alive; nh = a.float().sum(1)
        ty = s.typ[:, s.d:s.o0]; frac = [float(((ty == k) & a).float().sum() / a.float().sum().clamp(min=1)) for k in range(NTYPES)]
        i = torch.arange(s.N, device=dev)[:, None]; j = torch.arange(s.N, device=dev)[None, :]
        live = s.M & (s.W.abs() > 0.05); back = (live & (j <= i)[None]).float().sum() / live.float().sum().clamp(min=1)
        pr = torch.softmax(s.oplog, 1).mean(0)
        return dict(hidden=float(nh.mean()), capacity=s.H, sigma=float(s.sigma.mean()), rate=float(s.rate.mean()), T=float(s.T.float().mean()), beta=float(s.beta.mean()),
                    back_edges=float(back), types=[round(f, 2) for f in frac], op_probs={n: round(float(p), 3) for n, p in zip(OPS, pr)})

    # ---- mutation: the individual's own genes decide how it mutates (self-adaptive) ---------------------------------------------------------
    def mutate(s):
        P = s.P
        s.sigma.copy_((s.sigma * torch.exp(torch.randn(P, device=dev) * 0.2)).clamp(0.02, 1.0))          # in place, so that mutate() also works on a view of a bigger population
        s.rate.copy_((s.rate * torch.exp(torch.randn(P, device=dev) * 0.2)).clamp(0.3, 4.0))
        s.oplog.copy_((s.oplog + torch.randn_like(s.oplog) * 0.15).clamp(-5, 3))
        nmut = torch.poisson(s.rate).clamp(1, 6).long(); probs = torch.softmax(s.oplog, 1)
        for r in range(int(nmut.max())):                                                         # one sync for the whole loop
            active = nmut > r
            op = torch.where(active, torch.multinomial(probs, 1)[:, 0], len(OPS))               # inactive brains go to a spare bucket
            order = op.argsort(); cnt = torch.bincount(op, minlength=len(OPS) + 1).tolist()     # one host sync per round instead of one per operator
            for t, (name, part) in enumerate(zip(OPS, order.split(cnt))):
                if cnt[t]: getattr(s, '_' + name)(part)

    def _weight(s, idx, sigma=None):
        """each existing edge is perturbed with probability 0.15 by N(0, sigma^2) (performance pass 2: drawn on the edge list, O(edges) instead of O(N^2); same distribution)"""
        n = len(idx); sg = s.sigma[idx] if sigma is None else (sigma.view(-1) if torch.is_tensor(sigma) else torch.full((n,), float(sigma), device=dev))
        nz = s.M[idx].nonzero(); nz = nz[torch.rand(len(nz), device=dev) < 0.15]
        s.W[idx[nz[:, 0]], nz[:, 1], nz[:, 2]] += torch.randn(len(nz), device=dev) * sg[nz[:, 0]]
        s.b[idx] += torch.randn(n, s.N, device=dev) * (sg * 0.3)[:, None] * (torch.rand(n, s.N, device=dev) < 0.15)
        s.beta[idx] = (s.beta[idx] * torch.exp(torch.randn(n, device=dev) * 0.2 * (torch.rand(n, device=dev) < 0.15))).clamp(0.25, 8.0)

    def _pick_edge(s, idx):
        """one uniformly random existing edge per brain in idx (same distribution as multinomial over the mask, but O(edges)): returns (has_edge mask, i, j) for the brains that have one"""
        nz = s.M[idx].nonzero(); n = len(idx); cnt = torch.bincount(nz[:, 0], minlength=n); ok = cnt > 0
        off = torch.cumsum(cnt, 0) - cnt; r = torch.minimum((torch.rand(n, device=dev) * cnt).long(), (cnt - 1).clamp(min=0))
        e = nz[(off + r)[ok]]
        return ok, e[:, 1], e[:, 2]

    def _add_edge(s, idx):                                                                        # ANY direction: source any node, target any non-input node
        n = len(idx); i = torch.randint(0, s.N, (n,), device=dev); j = torch.randint(s.d, s.N, (n,), device=dev)
        s.M[idx, i, j] = True; s.W[idx, i, j] = torch.randn(n, device=dev) * 0.5

    def _rm_edge(s, idx):
        ok, i, j = s._pick_edge(idx)
        if len(i): s.M[idx[ok], i, j] = False

    def _free_slot(s, idx):
        dead = ~s.alive[idx]; has = dead.any(1)
        return has, s.d + dead.float().argmax(1)

    def _add_node(s, idx):
        has, slot = s._free_slot(idx); idx, slot = idx[has], slot[has]; n = len(idx)
        if not n: return
        src = torch.randint(0, s.N, (n, 3), device=dev); tgt = torch.randint(s.d, s.N, (n, 2), device=dev); ar = idx[:, None]
        s.alive[idx, slot - s.d] = True; s.b[idx, slot] = 0
        s.typ[idx, slot] = torch.where(torch.rand(n, device=dev) < 0.5, torch.zeros(n, dtype=torch.long, device=dev), torch.randint(1, NTYPES, (n,), device=dev))
        s.M[ar.expand(-1, 3), src, slot[:, None].expand(-1, 3)] = True; s.W[ar.expand(-1, 3), src, slot[:, None].expand(-1, 3)] = torch.randn(n, 3, device=dev) * 0.5
        s.M[ar.expand(-1, 2), slot[:, None].expand(-1, 2), tgt] = True; s.W[ar.expand(-1, 2), slot[:, None].expand(-1, 2), tgt] = torch.randn(n, 2, device=dev) * 0.3

    def _rm_node(s, idx):
        a = s.alive[idx].float(); ok = a.sum(1) > 0; idx, a = idx[ok], a[ok]
        if not len(idx): return
        h = torch.multinomial(a, 1)[:, 0]; slot = s.d + h
        s.alive[idx, h] = False; s.M[idx, :, slot] = False; s.M[idx, slot, :] = False

    def _dup_node(s, idx):                                                                        # gene duplication then drift
        a = s.alive[idx].float(); ok = a.sum(1) > 0; idx, a = idx[ok], a[ok]
        if not len(idx): return
        k = s.d + torch.multinomial(a, 1)[:, 0]
        has, s2 = s._free_slot(idx); idx, k, s2 = idx[has], k[has], s2[has]
        if not len(idx): return
        s.alive[idx, s2 - s.d] = True; s.typ[idx, s2] = s.typ[idx, k]; s.b[idx, s2] = s.b[idx, k]
        s.M[idx, :, s2] = s.M[idx, :, k]; s.W[idx, :, s2] = s.W[idx, :, k] + torch.randn(len(idx), s.N, device=dev) * 0.05 * s.M[idx, :, k]
        s.M[idx, s2, :] = s.M[idx, k, :]; s.W[idx, s2, :] = s.W[idx, k, :] + torch.randn(len(idx), s.N, device=dev) * 0.05 * s.M[idx, k, :]

    def _split_edge(s, idx):                                                                      # NEAT-style: insert a pass-through node on an edge (function-preserving)
        ok, i, j = s._pick_edge(idx); idx = idx[ok]
        if not len(idx): return
        has, slot = s._free_slot(idx); idx, i, j, slot = idx[has], i[has], j[has], slot[has]
        if not len(idx): return
        w = s.W[idx, i, j].clone()
        s.M[idx, i, j] = False
        s.alive[idx, slot - s.d] = True; s.typ[idx, slot] = 5; s.b[idx, slot] = 0                  # identity node
        s.M[idx, i, slot] = True; s.W[idx, i, slot] = 1.0; s.M[idx, slot, j] = True; s.W[idx, slot, j] = w

    def _change_type(s, idx):
        a = s.alive[idx].float(); ok = a.sum(1) > 0; idx, a = idx[ok], a[ok]
        if not len(idx): return
        k = s.d + torch.multinomial(a, 1)[:, 0]; s.typ[idx, k] = torch.randint(0, NTYPES, (len(idx),), device=dev)

    def _change_T(s, idx):
        s.T[idx] = (s.T[idx] + torch.where(torch.rand(len(idx), device=dev) < 0.5, -1, 1)).clamp(s.min_T, 6)

    def _add_loop(s, idx):                                                                         # a self-connection or feedback edge on a hidden node = memory
        a = s.alive[idx].float(); ok = a.sum(1) > 0; idx, a = idx[ok], a[ok]
        if not len(idx): return
        k = s.d + torch.multinomial(a, 1)[:, 0]; s.M[idx, k, k] = True; s.W[idx, k, k] = torch.randn(len(idx), device=dev) * 0.5


    # ---- random-size variation (user's repopulation rule, 2026-10-26): each copy gets a variation whose SCALE is itself random ------------------------------
    def vary(s):
        """tiny nudge of a few numbers (35%) | broad weight perturbation with random step (30%) | 1-3 random structural operations (25%) | re-randomise an entire module:
        1-8 hidden nodes with all their connections, type and bias (10%)"""
        P = s.P; u = torch.rand(P, device=dev); ar = torch.arange(P, device=dev)
        i1 = ar[u < 0.35]
        for _ in range(3):                                                                           # nudge up to three existing weights by 0.01..0.3
            ok, pi, pj = s._pick_edge(i1); ii = i1[ok]
            if len(ii): s.W[ii, pi, pj] += torch.randn(len(ii), device=dev) * 10 ** (-2 + 1.5 * torch.rand(len(ii), device=dev))
        i2 = ar[(u >= 0.35) & (u < 0.65)]
        if len(i2): s._weight(i2, 10 ** (-1 + torch.rand(len(i2), device=dev)))                     # step 0.1..1.0
        i3 = ar[(u >= 0.65) & (u < 0.9)]
        if len(i3):
            pr = torch.ones(len(OPS), device=dev); pr[0] = 0.0                                       # structural operators only
            nops = torch.randint(1, 4, (len(i3),), device=dev)
            for r in range(3):
                act = i3[nops > r]
                if not len(act): break
                op = torch.multinomial(pr.expand(len(act), -1), 1)[:, 0]
                order = op.argsort(); cnt = torch.bincount(op, minlength=len(OPS)).tolist()
                for t, (name, part) in enumerate(zip(OPS, order.split(cnt))):
                    if t > 0 and cnt[t]: getattr(s, '_' + name)(act[part])
        i4 = ar[u >= 0.9]
        if len(i4):                                                                                  # module replacement
            a = s.alive[i4]; cnt = a.sum(1); m = torch.randint(1, 9, (len(i4),), device=dev)
            sc = torch.rand(len(i4), s.H, device=dev) * a - (~a).float()                              # dead slots rank last
            rank = sc.argsort(1, descending=True).argsort(1); chosen = (rank < m[:, None]) & a
            nm = torch.zeros(len(i4), s.N, dtype=torch.bool, device=dev); nm[:, s.d:s.o0] = chosen
            none = ~chosen.any(1)                                                                     # no hidden nodes yet: re-randomise one output node's inputs instead
            if bool(none.any()): nm[none, s.o0 + torch.randint(0, s.K, (int(none.sum()),), device=dev)] = True
            sel = (nm[:, None, :] | nm[:, :, None]) & s.M[i4]
            s.W[i4] = torch.where(sel, torch.randn(len(i4), s.N, s.N, device=dev) * 0.5, s.W[i4])
            s.typ[i4] = torch.where(nm, torch.randint(0, NTYPES, (len(i4), s.N), device=dev), s.typ[i4])
            s.b[i4] = torch.where(nm, torch.randn(len(i4), s.N, device=dev) * 0.3, s.b[i4])


    # ---- large-population helpers (10,000-brain runs): views, chunked evaluation, in-place replacement -------------------------------------------------
    def view(s, a, b):
        """a Pop2 sharing storage with s[a:b] (no copy); in-place operations on it change s"""
        return Pop2(b - a, s.d, s.K, s.H, s.stochastic, s.S, tensors={k: getattr(s, k)[a:b] for k in FIELDS})

    def assign_where(s, mask, src):
        """s[i] = src[i] for every i with mask[i] (in place)"""
        for k in FIELDS: getattr(s, k)[mask] = getattr(src, k)[mask]

    @torch.no_grad()
    def forward_chunked(s, x, chunk=500):
        out = torch.empty(s.P, x.shape[0], s.K, device=dev)
        for idx in s.sorted_chunks(chunk): out[idx] = s.take(idx).forward(x)
        return out

    @torch.no_grad()
    def complexity_chunked(s, chunk=2000):
        out = [s.view(a, min(a + chunk, s.P)).complexity() for a in range(0, s.P, chunk)]
        return tuple(torch.cat([o[i] for o in out]) for i in range(3))

    def mutate_chunked(s, chunk=2000):
        for a in range(0, s.P, chunk): s.view(a, min(a + chunk, s.P)).mutate()

    def vary_chunked(s, chunk=2000):
        for a in range(0, s.P, chunk): s.view(a, min(a + chunk, s.P)).vary()


    def assign_idx(s, idx, src):
        """s[idx] = src (all fields, in place); src has len(idx) genomes"""
        for k in FIELDS: getattr(s, k)[idx] = getattr(src, k)

    # ---- fast forward (performance pass, 2026-10-26): same function as forward(), computed only where it matters ------------------------------------------
    fast = True            # forward() dispatches to forward_fast(); set Pop2.fast = False to use the original reference implementation

    def live_extent(s):
        """per brain: index of its highest alive hidden slot + 1 (0 if none)"""
        return (s.alive.long() * torch.arange(1, s.H + 1, device=dev)).amax(1)

    fwd_budget = float(os.environ.get('NB_FWD_MB', '700')) * 2**20      # bytes of activations one forward block may use (keeps VRAM bounded for any population size)
    fwd_dtype = torch.bfloat16 if (dev.type == 'cuda' and os.environ.get('NB_FWD_DTYPE', 'fp32') == 'bf16') else torch.float32      # default fp32 = bit-identical to the reference forward (test_fast.py); NB_FWD_DTYPE=bf16 is ~1.7x faster on the forward but outputs differ by up to 0.06

    @torch.no_grad()
    def forward_fast(s, x):
        """performance pass 2 (2026-10-26): exact same function as before (outputs identical up to random draws), but (a) a block is split automatically so its activations fit the memory budget, so
        no chunk size has to be chosen by hand, (b) brains with no stochastic node use ONE noise sample instead of S (exact: the S samples would be identical), (c) one host sync per block."""
        hm = int(s.live_extent().max()) if s.P else 0
        BS = x.shape[-2] * (s.S if s.stochastic else 1); per = BS * (s.d + hm + s.K) * 4 * 6
        n = max(1, int(s.fwd_budget // max(per, 1)))
        if s.P <= n: return s._fwd_block(x)
        return torch.cat([s.view(a, min(a + n, s.P))._fwd_block(x) for a in range(0, s.P, n)])

    min_T = 1               # diagnostic switch (multi_big --min_T): lowest number of internal steps evolution may choose
    trace = None            # set to a dict to collect per-stage seconds of _fwd_block (bench only)

    @torch.no_grad()
    def _fwd_block(s, x):
        tr = s.trace; tk = [time.time()]
        def mk(name):
            if tr is not None: torch.cuda.synchronize(); n = time.time(); tr[name] = tr.get(name, 0.0) + n - tk[0]; tk[0] = n
        per_brain = x.dim() == 3                                  # x of shape (P,B,d): every brain gets its own input (used by the survival world)
        P, B, d, K = s.P, x.shape[-2], s.d, s.K
        hm, Tm = torch.stack([s.live_extent().max(), s.T.max()]).tolist() if P else (0, 1)
        if Tm == 1: hm = 0                                         # exact: at step 1 only the inputs are non-zero, so with T=1 the hidden nodes cannot influence any output
        present = set(torch.unique(s.typ[:, d:d + hm][s.alive[:, :hm]]).tolist()) if hm else set()
        noisy = s.stochastic and 0 in present and Tm >= 2; S = s.S if noisy else 1               # no stochastic node that can matter: every noise sample would be identical
        xr = (x.repeat(1, S, 1) if S > 1 else x) if per_brain else (x.repeat(S, 1) if S > 1 else x); BS = xr.shape[-2]
        cols = torch.cat([torch.arange(d, device=dev), d + torch.arange(hm, device=dev), s.o0 + torch.arange(K, device=dev)])
        al = torch.cat([torch.ones(P, d, device=dev), s.alive[:, :hm].float(), torch.ones(P, K, device=dev)], 1)
        Wc = (s.W * s.M)[:, cols][:, :, cols] * al[:, :, None] * al[:, None, :]
        dt = s.fwd_dtype; Wc = Wc.to(dt); mk('setup W')
        bc = s.b[:, cols][:, None, :].to(dt); beta = s.beta[:, None, None].to(dt); Nc = d + hm + K
        a = torch.zeros(P, BS, Nc, device=dev, dtype=dt); a[:, :, :d] = xr          # xr broadcasts for a shared input (BS,d) or is per brain (P,BS,d)
        ty = s.typ[:, d:d + hm][:, None, :]
        out = torch.zeros(P, BS, K, device=dev, dtype=dt); mk('setup a')
        for t in range(1, Tm + 1):
            if t == Tm:                                              # last step: nothing reads the hidden nodes afterwards, so only the output columns are computed
                q = torch.sigmoid(beta * torch.baddbmm(bc[:, :, d + hm:], a, Wc[:, :, d + hm:])); out = torch.where((s.T == t)[:, None, None], q, out); mk('last step (outputs only)')
                break
            z = torch.baddbmm(bc, a, Wc); mk('matmul')
            if hm:
                zh = z[:, :, d:d + hm]; h = torch.tanh(zh)
                if noisy: h = torch.where(ty == 0, (torch.rand_like(zh) < torch.sigmoid(beta * zh)).to(dt) * 2 - 1, h)
                if 2 in present: h = torch.where(ty == 2, torch.relu(zh).clamp(max=3.0), h)
                if 3 in present: h = torch.where(ty == 3, torch.sin(zh), h)
                if 4 in present: h = torch.where(ty == 4, torch.exp(-zh * zh), h)
                if 5 in present: h = torch.where(ty == 5, zh.clamp(-3.0, 3.0), h)
                a[:, :, d:d + hm] = h
            mk('hidden activations')
            q = torch.sigmoid(beta * z[:, :, d + hm:]); a[:, :, d + hm:] = q
            out = torch.where((s.T == t)[:, None, None], q, out); mk('outputs')
        out = out.float()
        return out.view(P, S, B, K).mean(1) if S > 1 else out

    @torch.no_grad()
    def sorted_chunks(s, chunk):
        """groups of brains with similar T and live size, so no brain pays for another's extra steps or empty slots"""
        key = s.T * 1024 + s.live_extent()
        return torch.argsort(key).split(chunk)


# ---- refill by random collections of the current brains (user, 2026-10-26) ---------------------------------------------------------------------------------------
MODES = ('copy', 'vary', 'cross', 'blend', 'scramble', 'fresh', 'opposite')
MODE_P = (0.25, 0.25, 0.20, 0.10, 0.10, 0.10, 0.0)                                                # 'opposite' dropped at the user's request (kept as a mode, probability 0)


@torch.no_grad()
def spawn_children(S, n, probs=MODE_P, chunk=1500):
    """n new brains built from the current brains S. Each child draws its own mode: carbon copy of a random brain | random-sized variation | crossover of 3 random brains
    (each node's connections from one of them) | blend (average) of 2 | scramble (structure kept, all weights re-randomised) | fresh (brand-new random brain) |
    exact opposite (all connections into the outputs negated, so the brain predicts the complement). Parents are uniformly random, not rank-weighted.
    returns (children, mode_id (n,), parent_index (n,))"""
    nk = S.P; mode = torch.multinomial(torch.tensor(probs, device=dev), n, replacement=True)
    parts = mode.argsort().split(torch.bincount(mode, minlength=len(MODES)).tolist())                 # one host sync for all modes
    a = torch.randint(0, nk, (n,), device=dev); b = torch.randint(0, nk, (n,), device=dev); c = torch.randint(0, nk, (n,), device=dev)
    out = S.take(a)
    N, d, o0, H = S.N, S.d, S.o0, S.H
    for ch in parts[1].split(chunk):                                           # random-sized variation
        sub = out.take(ch); sub.vary(); out.assign_idx(ch, sub)
    for ch in parts[2].split(chunk):                                           # crossover of three random brains, node by node
        Da, Db, Dc = S.take(a[ch]), S.take(b[ch]), S.take(c[ch]); m = len(ch)
        pk = torch.randint(0, 3, (m, N), device=dev); sel = lambda x, y, z, pp: torch.where(pp == 0, x, torch.where(pp == 1, y, z))
        out.W[ch] = sel(Da.W, Db.W, Dc.W, pk[:, None, :]); out.M[ch] = sel(Da.M, Db.M, Dc.M, pk[:, None, :])
        out.b[ch] = sel(Da.b, Db.b, Dc.b, pk); out.typ[ch] = sel(Da.typ, Db.typ, Dc.typ, pk); out.alive[ch] = sel(Da.alive, Db.alive, Dc.alive, pk[:, d:o0])
        dn = torch.randint(0, 3, (m,), device=dev)                                                  # mechanism genes from one donor
        for f in ('T', 'beta', 'sigma', 'rate'): getattr(out, f)[ch] = sel(getattr(Da, f), getattr(Db, f), getattr(Dc, f), dn)
        out.oplog[ch] = sel(Da.oplog, Db.oplog, Dc.oplog, dn[:, None])
        del Da, Db, Dc
    for ch in parts[3].split(chunk):                                           # blend: average of two brains (union of connections)
        Da, Db = S.take(a[ch]), S.take(b[ch])
        both = Da.M & Db.M; out.M[ch] = Da.M | Db.M
        out.W[ch] = torch.where(both, (Da.W + Db.W) / 2, torch.where(Da.M, Da.W, Db.W)); out.b[ch] = (Da.b + Db.b) / 2; out.alive[ch] = Da.alive | Db.alive
        out.typ[ch] = torch.where(torch.rand(len(ch), N, device=dev) < 0.5, Da.typ, Db.typ); del Da, Db
    ch = parts[4]                                                                                    # scramble: same structure, every weight re-randomised
    for cc in ch.split(chunk):
        out.W[cc] = torch.where(out.M[cc], torch.randn(len(cc), N, N, device=dev) * 0.5, torch.zeros(1, device=dev)); out.b[cc] = torch.randn(len(cc), N, device=dev) * 0.3
    ch = parts[5]                                                                                    # fresh: brand-new random brains
    for cc in ch.split(chunk): out.assign_idx(cc, Pop2(len(cc), d, S.K, H, S.stochastic, S.S, init_edges=0))
    ch = parts[6]                                                                                    # exact opposite: negate everything feeding the outputs
    out.W[ch, :, o0:] *= -1.0; out.b[ch, o0:] *= -1.0
    return out, mode, a
