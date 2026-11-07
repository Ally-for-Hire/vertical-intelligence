"""A concept-library learner for the rune universe, metered in energy, with a sleep that it chooses itself.

Three modes (everything else identical, so differences are due to what is remembered):
  fresh : no memory. Every episode, every function is found by searching small programs from scratch (smallest consistent program first).
  lib   : remembers the programs it has solved and tries those first (it can re-use a function it already knows under a new word).
  abs   : lib + SLEEP. While asleep it looks across everything it knows for parts that keep recurring and names them (abstractions: new building blocks).
          Search then treats each named block as a single step, so programs built from known blocks are found much sooner (the DreamCoder idea).

Sleep is the learner's own choice, never scheduled:
  NEED  = sleep debt (energy spent since the last sleep / need_scale). While in debt every evaluation costs more (tired_cost), so ignoring sleep has a price.
  WANT  = backlog (programs learned since the last sleep / library size) x (1 - novelty), where novelty is the share of this episode's functions that needed a fresh search.
  It sleeps when max(need, want) >= sleep_threshold, wakes by itself once debt and backlog are low (or after sleep_max episodes as a runaway guard).
  Episodes that arrive while it sleeps are missed (they are not learned from and not scored).

Energy: every evaluation costs, there is a cost just to take part in an episode, stored memories cost upkeep, and consolidating costs. Nothing is free.
"""
import math
from collections import Counter

import runes as R

C_EVAL = 0.0004       # energy per (candidate program, example) evaluation
C_EPISODE = 0.05      # sensing and being alive for one episode
C_UPKEEP = 0.0002     # per remembered program, per episode
C_COMPRESS = 0.002    # per program node examined while consolidating
MAX_SIZE = 9
SEARCH_CAP = 300_000  # evaluations before a search gives up on one function
TIRED_CAP = 3.0

UNARY = {'rev': lambda v: v[::-1], 'head': lambda v: v[:1], 'tail': lambda v: v[1:]}


def subterms(t):
    yield t
    if t[0] in ('cat', 'rev', 'head', 'tail'):
        for c in t[1:]:
            yield from subterms(c)
    elif t[0] == 'lib':
        for c in t[2:]:
            yield from subterms(c)


def children(t):
    return t[2:] if t[0] == 'lib' else (t[1:] if t[0] in ('cat', 'rev', 'head', 'tail') else ())


def match(pat, t, binds):
    if pat[0] == 'v':
        if pat[1] in binds:
            return binds[pat[1]] == t
        binds[pat[1]] = t
        return True
    if pat[0] != t[0]:
        return False
    if pat[0] == 'a':
        return pat[1] == t[1]
    if pat[0] == 'lib' and pat[1] != t[1]:
        return False
    pc, tc = children(pat), children(t)
    return len(pc) == len(tc) and all(match(a, b, binds) for a, b in zip(pc, tc))


def rebuild(t, kids):
    return (t[0], t[1], *kids) if t[0] == 'lib' else (t[0], *kids)


def rewrite(t, pat, name, nv):
    b = {}
    if t[0] != 'a' and match(pat, t, b):
        return ('lib', name, *[rewrite(b[i], pat, name, nv) for i in range(nv)])
    return rebuild(t, [rewrite(c, pat, name, nv) for c in children(t)]) if children(t) else t


def patterns(sub):
    """candidate abstractions from one subterm: generalise its argument leaves; and also a repeated sub-structure"""
    def gen(t, mapping, repl):
        if t in repl:
            return repl[t]
        if t[0] == 'a':
            return mapping.setdefault(t, ('v', len(mapping) + (1 if repl else 0)))
        if not children(t):
            return t
        return rebuild(t, [gen(c, mapping, repl) for c in children(t)])
    out = []
    p = gen(sub, {}, {})
    out.append(p)
    counts = Counter(s for s in subterms(sub) if s is not sub and s[0] != 'a' and children(s))
    for s, c in counts.items():
        if c >= 2 and s != sub:
            out.append(gen(sub, {}, {s: ('v', 0)}))
    return [p for p in out if len({v for v in vars_of(p)}) <= 2]


def vars_of(p):
    if p[0] == 'v':
        yield p[1]
    for c in children(p):
        yield from vars_of(c)


def nvars(p):
    return len(set(vars_of(p)))


class Learner:
    def __init__(self, mode, need_scale=120.0, sleep_threshold=0.8, tired_cost=0.5, want_weight=1.0, wake_level=0.2, sleep_max=3):
        assert mode in ('fresh', 'lib', 'abs')
        self.mode = mode
        self.library = {}          # key -> {'arity','prog','uses'}
        self.abstractions = {}     # name -> {'arity','body'}
        self.need_scale, self.sleep_threshold, self.tired_cost = need_scale, sleep_threshold, tired_cost
        self.want_weight, self.wake_level, self.sleep_max = want_weight, wake_level, sleep_max
        self.debt, self.new_since_sleep, self.asleep, self.slept_for = 0.0, 0, False, 0
        self.sleeps = 0
        self.work = 0

    # ------------------------------------------------------------ one episode
    def run_episode(self, ep):
        self.work = 0
        searches = hits = 0
        study = [(R.parse(t), o) for t, o in ep['study']]
        queries = [(R.parse(t), o) for t, o in ep['query']]
        prims = {t[0][0]: o for t, o in ep['study'] if len(t) == 1}
        arity = {w: (1 if k == 'U' else 2) for w, k in ep['kind'].items()}
        funcs, unresolved, log = {}, set(arity), []
        progress = True
        while unresolved and progress:
            progress = False
            for f in sorted(unresolved):
                ctxs = self._contexts(f, study, prims, funcs)
                if not ctxs:
                    continue
                w0 = self.work
                prog, how = self._solve(arity[f], ctxs)
                if prog is None:
                    prog, how = (R.A(0), 'failed')
                funcs[f] = (arity[f], prog)
                unresolved.discard(f)
                progress = True
                hits += how == 'lib'
                searches += how != 'lib'
                log.append(dict(name=ep['word_to_name'][f], how=how, work=self.work - w0, correct=self._same(prog, ep['funcs_true'][f][1], arity[f])))
        for f in unresolved:
            funcs[f] = (arity[f], R.A(0))
        correct = 0
        for tree, out in queries:
            try:
                correct += R.evaluate(tree, prims, funcs, self.abstractions) == out
            except KeyError:
                pass
        tired = 1.0 + self.tired_cost * min(self.debt, TIRED_CAP) if self.mode == 'abs' else 1.0
        energy = self.work * C_EVAL * tired + C_EPISODE + C_UPKEEP * len(self.library) * (1 if self.mode != 'fresh' else 0)
        if self.mode != 'fresh':
            for f, (a, prog) in funcs.items():
                self._remember(a, prog)
        novelty = searches / max(1, searches + hits)
        return dict(acc=correct / len(queries), energy=energy, work=self.work, searches=searches, hits=hits, novelty=novelty, functions=log)

    def _remember(self, arity, prog):
        key = repr(prog)
        if key in self.library:
            self.library[key]['uses'] += 1
        else:
            self.library[key] = dict(arity=arity, prog=prog, uses=1)
            self.new_since_sleep += 1

    # ------------------------------------------------------------ learning one function from the examples around it
    def _contexts(self, f, study, prims, funcs):
        out = []
        for tree, expected in study:
            nodes = list(self._nodes(tree))
            if sum(n.word == f and n.kind != 'P' for n in nodes) != 1:
                continue
            if any(n.kind != 'P' and n.word != f and n.word not in funcs for n in nodes) or any(n.kind == 'P' and n.word not in prims for n in nodes):
                continue
            target = next(n for n in nodes if n.word == f and n.kind != 'P')
            args = tuple(R.evaluate(k, prims, funcs, self.abstractions) for k in target.kids)
            out.append(dict(args=args, expected=expected, tree=tree, target=target.id, prims=prims, funcs=funcs))
        return out

    @staticmethod
    def _nodes(tree):
        yield tree
        for k in tree.kids:
            yield from Learner._nodes(k)

    def _final(self, c, local):
        return R.evaluate(c['tree'], c['prims'], c['funcs'], self.abstractions, override={c['target']: local})

    def _solve(self, arity, ctxs):
        if self.mode != 'fresh':
            for key, e in sorted(self.library.items(), key=lambda kv: -kv[1]['uses']):
                if e['arity'] != arity:
                    continue
                ok = True
                for c in ctxs:
                    self.work += 1
                    if self._final(c, R.ev(e['prog'], c['args'], self.abstractions)) != c['expected']:
                        ok = False
                        break
                if ok:
                    return e['prog'], 'lib'
        prog = self._enumerate(arity, ctxs)
        return prog, ('search' if prog is not None else 'failed')

    def _enumerate(self, arity, ctxs):
        n = len(ctxs)
        ops1 = [(k, f, None) for k, f in UNARY.items()]
        ops2 = [('cat', lambda a, b: a + b, None)]
        if self.mode == 'abs':
            for name, a in self.abstractions.items():
                fn = (lambda body: (lambda *xs: R.ev(body, xs, self.abstractions)))(a['body'])
                (ops1 if a['arity'] == 1 else ops2).append((name, fn, name))
        seen, by_size, start = set(), {1: []}, self.work
        for i in range(arity):
            vals = tuple(c['args'][i] for c in ctxs)
            if vals not in seen:
                seen.add(vals)
                by_size[1].append((R.A(i), vals))
        if self._check_new(by_size[1], ctxs):
            return self._found
        for s in range(2, MAX_SIZE + 1):
            new = []
            for name, fn, lib_name in ops1:
                for t, v in by_size.get(s - 1, []):
                    self.work += n
                    nv = tuple(fn(x) for x in v)
                    if nv not in seen:
                        seen.add(nv)
                        new.append((('lib', lib_name, t) if lib_name else (name, t), nv))
            for name, fn, lib_name in ops2:
                for sa in range(1, s - 1):
                    for ta, va in by_size.get(sa, []):
                        for tb, vb in by_size.get(s - 1 - sa, []):
                            self.work += n
                            nv = tuple(fn(x, y) for x, y in zip(va, vb))
                            if nv not in seen:
                                seen.add(nv)
                                new.append((('lib', lib_name, ta, tb) if lib_name else (name, ta, tb), nv))
                if self.work - start > SEARCH_CAP:
                    return None
            by_size[s] = new
            if self._check_new(new, ctxs):
                return self._found
            if self.work - start > SEARCH_CAP:
                return None
        return None

    def _check_new(self, entries, ctxs):
        for term, vals in entries:
            ok = True
            for c, local in zip(ctxs, vals):
                self.work += 1
                if self._final(c, local) != c['expected']:
                    ok = False
                    break
            if ok:
                self._found = term
                return True
        return False

    @staticmethod
    def _probe(arity, k=14):
        import itertools
        pool = [(), ('R',), ('G',), ('R', 'G'), ('B', 'Y', 'R'), ('Y', 'Y'), ('G', 'B', 'R', 'Y'), ('R', 'R', 'G')]
        return [tuple(c) for c in itertools.islice(itertools.product(pool, repeat=arity), 0, None, max(1, len(pool) ** arity // k))]

    def _same(self, prog, truth, arity):
        return all(R.ev(prog, a, self.abstractions) == R.ev(truth, a) for a in self._probe(arity))

    # ------------------------------------------------------------ the sleep decision (nothing schedules it)
    def after_episode(self, res):
        if self.mode != 'abs':
            return
        self.debt += res['energy'] / self.need_scale
        backlog = min(1.0, self.new_since_sleep / max(1, len(self.library)))
        need, want = self.debt, self.want_weight * backlog * (1.0 - res['novelty'])
        self.need_now, self.want_now = need, want
        if max(need, want) >= self.sleep_threshold:
            self.asleep, self.slept_for = True, 0
            self.sleeps += 1
            self.last_reason = 'need' if need >= want else 'want'

    def sleep_step(self):
        """one episode-slot of sleep: consolidate (find recurring parts and name them), pay for it, recover, maybe wake"""
        examined = sum(R.size(e['prog']) for e in self.library.values())
        cost = C_COMPRESS * examined
        self.compress()
        self.debt *= 0.4
        self.new_since_sleep = int(self.new_since_sleep * 0.5)
        self.slept_for += 1
        backlog = min(1.0, self.new_since_sleep / max(1, len(self.library)))
        if (self.debt < self.wake_level and backlog < self.wake_level) or self.slept_for >= self.sleep_max:
            self.asleep = False
        return cost

    def compress(self):
        for _ in range(6):
            progs = {k: e['prog'] for k, e in self.library.items()}
            cands = {}
            for p in progs.values():
                for sub in subterms(p):
                    if sub[0] != 'a' and R.size(sub) >= 3:
                        for pat in patterns(sub):
                            if 1 <= nvars(pat) <= 2:
                                cands[repr(pat)] = pat
            best, best_gain = None, 0
            for pat in cands.values():
                nv, gain, occ = nvars(pat), -R.size(pat), 0
                for p in progs.values():
                    for sub in subterms(p):
                        b = {}
                        if sub[0] != 'a' and match(pat, sub, b):
                            occ += 1
                            gain += R.size(sub) - (1 + sum(R.size(b[i]) for i in range(nv)))
                if occ >= 2 and gain > best_gain:
                    best, best_gain = pat, gain
            if best is None:
                return
            name, nv = f'abs{len(self.abstractions) + 1}', nvars(best)
            self.abstractions[name] = dict(arity=nv, body=best)
            for k in list(self.library):
                e = self.library.pop(k)
                e['prog'] = rewrite(e['prog'], best, name, nv)
                self.library[repr(e['prog'])] = e
