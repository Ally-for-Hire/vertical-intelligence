"""The rune universe: made-up words whose meanings must be induced from examples.

Format follows Lake & Baroni (2023, Nature, "Human-like systematic generalization through a meta-learning neural network"): each episode has 4 primitive words
(each a colour) and 3 function words (rules that combine things), the word-to-meaning assignment is re-drawn every episode, and the learner sees 14 study
examples and must answer 10 queries that need new compositions. Their grammar had fep (repeat three times), blicket (alternate), kiki (concatenate reversed).

What this adds (the published data has neither):
  * a hidden pool of FUNCTIONS built from a few latent BLOCKS (double, flip, reverse, pick-head, cut-tail), so we know the ground truth;
  * DOMAINS that share blocks to different degrees, including a control that shares none, so "do all problems relate?" can be tested.

Programs are tiny expression trees over atoms: ('a', i) argument i; ('cat', x, y) concatenate; ('rev', x) reverse; ('head', x) first item; ('tail', x) all but the first.
Instruction syntax is visible (each token is tagged P primitive, U unary postfix function, B binary infix function); only the MEANINGS are hidden.
Instructions parse left to right: expr := P | expr U | expr B P.
"""
import random

COLORS = ('R', 'G', 'B', 'Y')
PRIM_WORDS = ['dax', 'wif', 'lug', 'zup', 'mip', 'tof', 'neb', 'sab', 'rin', 'kel']
FUNC_WORDS = ['fep', 'blicket', 'kiki', 'zorb', 'mave', 'pliv', 'trun', 'glek', 'sorp', 'yune']


def A(i): return ('a', i)
def CAT(x, y): return ('cat', x, y)
def REV(x): return ('rev', x)
def HEAD(x): return ('head', x)
def TAIL(x): return ('tail', x)


def ev(t, args, lib=None):
    """evaluate a program on argument tuples"""
    k = t[0]
    if k == 'a':
        return args[t[1]]
    if k == 'v':
        return args[t[1]]
    if k == 'cat':
        return ev(t[1], args, lib) + ev(t[2], args, lib)
    if k == 'rev':
        return ev(t[1], args, lib)[::-1]
    if k == 'head':
        return ev(t[1], args, lib)[:1]
    if k == 'tail':
        return ev(t[1], args, lib)[1:]
    if k == 'lib':  # a learned abstraction: ('lib', name, arg terms...)
        body = lib[t[1]]['body']
        return ev(body, tuple(ev(x, args, lib) for x in t[2:]), lib)
    raise ValueError(k)


def size(t):
    return 1 + sum(size(c) for c in t[1:] if isinstance(c, tuple)) if t[0] in ('cat', 'rev', 'head', 'tail', 'lib') else 1


# ---- the hidden pool of functions. blocks: DBL(t)=cat(t,t), FLIP(x,y)=cat(y,x), REV(t)=rev(t), PICK(x,y)=cat(head x, head y), CUT(t)=tail(t)
DBL = lambda t: CAT(t, t)
FLIP = lambda x, y: CAT(y, x)
TRUE_FUNCS = {
    # name: (arity, program, latent blocks used)
    'twice':   (1, DBL(A(0)), {'DBL'}),
    'quad':    (1, DBL(DBL(A(0))), {'DBL'}),
    'swap':    (2, FLIP(A(0), A(1)), {'FLIP'}),
    'swapdbl': (2, DBL(FLIP(A(0), A(1))), {'DBL', 'FLIP'}),
    'dblswap': (2, FLIP(DBL(A(0)), A(1)), {'DBL', 'FLIP'}),
    'reverse': (1, REV(A(0)), {'REV'}),
    'rdbl':    (1, REV(DBL(A(0))), {'REV', 'DBL'}),
    'revcat':  (2, REV(CAT(A(0), A(1))), {'REV'}),
    'cut':     (1, TAIL(A(0)), {'CUT'}),
    'heads':   (2, CAT(HEAD(A(0)), HEAD(A(1))), {'PICK'}),
    'tailcat': (2, CAT(TAIL(A(0)), TAIL(A(1))), {'CUT'}),
}
DOMAINS = {
    'D1': ['twice', 'quad', 'swap'],        # blocks DBL, FLIP
    'D2': ['swapdbl', 'dblswap', 'quad'],   # NEW functions built only from D1's blocks, plus one function seen before
    'D3': ['reverse', 'rdbl', 'revcat'],    # a new block (REV) plus reused ones
    'D4': ['cut', 'heads', 'tailcat'],      # control: blocks (head, tail) shared with nothing above
}


class Node:
    __slots__ = ('kind', 'word', 'kids', 'id')

    def __init__(self, kind, word, kids=()):
        self.kind, self.word, self.kids, self.id = kind, word, tuple(kids), None


def parse(tokens):
    """tokens: [(word, kind)] with kind P / U / B  ->  tree. Left to right: expr := P | expr U | expr B P"""
    i, nid = 0, [0]

    def mk(kind, word, kids=()):
        n = Node(kind, word, kids)
        n.id = nid[0]
        nid[0] += 1
        return n
    cur = mk('P', tokens[0][0])
    i = 1
    while i < len(tokens):
        w, k = tokens[i]
        if k == 'U':
            cur = mk('U', w, [cur])
            i += 1
        elif k == 'B':
            cur = mk('B', w, [cur, mk('P', tokens[i + 1][0])])
            i += 2
        else:
            raise ValueError('bad instruction')
    return cur


def evaluate(node, prims, funcs, lib=None, override=None):
    """funcs: word -> (arity, program). override: {node id: value} replaces that node's result (used to learn a function from the examples around it)"""
    if override and node.id in override:
        return override[node.id]
    if node.kind == 'P':
        return prims[node.word]
    args = tuple(evaluate(k, prims, funcs, lib, override) for k in node.kids)
    return ev(funcs[node.word][1], args, lib)


def make_episode(rng, domain, n_study=14, n_query=10, max_len=30):
    prims = rng.sample(PRIM_WORDS, 4)
    fwords = rng.sample(FUNC_WORDS, 3)
    cols = list(COLORS)
    rng.shuffle(cols)
    prim_map = {w: (c,) for w, c in zip(prims, cols)}
    fnames = list(DOMAINS[domain])
    rng.shuffle(fnames)
    word_to_name = dict(zip(fwords, fnames))
    func_map = {w: (TRUE_FUNCS[n][0], TRUE_FUNCS[n][1]) for w, n in word_to_name.items()}
    kind = {w: ('U' if func_map[w][0] == 1 else 'B') for w in fwords}

    def rand_expr(ops, must=None):
        toks = [(rng.choice(prims), 'P')]
        for j in range(ops):
            f = must if (must and j == ops - 1) else rng.choice(fwords)
            toks.append((f, kind[f]))
            if kind[f] == 'B':
                toks.append((rng.choice(prims), 'P'))
        return toks

    def output(toks):
        return evaluate(parse(toks), prim_map, func_map)

    seen, study, query = set(), [], []

    def add(bucket, toks):
        key = tuple(toks)
        out = output(toks)
        if key in seen or not out or len(out) > max_len:
            return False
        seen.add(key)
        bucket.append((toks, out))
        return True

    for p in prims:                                   # each primitive on its own
        add(study, [(p, 'P')])
    for f in fwords:                                  # each function on simple arguments, twice
        got = 0
        while got < 2:
            got += add(study, rand_expr(1, must=f))
    while len(study) < n_study:                       # a few compositions
        add(study, rand_expr(2))
    while len(query) < n_query:                       # new, longer compositions
        add(query, rand_expr(rng.choice([2, 3, 4])))
    rng.shuffle(study)
    return dict(domain=domain, prims_true=prim_map, funcs_true=func_map, word_to_name=word_to_name, kind=kind, study=study, query=query)
