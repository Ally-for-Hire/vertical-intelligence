"""Q24: tabula-rasa student + teacher in a grounded toy world (cross-situational word learning).
World: a scene is a tuple with one value for each of 4 factors (sizes 5, 4, 3, 3 -> 15 concepts); 15 arbitrary tokens, one per concept, assigned by a random permutation
the student does not know. An interaction: the teacher shows a scene and utters ONE token that refers to a concept present in the scene.
Student (starts with literally no word meanings): for every token a posterior over the 15 concepts, updated by Bayes with likelihood (1-e) if the concept is present in the scene
else e (e = teacher error rate, 0 in this file's noiseless arms uses a tiny floor). Mastery = every token's posterior on its true concept >= 0.95.
Arms: babble (negative control: tokens unrelated to the scene = no teacher), random (random scene, random mentioned concept), round-robin (scene random, mentioned concept cycles
through all concepts: a fixed curriculum), adaptive (machine teaching: the teacher reads the student's posterior, picks the most uncertain token and shows a scene that contains
its referent and as few of that token's remaining candidate concepts as possible). Reports interactions to mastery over 300 students per arm."""
import numpy as np

FACTORS = [5, 4, 3, 3]
OFF = np.concatenate([[0], np.cumsum(FACTORS)])
NC = int(OFF[-1])
EPS_FLOOR = 1e-6
MAXN = 3000


def scene_concepts(scene):
    return [int(OFF[f] + v) for f, v in enumerate(scene)]


def run(arm, rng, eps_teacher=0.0):
    perm = rng.permutation(NC)                    # token i refers to concept perm[i]
    inv = np.argsort(perm)                        # concept c is named by token inv[c]
    logP = np.zeros((NC, NC))                     # logP[token, concept], uniform start
    rr = 0
    for n in range(1, MAXN + 1):
        if arm == "adaptive":
            P = np.exp(logP - logP.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
            tok = int(np.argmax(H + 1e-9 * rng.random(NC)))
            target = perm[tok]
            cand = P[tok] > 0.02
            f_t = int(np.searchsorted(OFF, target, side="right") - 1)
            scene = []
            for f in range(len(FACTORS)):
                if f == f_t:
                    scene.append(int(target - OFF[f]))
                else:
                    vals = np.arange(FACTORS[f])
                    free = [v for v in vals if not cand[int(OFF[f] + v)]]
                    pool = free if free else list(vals)
                    scene.append(int(rng.choice(pool)))
            sc = scene_concepts(scene)
        else:
            scene = [int(rng.integers(k)) for k in FACTORS]
            sc = scene_concepts(scene)
            if arm == "babble":
                tok = int(rng.integers(NC))               # token unrelated to the scene
            elif arm == "random":
                tok = int(inv[sc[int(rng.integers(len(sc)))]])
            else:                                         # round-robin curriculum over concepts
                c = rr % NC
                rr += 1
                f_t = int(np.searchsorted(OFF, c, side="right") - 1)
                scene[f_t] = int(c - OFF[f_t])
                sc = scene_concepts(scene)
                tok = int(inv[c])
        if arm != "babble" and rng.random() < eps_teacher:
            tok = int(rng.integers(NC))                   # a wrong utterance (error)
        present = np.zeros(NC, bool)
        present[sc] = True
        e = max(eps_teacher, EPS_FLOOR) if eps_teacher > 0 else EPS_FLOOR
        logP[tok] += np.where(present, np.log(1 - e), np.log(e))
        if n % 5 == 0 or n < 5:
            P = np.exp(logP - logP.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            if np.all(P[np.arange(NC), perm] >= 0.95):
                return n
    return MAXN + 1


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("concepts", NC, "tokens", NC, "; information needed >= %.1f bits (log2 of the 15! possible token assignments / reduced by sharing = see text)" % np.log2(float(np.prod(np.arange(1, NC + 1)))))
    for arm in ("babble", "random", "round-robin", "adaptive"):
        res = np.array([run(arm, rng) for _ in range(300)])
        ok = res <= MAXN
        print(f"{arm:12s} mastered {ok.mean()*100:5.1f}%  interactions median {np.median(res[ok]) if ok.any() else float('nan'):6.1f}  IQR [{np.percentile(res[ok],25) if ok.any() else float('nan'):5.1f},{np.percentile(res[ok],75) if ok.any() else float('nan'):5.1f}]  mean {res[ok].mean() if ok.any() else float('nan'):6.1f}")
