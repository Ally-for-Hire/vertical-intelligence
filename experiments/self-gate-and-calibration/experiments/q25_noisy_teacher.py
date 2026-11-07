"""Q25: noisy teacher x student skepticism, same toy world as q24. The teacher's utterance is replaced by a random token with probability e (a false statement).
Students: credulous (assumes e=1e-6, i.e. treats every utterance as true) vs calibrated (knows e and uses it in the Bayes update) vs verifying (calibrated, plus when a token's top two
candidates are within a factor of 4 of each other it spends one own 'experiment': it requests a test scene made only of the top candidate concept's factor values and checks whether the
teacher's token is confirmed; the experiment is noise-free but counted as a cost). Reports mastery rate and interactions (teacher utterances + own experiments) over 100 students."""
import sys
sys.path.insert(0, ".")
import numpy as np
import q24_teacher_student as Q

NC, OFF, FACTORS = Q.NC, Q.OFF, Q.FACTORS


def run(teacher, student, e, rng, maxn=4000):
    perm = rng.permutation(NC)
    inv = np.argsort(perm)
    logP = np.zeros((NC, NC))
    rr = 0
    se = {"credulous": 1e-6, "calibrated": max(e, 1e-6), "verifying": max(e, 1e-6)}[student]
    cost = 0
    for n in range(1, maxn + 1):
        P = np.exp(logP - logP.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        if teacher == "adaptive":
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
                    free = [v for v in range(FACTORS[f]) if not cand[int(OFF[f] + v)]]
                    scene.append(int(rng.choice(free if free else list(range(FACTORS[f])))))
        else:
            scene = [int(rng.integers(k)) for k in FACTORS]
            if teacher == "random":
                tok = int(inv[Q.scene_concepts(scene)[int(rng.integers(len(FACTORS)))]])
            else:
                c = rr % NC
                rr += 1
                f_t = int(np.searchsorted(OFF, c, side="right") - 1)
                scene[f_t] = int(c - OFF[f_t])
                tok = int(inv[c])
        if rng.random() < e:
            tok = int(rng.integers(NC))
        present = np.zeros(NC, bool)
        present[Q.scene_concepts(scene)] = True
        logP[tok] += np.where(present, np.log(1 - se), np.log(se))
        cost += 1
        if student == "verifying":
            P = np.exp(logP - logP.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            for t in range(NC):
                top2 = np.sort(P[t])[-2:]
                if top2[1] < 4 * top2[0] + 1e-12 and top2[1] < 0.95:
                    c1 = int(np.argmax(P[t]))                   # own experiment: is concept c1 really what token t names? the world answers noise-free
                    truth_is = (c1 == perm[t])
                    logP[t, c1] += np.log(0.999) if truth_is else np.log(1e-6)
                    cost += 1
        if n % 5 == 0:
            P = np.exp(logP - logP.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            if np.all(P[np.arange(NC), perm] >= 0.95):
                return cost
    return None


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    print("teacher | student | error e | mastered % | median cost (utterances + own experiments)")
    for e in (0.0, 0.05, 0.1, 0.2):
        for teacher in ("random", "adaptive"):
            for student in ("credulous", "calibrated", "verifying"):
                res = [run(teacher, student, e, rng) for _ in range(100)]
                ok = [r for r in res if r is not None]
                print(f"{teacher:9s} {student:11s} e={e:4.2f}  mastered {100*len(ok)/len(res):5.1f}%  median {np.median(ok) if ok else float('nan'):7.1f}")
