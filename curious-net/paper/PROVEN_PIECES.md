# Proven pieces: what the model may use (rule 'earn every component', author 2026-10-28)

The model under construction may use anything listed as USABLE here. To add or change a component, first prove it alone (registered test, fair baseline with equal tuning, manipulation check, own-goal check, fresh seeds; protocol s5), record it here with its audit status, then test any combination against its parts. Status is read through `paper/AUDIT_2026-10-28.md`: KEEP, or CAVEAT with the limit shown. Anything in the last table is NOT usable until redone. Update this file whenever a verdict changes.

## USABLE
| Piece | Evidence | Status | What it is good for | Limit that travels with it |
|---|---|---|---|---|
| Learned relevance gate (soft screen on inputs) | C30, C33, C34 | KEEP | small-data learners: attend to the inputs that matter; +0.035 +- 0.005 over a tuned soft screen | a learned per-input learning rate (Meta-SGD/ARD family), not evidence of causal structure; AUROC for true parents 0.56-0.73 |
| Replay at equal compute | C35 (B1/B1b) | KEEP | protects returning knowledge | costs new learning (about 5% worse on new regimes at equal compute) |
| Curiosity as learning progress | C40 (P3), X1.1 | KEEP | avoids the unlearnable-noise trap and the dark room | starves hard activities; slow to find a newly opened region (6x slower than chance in X1.1) |
| Optimistic start for untried options | X1.2 (P249-P252) | KEEP | finds a newly opened learnable region as fast as an oracle | derived fix on a toy bandit; does not rescue error-seeking |
| Viability margin / weakest-need value | X0.6, X0.7 | KEEP | designer-free graded warning at 2 needs | weaker at 4-6 needs and setting-sensitive (X0.8, X0.9) |
| Attend to every need (none ignored) | X0.7, X0.8, X0.9, X0.10, X1.3 | KEEP | robust in all five worlds: ignoring a need costs far more than any combination rule | toy worlds |
| Combine needs as unit-free level changes | X0.8, X0.9 | CAVEAT | sum of changes in levels; designer setpoints strong | a raw-unit sum is only as good as the units' fit to urgency; do not use raw units |
| Hidden need felt only as boredom, relief as reward | X0.10, X1.3 | KEEP (hostile reading recorded) | keeps an entity out of the dark room and the noise with no designed curiosity | the boredom signal was exact and noise-free; not yet shown with an estimated or lagged signal (X1.4) |
| Library law selection by BIC | C32, E9, E9c (C57) | CAVEAT | recovers a known law form from real data (Kepler 1.5, moons 1.5002) and rejects bad ones | the library contains the law by construction |
| Input-distance novelty (know the edge of what was taught) | C31 (E2) | CAVEAT | flags shifted inputs (AUROC 0.71-0.86) | 24 shifted worlds, one prior fit; own-uncertainty scores fail |
| Curriculum (easy to hard, self-paced) | C53 (B13) | KEEP | about 2.4x cheaper to criterion | toy tasks |
| Library learning of shared blocks | C43 (T3/T3b) | CAVEAT | transfer proportional to shared structure | the generator's blocks equal the learner's hypothesis class |
| Fair speed comparison of evolution and gradient descent | BENCH2 (P282-P287) | KEEP | on six small problems under a validation-based equal-interval stop, equal-size tuning and held-out seeds: tuned evolution is about 12.5x costlier than tuned Adam; default vs default 0.69; evolution unable on checker3 | evolution's tuning-seed cost is inflated 1.48x (use held-out numbers); Adam's default settings are poor on tiny problems; one dataset per problem; replaces the '0.51x / 9.4x' headline |
| Exact structure signal as the drive | X1.4 (BITS) | KEEP (hostile reading) | a reward equal to the exact bits of structure learned keeps the prediction-only entity alive and out of the dark room (0 deaths in 48 worlds) | needs the world to supply an exact bits count; vitality and boredom framing added nothing measurable; boredom noise sd 0.10 with a 10-step average starts to hurt (58% of worlds death-free) |
| Negative results worth keeping | C49 (locking weights fights relearning), C54/C55 (sleep is null unless learning saturates), C17 (randomness as mechanism null) | KEEP | what NOT to build in | |

## NOT USABLE until redone (VOID-REDO or relabelled, audit 2026-10-28)
Sleep timing rule (C56, P146-P150); motivation as a learning booster (C52, P126-P129); the 'confident misroutes are 40-46x more damaging' principle (CTX8) and what rests on it; time-context evidence (P183-P185, CTX12/13); emergent-optimizer speed ratios from series 3 ('0.51x / 9.4x') and the old benchmark that produced them (REPLACED by the BENCH2 numbers above); legacy C4, C8 (by construction), C2 (metric), C24/C25 (by construction); 'energy as regulariser' and 'few connections matter' as general claims (C41, C50, C22: only 'when the truth is sparse'); A3/T3 as statements about brains or learners (sensitivity toys).

## Candidate pieces, not yet earned (must be proven alone first)
Dependence detection beyond correlation; interventions to separate cause from correlation; functional-form search outside a fixed library; flagging an unexplained anomaly at a fixed false-alarm rate; inventing a missing term or constant; proposing a principle. See the proof ladder in the wiki page `vertical-intelligence/wiki/path-to-pre-quantum.md`.
