# Neuron-block research: handoff

Paste this README into the new code chat first, then continue from "Next steps".

## Project vision (from Rose/Ally)
Build a brain-like system that converts any problem into its own representation, solves it from prior context, and evolves. Inspired by Internal Family Systems (parts, a coordinating Self, burdens), Society of Mind, Adaptive Resonance Theory, and predictive processing. Goal: research, simulate, document, test; build reusable architecture so questions get answered by running experiments, not by guessing.

## Design principles stated so far
1. Everything is measured in time from A to B (microseconds). Instant thought = 0 us, average thought ~100 us.
2. Neurons live on a physical graph and can move. Base instinct = EXPANSION. Other neurons are combinations of batteries (store/release activation), expansion ability, etc.
3. The brain condenses itself toward a compact, efficient shape. Fitness = best result per unit of effort (size and time are soft costs, not a hard deadline).
4. The brain should ALWAYS attempt to expand. Criterion for expanding: decisions that satisfy the most neurons.
5. Neurons become progressively harder to affect as time goes on (hardening).
6. Outside sensor signals are trusted as fact with decaying relevance; internal signals are biased data, the system treats itself as biased data too and seeks to minimize uncertainty.
7. Submind spawning should not happen if it is negative for the whole; the "whole" should serve a goal bigger than self-preservation (open question; candidates: integration, Self-led coherence).
8. Reward for time on a problem only while uncertainty is dropping (to avoid stalling).

## Files
- neuroblock.py: base simulator. Brain class (2D positions, tree edges, transit time = length * 40 us, spawn on unexplained input, condense = move/merge/prune), run() harness, summarize().
- brain2.py: Brain2 adds rigid (learning rate and movement shrink with count/age) and expand (each cycle propose a specialist at a poorly-explained cluster of recent experience; keep if total fit gain is large enough).
- experiments2.py / results2.json: 4 variants x fast/best selection x 2 tasks x 2 drift modes, 8 seeds, plus hardening-strength sweep.
- experiments3.py: condensing ablation and expansion churn check.

## Results (8 seeds per condition; toy Gaussian-cluster task, so not evidence about real brains)
1. Condensing: no condensing ~94 us mean thought; physical moving ~31 us (edge length 1.0 -> 0.4); merge+prune cut neurons 27 -> 11 with accuracy 98.3% -> 99.5%. Moving buys speed, merge/prune buys compactness.
2. Selection rule: firing the fastest matching neuron gave 84% accuracy on the hard task; firing the best-fitting gave 97% for only ~7 us more. Speed-first selection let shallow neurons steal inputs.
3. Hardening: roughly halves seed-to-seed accuracy variance and helps slightly after drift, but thoughts get ~15-40% slower and early hard hardening (c0=3) made the brain grow to 38 neurons vs 21 (stiff neurons cannot track shifted concepts). Gentle hardening (c0 ~30-300) is the better balance.
4. Expansion drive is currently inert: ~25 accepted expansions per run but net neuron count barely changes, because merge/prune remove them and the same ones are re-proposed (without merge/prune only ~5 get accepted). Accuracy was near ceiling so there was little to gain.

## Caveats
- Differences under ~2 accuracy points are within noise at 8 seeds.
- The ~30 us floor is partly set by the min node spacing parameter (min_sep=0.3).
- 40 us per unit length and all tasks are assumptions.
- Why hardening slows thoughts is unverified (guess: layouts freeze before finishing compaction).

## Next steps
1. Give expansion a real job: a harder task where existing categories genuinely cannot explain input; protect newborn neurons from merge/prune for a grace period.
2. Replace the tree with a general graph (shortcut edges, batteries as delayed-fire nodes).
3. Add the Self-style coordinator and the "negative for the whole" check on spawning.
4. Add sensor-trust decay and a self-as-biased-data loop (predict own next error).
5. Add logging (births, deaths, vigilance, reward curves) so experiments stay comparable.
