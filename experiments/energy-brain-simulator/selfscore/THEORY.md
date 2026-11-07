# How culling changes accuracy: theory and tests

User request (2026-10-26): "an exact formula for how culling networks indirectly affects accuracy independent of ALL factors ... rock solid and foundational, tied into as much research and testing as possible."

Code: `theory.py` (closed forms, finite-population intensity, abstract simulator), `density.py` (distribution operator), `price_run.py` / `price_report.py` / `price_crn.py` (real networks), `theory_check1-4.py` (tests). Data: `runs/price_and_or10.json` (168 runs, 10,080 generations).

## 0. Short answer

* One statement is **exact and holds for every population, network type, problem, cull fraction, scoring noise and population size** (Theorem 1). It says the cull changes accuracy **only through one number**, the selection differential S, and S is nonzero only to the extent the cull's score is correlated with true accuracy. Culling never touches accuracy directly; it is indirect.
* The size of S has an **exact formula given the population and the scoring procedure** (Theorem 2), and a **closed form** under one distributional assumption, S = rho * i(n,k) * sigma_z (Theorem 2'), whose error I measured (4% to 64% depending on regime, section 5).
* What happens over MANY generations (including why there is an interior optimum cull fraction) follows from how S, the variance sigma_z^2 and mutation interact (Theorem 3). That layer is approximate; I give where it works, where it fails, and why.

"Independent of all factors" can only be true of Theorem 1. Everything else depends on the factors **only through four measurable quantities** (n, k, sigma_z, rho) plus the mutation/inner-life terms, so the theory is factor-free in the sense that no problem- or architecture-specific constant appears.

## 1. The algorithm analysed
A population of n individuals has true accuracy z_i (the expected held-out accuracy of individual i). Each generation: (a) inner life changes each individual by g_i (own growth; independent of the cull); (b) the cull ranks individuals by a SCORE s_i (accuracy on a random batch of Be patterns, the same batch for everyone) and keeps the top k (cull fraction c = 1 - k/n); (c) the survivors are copied back up to n, each copy changed by a small random m (the "slight randomisation").

## 2. Theorem 1 (exact accounting identity; Price equation specialised)
Let w_j be the number of members of the next population descended from individual j after the cull (w_j = 0 for culled individuals, 1 + number of its copies for survivors), and Delta_k the change in accuracy of copy k relative to its parent. Then, with zbar the mean accuracy,

    zbar_next - zbar_before = T_inner + S_w + T_kids
    T_inner = mean(g)                                   (own growth, before the cull)
    S_w     = (1/n) sum_j w_j z_j - zbar                (weighted selection differential)
    T_kids  = (1/n) sum_k Delta_k                       (transmission: effect of the random change on copies)

Proof. The next population consists of the k survivors and n-k copies; the total accuracy is sum_j (w_j) z_j + sum_k Delta_k. Divide by n and subtract zbar. QED (it is the Price equation with the covariance term written as the weighted mean of survivor accuracy).

This uses no assumption about networks, problems, noise, normality or population size. **Test:** 10,080 generations from 168 real-network runs (population sizes 64 and 256, scoring batches 32/128/1024, cull fractions 25% to a single winner): the largest residual is **exactly 0.0**. It is also a correctness test of the implementation of the cull.

Consequence (the "indirect" part): the cull enters ONLY via S_w. If the score carries no information about z, then E[S_w] = 0 and culling does nothing to accuracy (observed: cull 0% stays at chance in every experiment; at 0% there are no survivors to select).

## 3. Theorem 2 (selection differential)
Exact (given the population): E[S | z_1..z_n] = E_batch[ mean of z over the k individuals with the highest batch score ] - zbar, where the expectation is over the cull's own scoring randomness. For continuous populations drawn from a density f and scoring noise e, an individual of accuracy z survives with probability

    r_n(z) = E_e[ BinCDF(k-1; n-1, Sbar(z+e)) ]        (it survives iff at most k-1 of the other n-1 outrank it)

and S = integral z r(z) f(z) (n/k) dz - zbar. **Common random numbers (measured effect):** all individuals are scored on the SAME batch, so their errors are positively correlated; this makes the ranking more accurate than independent noise would. With the real shared batch the exact conditional expectation matches measurement (standardised residual mean between -0.10 and +0.35, expected 0; E 0.0276 vs measured 0.0279 at 99% cull); modelling independent noise per individual is biased low by 0.5 to 3.9 standard deviations (`price_crn.py`).

### Theorem 2' (closed form, bivariate-normal case; breeder's equation with truncation selection)
If (z, s) are jointly normal with correlation rho, S = rho * i(n,k) * sigma_z with rho = sigma_z / sqrt(sigma_z^2 + sigma_e^2), and i(n,k) is the EXPECTED mean of the top k of n standard normal draws,

    i(n,k) = (n/k) * integral x phi(x) P(Bin(n-1, 1-Phi(x)) <= k-1) dx        (exact finite-population intensity)
    i -> phi(x_p)/p  as n -> infinity with p = k/n, x_p = Phi^-1(1-p)

(the second form overestimates a single winner out of 256 by 5%: 2.97 vs the exact 2.83, and brute force agrees with the exact value). i(n,k) increases monotonically with the cull fraction, so **one-step gain always increases with harder culling**; that does not mean long-run gain does (section 4).
**Tests:** abstract simulation, 112 conditions (cull 0 to 99%, noise 0 to 2 sigma, n = 64 and 256): worst relative error of S **2.4%**. Real networks (6,011 generations with S > 0.002): median relative error against the exact conditional value **23%** using only the noise model for rho, 4-6% at 25-50% cull rising to 29% at 90%, 64% at 99% and 58% for a single winner; slope against exact 1.02; correlation 0.94. (Errors at heavy culls come from non-normal accuracy distributions, common random numbers and bounded accuracy.)

## 4. Theorem 3 (many generations; why there is an optimum cull fraction)
Survivors of a truncation have a reduced variance (Bulmer effect); with scoring reliability rho the variance after selection is

    sigma_s^2 = rho^2 sigma^2 v(n,k) + (1 - rho^2) sigma^2 (k-1)/k          v(n,k) = expected variance of the top k of n standard normals (-> 1 - i(i - x_p))

and the copies add mutation variance, so (f = (n-k)/n, mutation mean mu_m and variance sigma_m^2)

    zbar' = zbar + g + rho i(n,k) sigma + f mu_m ,        sigma'^2 = sigma_s^2 + f sigma_m^2 + f (1-f) mu_m^2

Setting the variance at its fixed point gives the long-run gain per generation as a function of cull fraction. Two competing effects: i(n,k) rises with the cull (more selection per generation), but sigma falls with the cull (diversity is consumed faster than mutation rebuilds it), and rho is low for noisy scores, so a very hard cull selects mostly luck.
**Tests (abstract simulation, 200-300 repetitions, `theory_check4.py`):** without scoring noise the gain rises with the cull to the single winner in simulation AND theory (0.0508 vs 0.0506 at 99% cull). With noise the simulation has an INTERIOR optimum at 95% cull for noise sigma_e 0.05 and 0.1; the two-moment recursion predicts 98% and 95%. **Real networks (`price_report.py` section 4):** at generation 3 accuracy increases monotonically with the cull (0.669, 0.681, 0.781, 0.826, 0.838, 0.853, 0.859 for 25%..single winner at Be = 1024), and by generation 40 the optimum is interior (0.912, 0.922, 0.969, 0.978, 0.977, 0.935, 0.947): best at 75-95%, worse at 99% and for a single winner. This agrees with the earlier cull-sweep experiments (and_or10: best 50-95%).
**Where it fails:** (1) A population that is a few lineages of near-identical clones (heavy culling) is not an independent sample; the distribution operator of `density.py`, which is exact for independent individuals, overshoots by +51% (mutation spread 0.02), +25% (0.04), +11% (0.08) at 20 generations, shrinking as copies become more varied, which locates the cause in clonal non-independence (drift and coalescence). The two-moment recursion errs by about 5-30% over 40 generations (more at mild culls, e.g. 2.7x low at 25%). (2) Accuracy is bounded, so near the ceiling the unbounded-trait theory overstates gains. (3) The distribution of accuracy is not normal. **These are open: no exact multi-generation law that includes clonal structure is claimed.**

## 5. What is exact, what is approximate
| statement | status | evidence |
|---|---|---|
| Theorem 1 accounting identity | exact, no assumptions | residual 0.0 over 10,080 real generations |
| Theorem 2 exact conditional E[S] (shared batch) | exact in expectation | standardised residual mean ~0; sd 0.56-1.10 |
| i(n,k) exact finite-population intensity | exact | matches brute force |
| Theorem 2' S = rho i sigma | approximate (normality) | 2.4% abstract; 4-64% real, by cull level |
| Theorem 3 variance recursion and optimum | approximate | optimum location matched (95% / 95%, 98%); gain curve 5-30% |
| clonal/drift effects at heavy culls | not captured | density operator +11% to +51% |

## 6. Literature (bibliographic details checked by web search on 2026-10-26; the formulas above are my own derivations tested numerically, NOT verified against the primary texts)
* G. R. Price (1970), "Selection and covariance", Nature 227:520-521 (the covariance/Price equation; the form with a transmission term is from later treatments, e.g. Frank 1995 J. Theor. Biol. 175:373-388).
* Falconer and Mackay, breeder's equation R = h^2 S and truncation selection intensity (textbook; not retrieved).
* H. Muhlenbein and D. Schlierkamp-Voosen (1993), "Predictive models for the breeder genetic algorithm I", Evolutionary Computation 1(1):25-49: applies the response-to-selection equation to genetic algorithms. Later: Voigt, Muhlenbein, Schlierkamp-Voosen (1996) for non-normal fitness distributions.
* T. Blickle and L. Thiele (1995), "A comparison of selection schemes used in genetic algorithms", ETH Zurich TIK Report 11: selection intensity, selection variance and loss of diversity for truncation and other schemes.
* M. G. Bulmer (1971), "The effect of selection on genetic variability", American Naturalist 105:201-211 (variance reduction under selection).
* H. N. Nagaraja (1981), Ann. Inst. Statist. Math. 33:437-448 and P. M. Burrows (1975), Biometrics 31:125-133 (finite-sample selection differential); H. L. Harter (1961), Biometrika 48:151-159 (normal order statistics) (existence confirmed, content not retrieved).
* H.-G. Beyer, D. V. Arnold (e.g. GECCO 2003; FOGA 2001; CEC 2001): progress of (mu/mu, lambda) evolution strategies under fitness noise (existence confirmed; the claim that noise lowers effective selection intensity is derived here, not quoted).
* Not searched, listed as context only: Fisher's fundamental theorem; Robertson's secondary theorem (named in Frank 1995); Hill-Robertson interference; effective population size and genetic drift (Wright, Crow-Kimura).

## 7. Open items
Multi-generation law with clonal structure (lineage/branching-process closure); bounded-trait (ceiling) correction; extend the real-network tests to models with hidden units and stochastic hidden nodes (here: deterministic networks with no hidden units so that accuracy is exactly computable); other cull rules (tournament, rank-proportional) which change only the weights w_j, leaving Theorem 1 intact.
