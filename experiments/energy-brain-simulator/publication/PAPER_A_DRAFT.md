# How truncation culling changes accuracy in evolving neural populations: an exact accounting, a shared-batch selection law, and where the closed form fails

DRAFT v0.1 (2026-10-26). Sections marked [TODO] need experiments from `EXPERIMENT_PLAN.md` E1 before submission. Numbers below are real and come from the files cited.

## Abstract
Evolving populations of neural networks are commonly culled by performance, yet the effect of the cull fraction on accuracy is rarely quantified. We give a three-layer account. (1) An exact accounting identity (a Price-equation specialisation) shows that culling changes mean accuracy only through a weighted selection differential, plus the networks' own change and the effect of the random change applied to copies; we verify it with zero residual on 10,080 generations of 168 real runs. (2) The expected selection differential has an exact form given the population and the scoring procedure; when all individuals are scored on the same random batch their scoring errors are correlated, which sharpens selection, and an independent-noise model is biased low by 0.5-3.9 standard deviations. (3) A closed form S = rho * i(n,k) * sigma, using the exact finite-population selection intensity i(n,k), matches Monte Carlo to 2.4% in the Gaussian case and a median 4-29% on real networks at culls up to 90%, and a two-moment recursion predicts an interior optimum cull (75-95%) that real networks also show; a distribution-level operator that is exact for independent individuals overshoots by +11% to +51% when the population is a few lineages of clones, locating a limit of the breeder's equation for culled neural populations. [TODO: replicate with hidden units, stochastic nodes, several problems, and tournament selection.]

## 1. Introduction
Truncation selection and the breeder's equation R = h^2 S are classical (Falconer; Mühlenbein & Schlierkamp-Voosen 1993 applied them to the breeder genetic algorithm; Blickle & Thiele 1995 analysed selection intensity and loss of diversity for truncation and other schemes). The Price equation (Price 1970) decomposes change into selection and transmission; it has been applied to evolutionary algorithms before (a GECCO 2005 paper; Altenberg; Langdon & Poli). We do not claim the identity is new. Our contribution is a measured, layered account for culled neural populations: which parts are exact, what the realistic error of the usual closed form is, and what breaks.

## 2. Setting and definitions
n individuals with true accuracy z_i; each generation: (a) inner change g_i; (b) rank by a noisy score s_i (accuracy on one random batch of B patterns, shared by all individuals); keep top k (cull fraction c = 1 - k/n); (c) refill with copies carrying a small random change m. Notation and proofs: `selfscore/THEORY.md`.

## 3. Results
### 3.1 Exact accounting (Theorem 1)
zbar' - zbar = T_inner + S_w + T_kids. Measured residual: 0.0 over 10,080 generations (`price_report.py`; populations 64 and 256, B in {32,128,1024}, cull 25% to single winner; deterministic networks without hidden units on and_or10, so z is exactly computable).
### 3.2 Exact shared-batch selection differential (Theorem 2)
With the real shared batch the exact conditional expectation matches measurement (standardised residual mean -0.10 to +0.35; E 0.0276 vs measured 0.0279 at 99% cull); an independent-noise version is biased low (`price_crn.py`).
### 3.3 Closed form and its error (Theorem 2')
Table: abstract worst error 2.4% over 112 conditions; real networks median error 23% overall; by cull level 6.4%, 4.3%, 14.7%, 28.7%, 38.0%, 63.5%, 58.1% for 25%, 50%, 75%, 90%, 95%, 99%, single winner.
### 3.4 Many generations and the interior optimum (Theorem 3)
Simulation: with scoring noise the best cull is 95% (two noise levels); two-moment theory predicts 98% and 95%. Real networks (generation 40, B=1024): accuracy 0.912, 0.922, 0.969, 0.978, 0.977, 0.935, 0.947 for 25%, 50%, 75%, 90%, 95%, 99%, single winner.
### 3.5 Where it fails
Clonal populations: distribution operator error +51% / +25% / +11% at mutation spread 0.02 / 0.04 / 0.08 (`theory_check3.py`).

## 4. Limitations (must stay in the paper)
Deterministic no-hidden-unit networks on one problem for the real-network validation; toy scale; accuracy bounded (ceiling effects not modelled); no theory for clonal structure; novelty of the shared-batch finding unverified against the simulation-optimisation literature (common random numbers, ranking and selection). [TODO E1]

## 5. Reproducibility
`python publication/reproduce.py` re-checks the identity, the finite-population intensity, the closed-form error and the hand-built-network engine tests in a few minutes. Seeds, code, data generators and logs are in the repository; randomness will be made stateless and block-keyed (E0) so runs are bit-repeatable across machines.

## References (existence confirmed by web search 2026-10-26; formulas here are our derivations, NOT checked against the primary texts)
* G. R. Price (1970), Selection and covariance, Nature 227:520-521.
* H. Mühlenbein, D. Schlierkamp-Voosen (1993), Predictive models for the breeder genetic algorithm I, Evolutionary Computation 1(1):25-49.
* T. Blickle, L. Thiele (1995), A comparison of selection schemes used in genetic algorithms, ETH Zurich TIK Report 11.
* M. G. Bulmer (1971), The effect of selection on genetic variability, American Naturalist 105:201-211.
* H. N. Nagaraja (1981), Ann. Inst. Statist. Math. 33:437-448; P. M. Burrows (1975), Biometrics 31:125-133; H. L. Harter (1961), Biometrika 48:151-159.
* H.-G. Beyer, D. V. Arnold, noisy (mu/mu, lambda)-ES progress (CEC 2001, FOGA 2001, GECCO 2003).
* GECCO 2005 paper applying the Price equation to evolutionary algorithms (listed at gpbib.cs.ucl.ac.uk/gecco2005/docs/p1371.pdf; authors to be confirmed).
* S. A. Frank (1995), George Price's contributions to evolutionary genetics, J. Theor. Biol. 175:373-388.
