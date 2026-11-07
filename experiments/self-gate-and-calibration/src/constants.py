"""Single registry of every number the AGENT uses. Status vocabulary:
  DERIVED    follows from theory (MDL / KT / Bayes); no free choice
  MEASURED   backed by an experiment in this project (ref = results file)
  USER       specified by the user's brief
  NUMERICAL  only guards against divide-by-zero / overflow
  WORLD      defines the environment, not the agent (env.py)
  ARBITRARY  a free choice nobody has justified: an OPEN PROBLEM. Each must either be replaced by a derived rule or backed by a sweep.
Code reads values with C("name"); the audit (audit_constants.py) fails on any literal not routed through here."""
REG = {}


def _r(name, value, status, where, note=""):
    REG[name] = dict(value=value, status=status, where=where, note=note)


# ---- learner (cbn.py) --------------------------------------------------------------------------------------------
_r("kt_prior", 0.5, "DERIVED", "cbn", "Krichevsky-Trofimov estimator: add 1/2 to each count")
_r("param_cost_bits", 0.5, "DERIVED", "cbn", "MDL: each free parameter costs 1/2 log2 n bits")
_r("kl_clip", 1e-6, "NUMERICAL", "cbn", "probability clip inside KL / log")
_r("parent_cap", 4, "ARBITRARY", "cbn", "max parents per neuron. MDL already limits growth (table size doubles per parent); cap is a compute budget. TODO: remove, or measure effect of 3/4/5/6")
_r("n_thresholds", 4, "ARBITRARY", "cbn.Encoder", "level bits per scalar. TODO: choose by MDL per variable or sweep")
_r("range_percentile", 1, "ARBITRARY", "cbn.Encoder", "thresholds spread over [p1,p99] of warm-up data")
_r("soft_width", 0.3, "ARBITRARY", "cbn.Encoder", "sigmoid width as fraction of bin spacing; expC showed no clear effect over {0.1,0.3,1.0} on 5-bit encoding (needs re-test on 8-bit)")
_r("warmup_ticks", 500, "ARBITRARY", "run", "random-action ticks used only to calibrate thresholds. TODO: calibrate online")
_r("warmup_throttle_bias", 0.8, "ARBITRARY", "run", "P(throttle) in warm-up so speed range is seen")
_r("grow_every", 500, "ARBITRARY", "run/agent", "ticks between structure searches. TODO: tie to sleep (searches happen only in sleep) or MDL-derived trigger")
_r("min_data_for_search", 400, "ARBITRARY", "cbn.grow", "no search before this many samples")
_r("search_window", 3000, "ARBITRARY", "cbn.grow", "recent samples used to screen parents. Replaced by 'since last sleep' once sleep is on")
_r("screen_top_m", 8, "ARBITRARY", "cbn.grow", "candidates per target passed from the marginal screen to the exact test (compute budget)")
_r("screen_min_gain_bits", 1.0, "ARBITRARY", "cbn.grow", "marginal gain below which a candidate is not tested exactly")
_r("random_floor", 0.05, "ARBITRARY", "cbn.choose_action", "epsilon-random action probability")
_r("lp_fast", 100, "ARBITRARY", "cbn.observe", "fast EMA horizon of loss (learning progress)")
_r("lp_slow", 1000, "ARBITRARY", "cbn.observe", "slow EMA horizon of loss (learning progress)")
_r("lp_floor", 1e-3, "ARBITRARY", "cbn.choose_action", "learning-progress floor so no target gets zero weight")
_r("n_actions", 6, "WORLD", "cbn", "steer {L,S,R} x throttle {0,1}")
# ---- importance (cbn.update_importance / ranked.TaskRanker) ----------------------------------------------------------
_r("p_init", 0.01, "USER", "importance", "user: importance 'starts near 0'. Exact value is arbitrary but see sensitivity test (TODO expH)")
_r("park_p", 0.03, "ARBITRARY", "cbn.grow", "variables with importance below this are parked (not searched)")
_r("park_min_updates", 6, "ARBITRARY", "cbn.grow", "importance updates before parking is allowed")
_r("park_recheck_every", 4, "ARBITRARY", "cbn.grow", "searches between re-checks of parked variables")
_r("task_hops", 3, "ARBITRARY", "cbn.update_importance", "hops of task-relevance propagation (superseded by TaskRanker)")
_r("task_hop_decay", 0.5, "ARBITRARY", "cbn.update_importance", "per-hop decay (superseded by TaskRanker)")
# ---- experiments / policy -----------------------------------------------------------------------------------------
_r("hold_ticks", 10, "ARBITRARY", "agent", "an action is held this long (a timed experiment)")
_r("head_horizons", (5, 10), "ARBITRARY", "cbn.FutureHeads", "horizons of the future-score heads")
_r("n_outcome_classes", 4, "ARBITRARY", "ranked", "future score quantised into this many classes for ranking")
_r("rank_bins", 3, "ARBITRARY", "ranked.TaskRanker", "tercile levels per variable when ranking")
_r("rank_max_vars", 4, "ARBITRARY", "ranked.TaskRanker", "greedy steps")
_r("table_bins", {1: 9, 2: 9, 3: 5, 4: 3}, "ARBITRARY", "ranked.Table", "bins per variable by table dimensionality (keeps cells ~ data)")
_r("table_prior_strength", 2, "ARBITRARY", "ranked.Table", "pseudo-samples shrinking a cell to the global mean")
_r("select_k", 3, "ARBITRARY", "ranked.filtered_select", "variables kept")
_r("redundancy_ratio", 0.2, "ARBITRARY", "ranked.filtered_select", "keep a variable only if it adds this share of its own marginal gain")
_r("exploit_depth", 8, "ARBITRARY", "cbn.exploit_action", "rollout depth (rollout exploit is unused by the best pipeline)")
_r("exploit_samples", 4, "ARBITRARY", "cbn.exploit_action", "sampled rollouts")
_r("eval_ticks", 500, "ARBITRARY", "experiments", "length of a driving evaluation")
_r("eval_starts", 3, "ARBITRARY", "experiments", "evaluation starts averaged")


def C(name):
    return REG[name]["value"]


def problems():
    return [(n, e) for n, e in REG.items() if e["status"] == "ARBITRARY"]


def used_summary():
    from collections import Counter
    return dict(Counter(e["status"] for e in REG.values()))


# ---- v2 additions --------------------------------------------------------------------------------------------------
_r("changepoint_cost", "log2(n_eff)", "DERIVED", "core", "declare a change in a neuron when accumulated excess surprise exceeds the code length of a changepoint position (MDL)")
_r("evidence_discount", "2^(-excess bits)", "DERIVED", "core", "old evidence is worth 2^-S after S bits of excess surprise (likelihood ratio against the fresh model)")
_r("learning_rate", "1/(1+n_eff)", "DERIVED", "core", "per-item rate = reciprocal of effective evidence; falls on correct predictions, rises after a detected change")
_r("sleep_min_new", "log2(Bi)+0.5*log2(n)", "DERIVED", "core.sleep", "fewest new samples that could pay for one parent under the MDL gate")
_r("sleep_flat_test", "BIC: slope model vs constant", "DERIVED", "core.sleep", "awake loss curve counts as flat when BIC prefers the constant model")
_r("sleep_backoff", "T_at_last_sleep", "DERIVED", "core.sleep", "if the last sleep changed nothing, wait for the history to double (amortises sleep cost to O(T))")
_r("sleep_end", "fixed point of total description length", "DERIVED", "core.sleep", "each accepted step lowers D = code length + complexity, so the loop terminates by itself")
_r("concept_unit_ops", ("AND", "XOR"), "ARBITRARY", "core.sleep", "operators allowed for hidden concept units; only 2-input units; no evidence these are the right family")
_r("parent_cap_compute", 4, "ARBITRARY", "core", "table size is 2^parents: compute budget, same as parent_cap")
_r("trust_factor", "g = 2^(-S), S <- max(0, S + CE - H(p))", "DERIVED", "core", "awake trust in permanent memory: two-sided CUSUM of excess surprise; rises again on correct predictions, no threshold")
_r("importance_rate", "alpha_v = 1/(1+m_v), m_v <- (m_v+1) 2^(-excess)", "DERIVED", "core.sleep", "importance treated as Gaussian with variance p(1-p)/(m+1); m counts consistent measurements; replaces alpha0/alpha_min/tau")
_r("sleep_only_structure", True, "USER", "core", "user: all permanent connections and importances are made in sleep from the day's data; awake only self-corrects")
_r("hidden_unit_inputs", 2, "ARBITRARY", "core.sleep", "concept units combine exactly 2 base inputs")
_r("sleep_scheduled_period", 500, "ARBITRARY", "agent2", "ONLY for the scheduled-sleep ablation arm: sleep every this many ticks")
_r("grid_bits", 49, "WORLD", "encode", "7x7 tile view")
_r("importance_half_saturation", "median positive contribution", "DERIVED", "core.sleep", "r = G/(G+median(G>0)): scale-free squashing of bits-saved into [0,1)")
_r("report_every", 500, "REPORT", "agent2", "logging interval only; does not affect the agent")
_r("seed_offset_warmup", 77, "NUMERICAL", "agent2", "decorrelates RNG streams")
_r("seed_offset_eval", 9, "NUMERICAL", "agent2", "decorrelates RNG streams")
_r("seed_offset_policy", 5, "NUMERICAL", "agent2", "decorrelates RNG streams")
_r("history_slack", 10, "NUMERICAL", "agent2", "extra rows in history buffers")
_r("expert_gain", 0.35, "ARBITRARY", "agent2", "hand-written reference controller only (not the agent): lateral gain. It is a yardstick, not a result")
_r("expert_deadband", 0.05, "ARBITRARY", "agent2", "hand-written reference controller only: steering deadband")
_r("run_length", 16000, "EXPERIMENT", "agent2", "default number of awake ticks in a run; experiments state their own")
_r("demo_length", 6000, "EXPERIMENT", "agent2", "smoke-test length")
_r("registry_capacity", 2048, "NUMERICAL", "core2", "buffer size for the global name registry; not a modelling choice")
_r("variant_buffer", 6, "ARBITRARY", "core2", "max stored regimes (variants) per neuron; a fork is blocked when full (counted as fork_blocked). TODO: derive or measure how many are ever needed")
_r("problems_py", "world definitions of the curriculum problems", "WORLD", "problems", "constants inside problems.py define each problem's physics, not the learner")
_r("first_window", 500, "ARBITRARY", "lifelong", "ticks at the start of a problem over which forward transfer and trust are measured (reporting window)")
_r("mlp_replay_buffer", 2000, "ARBITRARY", "baselines2", "BASELINE-only hyperparameter: reservoir size for the replay MLP")
_r("maint_per_tick", 4, "ARBITRARY", "core3", "neurons examined by incremental maintenance each tick (compute budget per tick)")
_r("pair_candidates", 4, "ARBITRARY", "core3", "single candidates paired with each action bit for joint (action x variable) shadow tables")
_r("fork_patience", "mass doubling", "DERIVED", "core3", "an indistinguishable fresh table is absorbed only after its evidence has doubled (sequential-test doubling trick)")
