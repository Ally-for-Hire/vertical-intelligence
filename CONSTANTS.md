# Constants register (generated from src/constants.py; run `python src/audit_constants.py` to re-check)

Every number the agent uses is declared here. **ARBITRARY = an open problem**: a free choice nobody has justified. The audit fails if any numeric literal in `src/` is not routed through the registry. **Status (iteration 13): the audit currently FAILS with 76 undeclared numeric literals, all in the self-learning-network line (worlds.py 40, selfnn.py 15, nn.py 9, seqnet.py 8, tfm.py 4): e.g. the 100-tick descriptor window, the 300-tick label window, the gate floor 1e-3, the prior network size and steps, the Bonferroni levels. They are unbacked choices and must be registered (queue R2).**

Counts: ARBITRARY 39, DERIVED 12, EXPERIMENT 2, NUMERICAL 5, REPORT 1, USER 2, WORLD 2


## ARBITRARY

| name | value | where | note |
|---|---|---|---|
| parent_cap | 4 | cbn | max parents per neuron. MDL already limits growth (table size doubles per parent); cap is a compute budget. TODO: remove, or measure effect of 3/4/5/6 |
| n_thresholds | 4 | cbn.Encoder | level bits per scalar. TODO: choose by MDL per variable or sweep |
| range_percentile | 1 | cbn.Encoder | thresholds spread over [p1,p99] of warm-up data |
| soft_width | 0.3 | cbn.Encoder | sigmoid width as fraction of bin spacing; expC showed no clear effect over {0.1,0.3,1.0} on 5-bit encoding (needs re-test on 8-bit) |
| warmup_ticks | 500 | run | random-action ticks used only to calibrate thresholds. TODO: calibrate online |
| warmup_throttle_bias | 0.8 | run | P(throttle) in warm-up so speed range is seen |
| grow_every | 500 | run/agent | ticks between structure searches. TODO: tie to sleep (searches happen only in sleep) or MDL-derived trigger |
| min_data_for_search | 400 | cbn.grow | no search before this many samples |
| search_window | 3000 | cbn.grow | recent samples used to screen parents. Replaced by 'since last sleep' once sleep is on |
| screen_top_m | 8 | cbn.grow | candidates per target passed from the marginal screen to the exact test (compute budget) |
| screen_min_gain_bits | 1.0 | cbn.grow | marginal gain below which a candidate is not tested exactly |
| random_floor | 0.05 | cbn.choose_action | epsilon-random action probability |
| lp_fast | 100 | cbn.observe | fast EMA horizon of loss (learning progress) |
| lp_slow | 1000 | cbn.observe | slow EMA horizon of loss (learning progress) |
| lp_floor | 0.001 | cbn.choose_action | learning-progress floor so no target gets zero weight |
| park_p | 0.03 | cbn.grow | variables with importance below this are parked (not searched) |
| park_min_updates | 6 | cbn.grow | importance updates before parking is allowed |
| park_recheck_every | 4 | cbn.grow | searches between re-checks of parked variables |
| task_hops | 3 | cbn.update_importance | hops of task-relevance propagation (superseded by TaskRanker) |
| task_hop_decay | 0.5 | cbn.update_importance | per-hop decay (superseded by TaskRanker) |
| hold_ticks | 10 | agent | an action is held this long (a timed experiment) |
| head_horizons | (5, 10) | cbn.FutureHeads | horizons of the future-score heads |
| n_outcome_classes | 4 | ranked | future score quantised into this many classes for ranking |
| rank_bins | 3 | ranked.TaskRanker | tercile levels per variable when ranking |
| rank_max_vars | 4 | ranked.TaskRanker | greedy steps |
| table_bins | {1: 9, 2: 9, 3: 5, 4: 3} | ranked.Table | bins per variable by table dimensionality (keeps cells ~ data) |
| table_prior_strength | 2 | ranked.Table | pseudo-samples shrinking a cell to the global mean |
| select_k | 3 | ranked.filtered_select | variables kept |
| redundancy_ratio | 0.2 | ranked.filtered_select | keep a variable only if it adds this share of its own marginal gain |
| exploit_depth | 8 | cbn.exploit_action | rollout depth (rollout exploit is unused by the best pipeline) |
| exploit_samples | 4 | cbn.exploit_action | sampled rollouts |
| eval_ticks | 500 | experiments | length of a driving evaluation |
| eval_starts | 3 | experiments | evaluation starts averaged |
| concept_unit_ops | ('AND', 'XOR') | core.sleep | operators allowed for hidden concept units; only 2-input units; no evidence these are the right family |
| parent_cap_compute | 4 | core | table size is 2^parents: compute budget, same as parent_cap |
| hidden_unit_inputs | 2 | core.sleep | concept units combine exactly 2 base inputs |
| sleep_scheduled_period | 500 | agent2 | ONLY for the scheduled-sleep ablation arm: sleep every this many ticks |
| expert_gain | 0.35 | agent2 | hand-written reference controller only (not the agent): lateral gain. It is a yardstick, not a result |
| expert_deadband | 0.05 | agent2 | hand-written reference controller only: steering deadband |

## USER

| name | value | where | note |
|---|---|---|---|
| p_init | 0.01 | importance | user: importance 'starts near 0'. Exact value is arbitrary but see sensitivity test (TODO expH) |
| sleep_only_structure | True | core | user: all permanent connections and importances are made in sleep from the day's data; awake only self-corrects |

## DERIVED

| name | value | where | note |
|---|---|---|---|
| kt_prior | 0.5 | cbn | Krichevsky-Trofimov estimator: add 1/2 to each count |
| param_cost_bits | 0.5 | cbn | MDL: each free parameter costs 1/2 log2 n bits |
| changepoint_cost | log2(n_eff) | core | declare a change in a neuron when accumulated excess surprise exceeds the code length of a changepoint position (MDL) |
| evidence_discount | 2^(-excess bits) | core | old evidence is worth 2^-S after S bits of excess surprise (likelihood ratio against the fresh model) |
| learning_rate | 1/(1+n_eff) | core | per-item rate = reciprocal of effective evidence; falls on correct predictions, rises after a detected change |
| sleep_min_new | log2(Bi)+0.5*log2(n) | core.sleep | fewest new samples that could pay for one parent under the MDL gate |
| sleep_flat_test | BIC: slope model vs constant | core.sleep | awake loss curve counts as flat when BIC prefers the constant model |
| sleep_backoff | T_at_last_sleep | core.sleep | if the last sleep changed nothing, wait for the history to double (amortises sleep cost to O(T)) |
| sleep_end | fixed point of total description length | core.sleep | each accepted step lowers D = code length + complexity, so the loop terminates by itself |
| trust_factor | g = 2^(-S), S <- max(0, S + CE - H(p)) | core | awake trust in permanent memory: two-sided CUSUM of excess surprise; rises again on correct predictions, no threshold |
| importance_rate | alpha_v = 1/(1+m_v), m_v <- (m_v+1) 2^(-excess) | core.sleep | importance treated as Gaussian with variance p(1-p)/(m+1); m counts consistent measurements; replaces alpha0/alpha_min/tau |
| importance_half_saturation | median positive contribution | core.sleep | r = G/(G+median(G>0)): scale-free squashing of bits-saved into [0,1) |

## NUMERICAL

| name | value | where | note |
|---|---|---|---|
| kl_clip | 1e-06 | cbn | probability clip inside KL / log |
| seed_offset_warmup | 77 | agent2 | decorrelates RNG streams |
| seed_offset_eval | 9 | agent2 | decorrelates RNG streams |
| seed_offset_policy | 5 | agent2 | decorrelates RNG streams |
| history_slack | 10 | agent2 | extra rows in history buffers |

## WORLD

| name | value | where | note |
|---|---|---|---|
| n_actions | 6 | cbn | steer {L,S,R} x throttle {0,1} |
| grid_bits | 49 | encode | 7x7 tile view |

## EXPERIMENT

| name | value | where | note |
|---|---|---|---|
| run_length | 16000 | agent2 | default number of awake ticks in a run; experiments state their own |
| demo_length | 6000 | agent2 | smoke-test length |

## REPORT

| name | value | where | note |
|---|---|---|---|
| report_every | 500 | agent2 | logging interval only; does not affect the agent |
