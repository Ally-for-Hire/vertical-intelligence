"""Registry of every number used by the vertical-intelligence (vi/) code: the shared world W0, the
learners and the experiments. Same status vocabulary as src/constants.py:
  DERIVED    follows from theory; no free choice
  MEASURED   backed by an experiment in this project (ref = results file)
  USER       specified by the user
  NUMERICAL  only guards against divide-by-zero / overflow
  WORLD      defines the environment, not the agent
  ARBITRARY  a free choice nobody has justified: an OPEN PROBLEM
Code reads values with C("name"); vi/audit.py fails on any literal not routed through here."""
REG = {}


def _r(name, value, status, where, note=""):
    REG[name] = dict(value=value, status=status, where=where, note=note)


# ---- W0: the shared world (WORLD: defines the environment, not the agent) ---------------------------------------
_r("w0_feat_dim", 8, "WORLD", "w0", "features per object")
_r("w0_effect_scale", 0.05, "WORLD", "w0", "size of an object's effect on an internal variable, per unit of u.x")
_r("w0_effect_clip", 0.15, "WORLD", "w0", "max |effect| of one object on one variable")
_r("w0_energy_decay", 0.004, "WORLD", "w0", "energy lost per tick (metabolism)")
_r("w0_integrity_heal", 0.001, "WORLD", "w0", "integrity regained per tick (slow repair)")
_r("w0_noise_frac", 0.2, "WORLD", "w0", "share of objects from the unlearnable noise class")
_r("w0_noise_sd", 0.08, "WORLD", "w0", "sd of a noise object's effect (mean 0, independent of its features)")
_r("w0_season_len", 5000, "WORLD", "w0", "ticks per season")
_r("w0_season_keep", 0.5, "WORLD", "w0", "correlation of a new season's effect directions with the old (shared structure)")
_r("w0_respawn", 0.5, "WORLD", "w0", "internal variables after a death")
_r("w0_tradeoff_corr", 0.0, "WORLD", "w0", "correlation of the energy and integrity effect directions; 0 => food is often harmful")

# ---- learner (shared by every arm; only the value signal differs) ----------------------------------------------
_r("hidden", 32, "ARBITRARY", "learner", "hidden units of the one-step value net")
_r("lr_default", 1e-3, "USER", "learner", "Adam default (method lesson: report default AND tuned)")
_r("lr_grid", (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), "ARBITRARY", "learner", "tuning grid (per arm, own objective); extended down after the first grid picked its lowest value")
_r("adam_b1", 0.9, "DERIVED", "learner", "Adam standard")
_r("adam_b2", 0.999, "DERIVED", "learner", "Adam standard")
_r("adam_eps", 1e-8, "NUMERICAL", "learner", "Adam epsilon")
_r("epsilon", 0.05, "ARBITRARY", "learner", "random-action probability, identical in every arm")
_r("init_scale", 0.5, "ARBITRARY", "learner", "init sd multiplier (1/sqrt(fan_in) * this)")

# ---- value arms (X0.1) ----------------------------------------------------------------------------------------
_r("setpoint", (0.8, 0.8), "ARBITRARY", "x0_1 HOMEO", "designer setpoints for energy, integrity. X0.6 removes them (learned setpoints)")
_r("drive_power", 2, "ARBITRARY", "x0_1 HOMEO", "Euclidean drive D = sqrt(sum |h* - h|^2). Relabelled from DERIVED (2026-10-27, CORRECTION): Keramati-Gutkin's drive has two free exponents; Euclidean is one special case")

# ---- experiment X0.1 --------------------------------------------------------------------------------------------
_r("x01_ticks", 15000, "ARBITRARY", "x0_1", "lifetime ticks per run (3 seasons)")
_r("x01_worlds", 48, "USER", "x0_1", "protocol default: 48 worlds")
_r("x01_seed0", 70000, "USER", "x0_1", "fresh seed block for X0.1 (never used before in curious-net)")
_r("x01_tune_seed0", 69000, "USER", "x0_1", "separate seed block for B0's learning-rate tuning")
_r("x01_tune_worlds", 8, "ARBITRARY", "x0_1", "worlds used to tune B0's learning rate")
_r("x01_novel", 400, "ARBITRARY", "x0_1", "novel test objects for the valence test")
_r("x01_window", 1000, "ARBITRARY", "x0_1", "ticks after a season swap used for the recovery metric")
_r("per_10k", 1e4, "NUMERICAL", "x0_1", "reporting unit: deaths per 10,000 ticks")
_r("ci_lo", 2.5, "DERIVED", "stats", "95% CI lower percentile")
_r("ci_hi", 97.5, "DERIVED", "stats", "95% CI upper percentile")
_r("boot", 2000, "ARBITRARY", "stats", "bootstrap resamples for the 95% CI")
_r("x01_state_starving", (0.1, 0.9), "ARBITRARY", "x0_1", "probe state for the need test: low energy, healthy")
_r("x01_state_sated", (0.95, 0.9), "ARBITRARY", "x0_1", "probe state for the need test: full, healthy")
_r("x01_test_offset", 500000, "NUMERICAL", "x0_1", "seed offset so novel test objects never coincide with a world stream")
_r("x01_pilot_seed", 68000, "USER", "x0_1", "pilot seed (exploratory, outside the scored and tuning blocks)")
_r("win_below_normal", 0x4000, "NUMERICAL", "x0_1", "Windows BELOW_NORMAL_PRIORITY_CLASS (compute rule: low priority)")

# ---- experiment X0.1b (multi-step value; tests the myopia diagnosis of X0.1 P201/P202) ---------------------------
_r("gamma_grid", (0.5, 0.9, 0.99), "ARBITRARY", "x0_1b", "discount tuning grid (per arm, own objective); gamma = 0 is the X0.1 one-step control")
_r("gamma_default", 0.99, "ARBITRARY", "x0_1b", "standard RL default discount, used in the 'default' setting")
_r("x01b_seed0", 71000, "USER", "x0_1b", "fresh seed block for X0.1b scored run")
_r("x01b_tune_seed0", 72000, "USER", "x0_1b", "seed block for X0.1b tuning (lr x gamma per arm)")

# ---- experiment X0.6 (no designer setpoints; exact survival oracle; state-aware valence) --------------------------
_r("x06_seed0", 74000, "USER", "x0_6", "fresh seed block for the X0.6 scored run")
_r("x06_tune_seed0", 75000, "USER", "x0_6", "seed block for X0.6 tuning")
_r("x06_smoke_seed", 76000, "USER", "x0_6", "seed for the bug-check smoke run (not used to set any prediction)")
_r("oracle_grid", 41, "ARBITRARY", "oracle", "grid points per internal variable for the exact DP")
_r("oracle_samples", 600, "ARBITRARY", "oracle", "effect samples for the expectation in the DP")
_r("oracle_gamma", 0.99, "ARBITRARY", "oracle", "discount of the oracle's survival value")
_r("oracle_tol", 1e-4, "NUMERICAL", "oracle", "value-iteration convergence tolerance")
_r("oracle_maxit", 3000, "NUMERICAL", "oracle", "value-iteration iteration cap")
_r("oracle_seed", 4242, "NUMERICAL", "oracle", "seed of the effect sample in the DP")
_r("x06_indiff", 1e-4, "ARBITRARY", "x0_6", "oracle indifference: |Q_eat - Q_skip| below this is not counted in the agreement metric")
_r("x06_states", 200, "ARBITRARY", "x0_6", "states sampled uniformly over the viable range for the state-aware valence probe")
_r("x06_objs", 10, "ARBITRARY", "x0_6", "novel objects per sampled state")
_r("v_hidden", 16, "ARBITRARY", "x0_6 VIAB", "hidden units of the learned viability net V(h)")
_r("lr_v_grid", (3e-3, 1e-2, 3e-2), "ARBITRARY", "x0_6 VIAB", "tuning grid for the viability net's learning rate")
_r("lr_v_default", 1e-2, "ARBITRARY", "x0_6 VIAB", "default viability-net learning rate (default setting)")
_r("gamma_v", 0.99, "ARBITRARY", "x0_6 VIAB", "discount of the learned viability V(h) = -E[gamma^(time to death)]")
_r("vgrid", 11, "ARBITRARY", "x0_6", "grid points per variable for comparing the learned viability with the oracle")
_r("vgrid_lo", 0.05, "ARBITRARY", "x0_6", "lower edge of that comparison grid")
_r("vgrid_hi", 0.95, "ARBITRARY", "x0_6", "upper edge of that comparison grid")
_r("n_danger_min", 30, "ARBITRARY", "x0_6", "manipulation check: median informative probes per world must be at least this")
_r("vcorr_min", 0.5, "ARBITRARY", "x0_6", "manipulation check: median Pearson(learned V, oracle V) must exceed this")

# ---- experiment X0.7 (weakest-link margin drive; orthogonal world vs trade-off world) ----------------------------
_r("x07_rho", -0.7, "WORLD", "x0_7", "energy/integrity effect-direction correlation in the trade-off world (negative = food that helps one variable harms the other)")
_r("x07_seed0", 77000, "USER", "x0_7", "fresh seed block for the X0.7 scored run")
_r("x07_tune_seed0", 78000, "USER", "x0_7", "seed block for X0.7 tuning")
_r("x07_check_seed", 79000, "USER", "x0_7", "seed for the world-feasibility pre-check (oracle and random only; no learner)")
_r("x07_feasible_frac", 0.5, "ARBITRARY", "x0_7", "manipulation check: ORACLE deaths must be below this fraction of RANDOM deaths in each world")


def C(name):
    return REG[name]["value"]


def problems():
    return [(n, e) for n, e in REG.items() if e["status"] == "ARBITRARY"]
