---
title: Vertical Intelligence - Literature map (published experiments per findings-map section)
domain: vertical-intelligence
kind: literature
tags: [research-program, literature, experiments, prior-art]
updated: 2026-10-26
---

# Literature map

Read this if: you want to know what published experiments already say about a section of
[[findings-map]] before running its test in [[experiment-plan]]. It was compiled 2026-10-26 by
five parallel research agents using web search. **[v]** = citation and headline result checked
against a source this session (mostly abstracts or secondary pages, not full texts). **[u]** =
not verified: check before citing. Numbers are as reported by the source.

Five-line summary: the closest prior work to the "for me" is **Egbert, Barandiaran and Di Paolo
2010**, where attraction and repulsion emerge from a protocell's own metabolism with no
receptors. Nobody has shown a learning system discover its *own* setpoints. Learning-progress
curiosity has direct human evidence. Plasticity loss over long task sequences in deep nets is
established (Dohare 2024). The human reconsolidation "rewrite window" failed a high-powered
replication. Diversity-keeping search beats single-best search. Gradient descent *can* build
world models, which weakens one disconnect.

## 0. Value and the self ("for me")

| Experiment | Result | For us |
|---|---|---|
| Egbert, Barandiaran and Di Paolo 2010, PLoS Comput Biol 6:e1001004 [v] | Simulated protocells: metabolism directly sets tumbling. Attraction to metabolisable food and repulsion from toxins **with no receptors**; a compound becomes an attractant only after the cell can metabolise it; cells gather where metabolism is sustainable | Strong support for U17 and Q10: valence is defined by the cell's own viability. Template for X0.1 |
| Keramati and Gutkin 2014, eLife 3:e04811 [v] | Reward = reduction of drive (distance from setpoints); proof that reward seeking equals physiological stability when the discount < 1; risk aversion and anticipatory responses emerge | Support for X0.1. The **setpoints are designer-given**: the gap X0.6 attacks |
| Iizuka and Di Paolo 2007, Adaptive Behavior 15:359 [v-cit] | Preferences emerge as viability constraints on a plastic controller; a preference is not stored in one place | Supports U16/U17; the homeostatic regions were evolved |
| Hesp et al. 2021, Neural Comput 33:398 [v-cit] | Valence inferred from the agent's confidence in its own policies, in a T-maze | One signal for good and bad (U17), but model-internal, not viability |
| Friston 2013, J R Soc Interface 10:20130475 [v-cit]; critiques Bruineberg et al. 2022 BBS [v-cit], Aguilera et al. 2022 [v-cit] | A Markov blanket boundary self-organises in a "primordial soup"; critics argue a statistical blanket is not a real self | A "me" boundary can form, but its status is contested: X0.3 tests it |
| Varela, Maturana and Uribe 1974, Biosystems 5:187 [v-cit]; Beer 2020, Artificial Life 26:5 [v-cit] | Computational minimal self-producing, self-bounding systems | Self without value or behaviour |
| Adamala and Szostak 2013, Nat Chem 5:495 [v] | Protocells with an internal catalyst grow and win competition | "Better for the protocell" with no designer; selection, not a felt signal |
| Bechara et al. 1997, Science 275:1293 [v]; critique Maia and McClelland 2004, PNAS [u] | Skin-conductance warnings before conscious knowledge (Iowa Gambling Task) | Supports U16; contested on how unconscious the knowledge is |
| Botvinick and Cohen 1998, Nature 391:756 [v]; Moseley et al. 2008, PNAS 105:13169 [v] (failed replication Rohde et al. 2013 [u]); Ehrsson 2007, Lenggenhager et al. 2007, Science [v-cit] | Ownership is built from multisensory correlation; when the "me" boundary shifts, threat responses follow it | Q10: "mine" is learned. X0.7 tests ownership built from *consequences* |
| Tso and Adler 1974 [v-cit]; Sourjik and Berg 2004, Nature 428:437 [v-cit] | Different repellent classes use **different receptors**; receptor clusters signal cooperatively | **Correction:** E. coli does not have literally "one receptor for food and poison". Separate receptors feed a shared signalling pathway (convergence: [u] from these searches) |

**Gaps:** learned (not given) setpoints; a suppressed-drive breakdown test; one valence signal vs
two drives; ownership by consequences; a self/world partition vs a statistical blanket.

## 1. Drive

| Experiment | Result | For us |
|---|---|---|
| Kidd, Piantadosi and Aslin 2012, PLoS ONE 7:e36399 [v] | Infants look away from too-simple and too-complex sequences | Supports U9/U12 (not uncertainty reduction) |
| Ten, Kaushik, Oudeyer and Gottlieb 2021, Nat Commun [v] | Humans choose learning activities by **learning progress** | Direct support for the edge / learning-progress drive |
| Baranes, Oudeyer and Gottlieb 2014, Front Neurosci 8:317 [v] | People go easy to hard but **also sample unlearnable games** | Partial challenge to pure learning progress |
| Oudeyer, Kaplan and Hafner 2007, IEEE TEC 11 [v] | A learning-progress robot moves from easy to harder situations | Closest ML analogue of U9 |
| Burda et al. 2018 large-scale curiosity [v]; RND 2018 [v]; Pathak et al. 2017 [v] | No-reward curiosity works widely; a noisy TV **slows** but does not always stop learning | Consistent with C40 |
| Kang et al. 2009, Psych Sci 20:963 [v]; Gruber et al. 2014, Neuron 84:486 [v]; Bromberg-Martin and Hikosaka 2009, Neuron 63:119 [v] | Curiosity peaks at moderate confidence; curiosity boosts memory, even for incidental items; dopamine values information | Drive as a learning amplifier, which NeuroBlock's expand drive lacked |
| Eastwood et al. 2012 [v]; Geana et al. 2016 CogSci [v]; Wilson et al. 2014, Science 345:75 [v] (interpretation disputed) | Boredom = wanting but unable to engage; boredom rises in low-information settings and predicts exploration | Matches U9 almost verbatim |
| Lepper, Greene and Nisbett 1973 [v]; Deci, Koestner and Ryan 1999 meta-analysis, d = -0.28 to -0.40 [v] (disputed by Cameron et al. 2001) | Expected external rewards reduce intrinsic motivation | Basis for X1.4 |
| von Stumm et al. 2011 [v]; Credé et al. 2017 grit [u] | Curiosity + effort rival intelligence as predictors; grit overlaps conscientiousness | Supports U15, with a caveat on specificity |
| Berridge and Robinson 1998, Brain Res Rev 28:309 [v] | Dopamine is **wanting** (incentive salience), not liking or learning | Wanting is a separate variable from "can learn" (U9). Basis for X1.6 |
| Friston, Thornton and Clark 2012 [v]; Sun and Firestone 2020, TICS [v] | Dark-room rescued by built-in priors; critics say the rescue fails | No simulation found: X1.1 would be new |

## 2. Representation and 7. Substrate

| Experiment | Result | For us |
|---|---|---|
| Cornford et al. 2021 ICLR [v-cit] | Dale-constrained nets learn as well as standard ones if inhibition is built in as a normaliser | Signs need not hurt |
| Li, Cornford, Ghosh and Richards 2023 NeurIPS [v-cit] | The **spectrum** of the weights matters more than the sign constraint itself | Challenge to reading C42 as "signs matter": X7.1 must match spectra |
| Achterberg et al. 2023, Nat Mach Intell 5:1369 [v-cit] | A distance-based wiring cost yields modular, small-world, energy-efficient networks | Supports I6 and C41 |
| Clune, Mouret and Lipson 2013, Proc R Soc B 280:20122863 [v-cit]; Kashtan and Alon 2005, PNAS [v-cit] (replication contested) | A connection cost produces modularity and evolvability; modularly varying goals produce modules | Supports I6 and C43 |
| Frankle and Carbin 2019 ICLR [v] | Sparse "winning tickets" work only with their original initial values | Which connections, not just how few |
| Frémaux and Gerstner 2016 [v-cit]; Bellec et al. 2020 e-prop, Nat Commun [v-cit] | Three-factor local rules approach BPTT | A local rule can nearly match gradient descent (C51) |
| Kirkpatrick et al. 2017 EWC, PNAS [v]; Hebbian continual learning (Thangarasa et al.) [v-cit] | Protecting important weights reduces forgetting | **No direct test of use-only updates found**, so X2.2 would be new |
| Poirazi et al. 2003, Neuron [v-cit]; Gidon et al. 2020, Science 367:83 [v-cit] | One neuron behaves like a two-layer net; human dendrites support XOR-like computation (in a model) | Supports U13: the point neuron is the wrong unit |
| Constantinescu et al. 2016, Science 352:1464 [v-cit]; Whittington et al. 2020 TEM, Cell [v-cit] | A grid-like code for abstract concept spaces; structure separated from content transfers | Supports U13. TEM is gradient-trained |
| Li et al. 2023 Othello-GPT [v-cit]; Lake and Baroni 2023, Nature 623:115 [v] | Gradient descent builds a board-state world model; meta-learning gives human-like compositionality | **Weakens disconnect 3:** GD can build structured representations; the gap may be data and objective |
| Attwell and Laughlin 2001 [v-cit] (spike cost revised down 2012); Kuzawa et al. 2014, PNAS 111:13010 [v-cit] | Energy limits favour sparse codes; child brain glucose use peaks at about **66% of resting metabolism** | Corrects the chat figure ("close to half") |

## 3. Parts and 4. Memory

| Experiment | Result | For us |
|---|---|---|
| Shazeer et al. 2017 MoE [v]; Jacobs et al. 1991 [u] | Experts must be load-balanced or a few take over | Supports U7 (takeover) and C44 |
| Heald, Lengyel and Wolpert 2021 COIN, Nature 600:489 [v] | One contextual-inference computation creates, updates and expresses memories | Strongest analogue of the Self as **holder** (Q7) |
| Gershman, Blei and Niv 2010 [v]; Collins and Frank 2013 [v] | Latent-cause inference explains context effects; people build task sets uncued | Supports "context inference is the bottleneck" |
| Goyal et al. 2022 shared workspace [v] | A capacity-limited shared channel encourages specialisation | Q7 as a limited shared space (X3.3) |
| Gazzaniga split-brain interpreter [u] | The verbal hemisphere confabulates causes | Q7 as **predictor-narrator**, not the Self |
| Shadick et al. 2013 IFS RCT, J Rheumatol 40:1831 [v] | Small trial; benefits for pain and function | Weak clinical support for parts work; not a mechanism test |
| Nader et al. 2000 [u]; Kindt et al. 2009 [v]; Schiller et al. 2010 [v]; **failed replication Chalkia et al. 2020, Cortex 129:496 [v]** | A reactivated memory can be modified; the human "rewrite window" did not replicate | U11 must not lean on a window |
| Sevenster, Beckers and Kindt 2013, Science 339:830 [v] | A memory changes only when retrieval contains a **prediction error** | U11 refined: reactivation **with mismatch**, not mere expression |
| Schwabe and Wolf 2013 [v]; Easterbrook 1959 [u]; Hostinar, Sullivan and Gunnar 2014 [v] | Stress shifts learning from flexible to habit systems; social presence buffers stress | Supports U10 (threat narrows); a strategy shift, not only less learning |
| Sale et al. 2007 [v]; Nardou et al. 2019, Nature 569:116 [v]; Pizzorusso et al. 2002 [u] | Enrichment, or an oxytocin-dependent signal, reopens adult plasticity | A safety or bonding signal reopens learning: basis for X4.4 |
| **Dohare et al. 2024, Nature 632:768 [v]** | Deep nets **lose the ability to learn** over long task sequences; reinitialising a few little-used units keeps plasticity | Direct evidence for disconnect 5. **Challenges U5 as a blanket rule**: turnover helps |
| van de Ven et al. 2020, Nat Commun [v]; McClelland et al. 1995 [u] | Replay of internal representations, with no stored data, reduces forgetting | Supports C35 and context-modulated replay |

## 5. World, 6. Randomness, 8. Time

| Experiment | Result | For us |
|---|---|---|
| Guo et al. 2017 ICML [v]; Nguyen, Yosinski and Clune 2015 CVPR [v]; Hendrycks and Gimpel 2017 [v] | Modern nets are overconfident; near-certain labels on noise; max-softmax is a weak "unknown" signal | Strong support for U3 and C31 |
| Schulz and Bonawitz 2007 [v]; Cook, Goodman and Schulz 2011, Cognition 120:341 [v]; Gureckis and Markant 2012 [v] | Children explore and design experiments when evidence is **ambiguous** | Ambiguity triggers experiments (X5.5) |
| Bengio et al. 2009 curriculum [v] | Easy-to-hard ordering helps | Matches C53; GD does depend on order |
| Schmidt and Lipson 2009, Science 324:81 [v] (contested); Udrescu and Tegmark 2020 AI Feynman [v] | Laws distilled from data; structural priors carried the discovery | Neighbours of E9 |
| Lehman and Stanley 2011 novelty search [v]; Mouret and Clune 2015 MAP-Elites [v]; Such et al. 2017 [v] | Novelty and niche-keeping search beat objective-driven search on deceptive tasks | **Challenges C17 as stated**; the fix for score-ranked collapse (X6.2) |
| Salimans et al. 2017 ES [v] | Evolution strategies are competitive at large parallel scale | Qualifies the "12x slower" result; ES is a smoothed gradient estimate |
| Romera-Paredes et al. FunSearch, Nature 625:468 [v]; Novikov et al. 2025 AlphaEvolve [v-cit] | Generator + exact evaluator + evolutionary loop yield new results | U6 pattern: variation aimed by a model, selection by a verifier |
| Sio and Ormerod 2009 meta-analysis [v]; Wagner et al. 2004, Nature 427:352 [v] (percentages [u]); Knoblich et al. 1999 [v] | Incubation helps (small to medium); sleep raises insight; impasses break by relaxing constraints and re-chunking | Supports U15 and X1.3 (wall = representation) |
| Gupta et al. 2010, Neuron 65:695 [v]; Ólafsdóttir et al. 2015 eLife [v]; McDonnell and Ward 2011 [v] | Replay includes never-run paths; preplay is reward-biased; neural noise can help | Generative, goal-aimed randomness (I5, U6) |
| Sheppard and Vernon 2008 [u]: speed-ability r about -0.24 across 172 studies; Bi and Poo 1998 [v]; Markram et al. 1997 [v] | Speed explains a modest share of ability; spike **order** within ~20 ms decides strengthen vs weaken | Supports U14 (speed not required) and the order reading |

## Corrections to things said in chat on 2026-10-26

- Child brain energy: about **66%** of resting metabolism at the peak (Kuzawa 2014), not "close to half".
- E. coli: separate receptor types feed a shared pathway; not literally "one receptor array for both".
- Reconsolidation: the human rewrite-window result failed replication (Chalkia 2020); the robust part is that a memory changes only when retrieval contains a prediction error (Sevenster 2013).
- The Iowa Gambling Task and Wilson et al. 2014 are both contested in interpretation.
