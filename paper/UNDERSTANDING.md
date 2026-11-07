# A guide for you: what is going on, what the math says, where the limits are

Written for the project's owner, to be read slowly and argued with. It is updated by the research loop. Nothing here asks you to trust it: every claim points to a test in `LEDGER.md` or a source in `LITERATURE.md`.

## 1. What you are trying to build, in one paragraph

**Corrected by you (2026-10-22):** the dream is a language model that, given only physics as it was understood before quantum theory plus the measurements, proposes that quantum theory must exist, without ever being told. More broadly, a model that genuinely learns and creates semi-new things. The paragraph below describes the supporting pieces.

A learner that, the more problems it has attacked, gets faster at working out **what affects what** in a new problem, while behaving exactly like an ordinary neural network when it has no experience. The long-range goal is a language model that treats every input as a claim to be checked against what it already understands, knows when it lacks the variables to answer, researches to fill the gap, and improves itself across generations. Right now the project tests the smallest independent pieces of that, on small simulated worlds.

## 2. What the tests have found, in plain English

1. **Knowing what matters is worth a lot, but only when there is clutter.** With a few inputs a network learns just as fast whether or not it is told which ones matter. With 200 useless inputs mixed in, telling a network the true inputs cuts its early error roughly in half and makes it reach a good level about six times sooner (MLP), and cuts a transformer's early error by about 37%. (Ledger C9.)
2. **A network can learn, from earlier problems, which kinds of inputs tend to matter.** After about ten earlier problems it recovers roughly half of that benefit without being told anything about the new problem; after a hundred, about 60% with the best prior. A plain statistical screen with no experience recovers almost none on the varied problem distribution (but about a fifth on the simpler expL2 distribution), and that screen was a weak, all-or-nothing one; the fair soft version has not been run yet (queue R3). It does not keep improving forever with this design: it plateaus by about ten problems. (C10, C11.) **Important correction (iteration 13):** the 'percent of the oracle' numbers compare against a hard mask that keeps only the true parents, and that mask is not the best possible: the learned gate beats it in two settings. Read the absolute gains. And the gate does not learn the causal map: taught the TRUE parents, it does worse than when it learns from its own experience, because what a small network needs is 'which inputs are useful to predict with', not 'which inputs are causes'.
3. **Experience can hurt at first.** After a single earlier problem the learned prior sometimes made things worse (it over-fits to one world). This is ordinary over-fitting of the *prior*, not a flaw in the idea. It depends on the problem distribution: on the simpler expL2 distribution one problem already helped (+0.149). (C11.)
4. **With zero experience it is exactly a plain network.** Not approximately: bit for bit. But this is true by construction (the no-experience gate is a mask of ones), and any meta-learner that starts from a neutral modulation has the same property; it is a design check, not a discovery. (C12.)
5. **Old knowledge was not lost** in the count-table learner on a sequence of different problems (a neural network that was simply fine-tuned lost a little). (C8.)
6. **Some of your ideas did not survive a fair test in the form first stated**: curiosity did not beat random actions at finding which action does what in the simple worlds (random is already a good experiment when effects are immediate and independent), and the quantum-state encoding gave no advantage (a single qubit with real amplitudes is just a re-labelled probability). (C3, C16.)
7. **Literature warning**: neural networks trained across many tasks (meta-learning, in-context learning) are known to adapt like ideal Bayesian agents *inside the family of tasks they were trained on* (Ortega et al. 2019; Mikulik et al. 2020). So the claim "neural networks can't do this" is probably false as stated. What can still be true, and is testable, is narrower: it matters whether the system works on problems *outside* what it was trained on, whether it reads out an explicit "what affects what" map, whether it keeps improving without retraining, and whether it knows when it is out of its depth. (C14.)

## 3. The math you need, in one page

- **Prediction is compression.** A model that predicts well can encode the data in fewer bits. Total code length = bits to describe the model + bits to describe the data given the model (Minimum Description Length). A new connection is worth adding only if it saves more bits than it costs to describe. This is why the learner has no arbitrary thresholds for "add a link".
- **The KT estimator and the (1/2) log n rule.** For a yes/no outcome the best online estimator adds one half to each count; learning one probability costs about half of log2(n) bits (Krichevsky-Trofimov; verified). This price per parameter is why splitting data into many small groups needs a lot of evidence.
- **Why the first version thrashed.** Judging a new connection by how well it fits the data it was chosen on over-rewards it (the data were selected because they looked special). Judging it on data it has not yet seen (prequential testing) removes that bias.
- **Tracking change.** If the world changes, an online learner can mix "what I stored" with "what I am learning now" and shift weight quickly when the stored model stops predicting (fixed share; Herbster & Warmuth). This is the mechanism behind "recognise the drag changed".
- **Meta-learning is amortised Bayesian inference.** Training one network across many tasks teaches it the *prior* over tasks and to do the Bayesian update implicitly (Ortega 2019). Our relevance prior is a small, explicit version of the same thing. The same fact sets a ceiling: inside one family of problems, a meta-learner converges to the best prior for that family and then stops improving. 'Faster and faster forever' needs the family of problems itself to keep growing, which is a property of the teacher/curriculum, not of the student.
- **Training on your own outputs can collapse.** If each generation learns from labels produced by the previous generation, rare cases get lost and errors compound (Shumailov 2024). The recursive loop must be checked against ground truth the loop never sees.
- **Calibration.** A system "knows when it might be wrong" if its stated confidence matches how often it is right (Guo 2017). "Never wrong" is not attainable; "wrong only when it says it is unsure" is a measurable target.

## 4. Where the math supports your beliefs, and where it does not

- Supports: structure knowledge is valuable under clutter; priors over structure can be learned across problems; recognising a changed regime quickly has a clean theory; abstention/calibration is the right form of "knows its limits".
- Does not support (as stated): "neural networks cannot get faster across problems"; "quantum representation helps a classical learner"; "curiosity always beats random exploration"; "it should never be wrong".
- Open and interesting: whether an *explicit* effect map plus a generative model lets it generalise to families it was never trained on; whether structured randomness ("emotion") finds deeper connections than a single best learner; whether parts that spawn only when something is unexplained can build up complexity indefinitely.

## 5. Limits you should keep in mind

- All results are on small simulated worlds with random sparse dynamics; nothing here is evidence about language models, brains, or the real world yet.
- Results use few seeds in places; each ledger row says how many. Differences below about two standard errors should be read as noise.
- The learned relevance prior saturates by about ten problems. Its ceiling is an information ceiling: it sees eight statistics of a 100-tick window, and no amount of experience can beat the best possible use of those eight numbers.
- The recursive-improvement test (C13) is finished: no collapse over 150 generations, for a structural reason (fresh real data every generation, small carried state). Training the weights themselves on generated data would be expected to collapse.
- Everything was produced in one day on one machine and has not been replicated; two runs of the same protocol differ by more than their quoted error bars (exp N vs exp O).

## 6. Questions to chew on (not answered by the data)

1. What would make the "emotion" component more than noise: what property of the random biases would you want them to have so that they find connections the Intellectual side cannot?
2. When a part spawns because something is unexplained, what is the right notion of "unexplained": a persistent prediction error, a missing variable, or a contradiction between two stored beliefs?
3. For the ball-drop example: the system must decide whether "5 s from 15 m" is a measurement error, a different planet, or a different physics. What experiments would it run, and in what order?
4. Which of your beliefs would you be willing to abandon if a baseline matched the wrapper everywhere?

## 7. The fruit-fly brain scan and this project (added 2026-10-22)

What exists: a wiring diagram of one adult fruit-fly brain (about 139,000 neurons, about 50 million synapses; FlyWire, Nature, October 2024) and an earlier map of its central brain (hemibrain, 2020). The wiring says which neuron connects to which, roughly how many synapses, and a prediction of the chemical used. It does **not** say how strong each connection really is, how neuromodulators change things, how connections change with learning, or the small electrical connections (gap junctions), which are hard to see.
What people have already done with it: a simple spiking model of the whole central brain, with strengths set from synapse counts, predicted which neurons respond to taste and drive feeding, and the predictions were checked in living flies (Shiu et al.). A network built from the measured wiring of 64 cell types in the visual motion pathway, with the unknown parameters found by training it to detect motion, reproduced results from 26 experiments, and worked better because the real wiring is sparse (Lappalainen et al., Nature 2024). So "scan, code, then optimise" has already been done at circuit scale. Whether the same works for a whole brain is open and debated; the field's own report (State of Brain Emulation 2025) is the place to follow it.
How it ties to your idea: the project's central question, **what affects what**, is exactly what a connectome gives for free in a fly and what has to be inferred everywhere else (a new problem, a human brain without a scan, a language model's internals). A learner that gets fast at inferring structure from activity and experiments is therefore a *bridge*: it can be validated against a real wiring diagram where the answer is known (queue Q18-Q20), and later applied where it is not. Your "get a jump" plan, in measurable terms: (1) check that our assumptions about structure match real wiring (sparse, hubs, weights), (2) test whether our learner recovers a real connectome subgraph from simulated activity faster than baselines, (3) keep the representation (sparse graph plus learned strengths) compatible with how connectome-constrained models are built.
Limits to keep in mind: a model that is correct about wiring can still be wrong about function; every published connectome-based model needed assumptions or fitted parameters beyond the wiring.

**What the first real-wiring measurement says (2026-10-22).** In the public fly data, a typical neuron gets most of its input from very few partners: half of a neuron's incoming synapses come from about 3 partners, but to reach 90% you need about 9, and a few neurons listen to thousands. Our simulated worlds give each output 1-3 causal inputs, so they match the "main inputs" picture but not the long tail of weaker inputs. That is a useful warning: the part of the benefit that comes from *dropping irrelevant inputs* should survive, but the weak real inputs may be hard to detect with few samples, so the numbers from our toy worlds probably overstate how well structure can be recovered in real wiring. The next test (Q22) builds worlds with the measured distribution to check exactly that.

## 8. The offline parent-teacher idea (added 2026-10-22)

Your plan: a student with no language at all, taught continuously by an offline language model that acts as its parent. What is known: tiny models can learn to write coherent simple English when an LLM writes them a child-vocabulary curriculum (TinyStories); children learn language from under 100 million words and models normally need thousands of times more (BabyLM); agents can learn word meanings from scratch by acting in a simulated world and learn *new* words faster once they know some (Hermann et al.); a scripted "teacher bot" and a curriculum are a standard test bed, and the honest finding there is that current methods are still not sample-efficient at compositional language (BabyAI); infants find word boundaries from statistics alone in two minutes (Saffran et al.); and a theory exists of what the best teacher does (machine teaching: choose the smallest or best training set for *this* learner).
What is genuinely open and decides whether your version works: (1) **From literally zero** has to start before words: with perception, action and statistics (infants do), because symbols need something to point at. (2) **The student must be able to exceed the parent.** If it only imitates the parent it inherits every error; the way out is that it can check claims with its own experiments (this is where your curious-and-skeptical principle becomes mechanical), the same reason self-training works when a verifier filters (STaR) and collapses when it does not (model collapse). (3) **Generations**: a student that becomes the next teacher can make the language more structured (iterated-learning experiments) or lose the rare cases (collapse); a bottleneck plus verification is the difference we can measure. (4) The parent itself is an LLM with its own errors and biases.
What we will test first (queue Q24-Q27), at toy scale with a scripted teacher so the effects are clean: tabula-rasa student vs no teacher / random / fixed / adaptive teaching; noisy teacher with and without the student's own verification; chains of students; and whether the student can say "I don't know this" and ask. Choosing which offline model to run as the real parent is parked until you decide (it needs a downloaded model, hardware and a licence check).

**First results on the teacher idea (2026-10-22, toy world).** A student that knows no words was shown scenes and told one word at a time. With no real link between the words and the scene it never learned a word (meaning needs something to point at). A teacher who showed random scenes needed about 120 lessons for 15 words; one who cycled through the concepts needed about 70; one who watched what the student was still unsure about and picked each scene to rule out its remaining guesses needed 30, which is the minimum possible for these scenes (two lessons per word). So a parent that models its student is worth a lot, as the machine-teaching theory says.
A surprise: when the teacher's mistakes were random and independent, the student that simply believed everything did best. One wrong lesson is quickly outvoted by later right ones, so doubting costs extra lessons and buys nothing. Your skepticism principle is therefore not free: it pays off against **systematic** falsehoods (a parent who is consistently wrong about the same fact) and only when a fact can actually be checked by an experiment. That is the next test (Q25b), and it is the honest form of your ball-drop example.

**Result on recursive self-improvement (2026-10-22).** Eight independent chains each went through 150 problems; after every problem the network's own judgement of which inputs mattered was used to train the prior for the next. The benefit rose for about 20 problems and then stayed put at about half of what a perfect oracle would give; it did not decay over 150 generations, and the eight chains agreed closely. It matched what plain experience gives without the loop, so the loop is *safe* here but not *better*. Why it is safe in this setup, and what it would take to break it: each generation's labels come from real data from a new world (a built-in checker) and the soft gate never fully switches an input off, so a wrongly dropped input can still be rediscovered. The published collapse result (Shumailov et al.) concerns models trained on data they generated themselves; the next test (Q29) builds exactly that case, plus hard gating and noisy labels, to find where the loop starts to fail. This is the practical version of your "recursively self-improving" claim: it works when a real-world check stays in the loop.

**Result on "can ordinary neural-network tricks do the same thing?" (2026-10-22).** I gave the plain network three tools that need no earlier experience: a statistical screen of inputs, an L1-style penalty that pushes useless inputs to zero (group-lasso, tuned fairly over a wide range), and input dropout. They recovered 0%, 7% and less than nothing of the benefit of knowing the true inputs. Then I let networks learn from experience in two different ways: a small network on hand-built statistics recovered 61%, and a convolutional network reading the raw early time series recovered 52-54%. Two lessons. First, the benefit really does come from *experience across problems*, not from a generic trick. Second, **the thing that learns from experience is itself a neural network**, so the belief that neural networks cannot get faster across problems is wrong in this setting: they can, when they are trained across problems. What your design adds is not "a non-neural mechanism" but a particular arrangement: the plain network is untouched, starts exactly identical, and only a small gate learned from experience changes it. The open question is whether that arrangement holds up on problems that look different from the ones it was trained on (Q16).

**Does the learned gate survive problems that look different from its training problems? (2026-10-22)** I trained it on the standard worlds and applied it, without changing anything, to worlds with other dynamics mixes, more parents than it had ever seen, much more noise, decoy inputs that are noisy copies of real ones, and effects that arrive one step late. It kept roughly half to 60% of the oracle benefit in every case and never made things worse, while a simple fixed statistical screen actually hurt when decoys were copies. This is encouraging but it is a *mild* test: those worlds come from the same generator with different settings. The strong version, a genuinely different kind of dynamics (oscillators, discrete networks, real connectome-derived wiring), is still to be run (Q16b, Q19).

**Skeptical student against a wrong teacher (2026-10-22, the testable ball-drop).** The teacher made claims about which actions affect which outputs in a simulated world; the student could consult what it had learned from earlier problems and could run its own small experiment on a claim. When the teacher's errors were *systematic* (consistently over-connecting one output and denying one action any effect), a student that checked only the claims where its own experience disagreed with the teacher reached 0.887 accuracy against 0.807 for believing the teacher: eight points better using about three experiments per world. Against an *honest* teacher the same student lost seven points (it doubted true claims on weak evidence), and with random teacher errors of about 10% it broke even. Checking *every* claim was worse than believing everyone, because a single short experiment is only about 79% reliable. Lesson: skepticism is a bet on how reliable the teacher is compared with your own evidence; the right design is to learn how much to trust each source from how often it agrees with the others (queue Q30), not to doubt by default.

**Where the self-improvement loop breaks (2026-10-22).** I deliberately tried to break it three ways and was wrong about one of them. (1) *Labels computed from the network's own simulated data*, the "model collapse" setting: I expected decay; instead the loop started lower and caught up to the normal level by generation 60. Reason: each new problem trains a **fresh** network on **real** data, so nothing but a small gate carries over, and a gate cannot absorb its own mistakes the way a network trained on its own outputs does. This is a design lesson more than a victory: the published collapse needs the generated data to flow back into the *weights*, which this loop avoids. (2) *Hard exclusion*: switching inputs fully off made the loop slowly worse (the gain fell from +0.063 to +0.049 over 60 generations and the fraction of inputs flagged as important halved), because an input that is switched off can never prove it matters. This is a classic confirmation-bias trap and is what the soft gate avoids. (3) *Wrong labels* (20% flipped): the benefit stayed flat at about a third of normal; noise lowers the ceiling but did not make things worse over time. So the practical recipe for a recursively self-improving system looks like: keep new real data flowing in, never fully shut off an option you have not re-tested, and keep the part that changes from generation to generation small. The remaining unknown is the genuinely dangerous case (training the weights on generated data), queued as Q29b.

**Correction and result (2026-10-22): does real-looking wiring break our conclusions?** After measuring the fly connectome I warned that our toy worlds, which give every output only 1-3 inputs, might overstate how well structure could be recovered from real wiring. I then built worlds that copy the measured statistics: about 8 inputs per output, strengths that fall off like a power law (the top three carry about half the influence), about half inhibitory. The warning was too pessimistic. Knowing all true inputs still cut the error substantially; knowing only the three strongest already gave 77% of that gain (so the weak tail adds real but smaller value); and the prior learned from earlier problems recovered about 64% of the full gain, the same as in the simpler worlds, whether it was trained on simple or on fly-like worlds. What this does *not* test: real connectomes have hubs, loops, motifs and delays that this generator does not copy. The honest summary: the benefit of learning "what tends to matter" is not an artefact of the toy generator's sparsity; whether it carries to real wiring dynamics is the connectome-derived test (Q19).

**Learning how much to trust each source (2026-10-22).** Instead of doubting by default, the student kept a running estimate of how reliable the teacher, its own learned prior and its own experiments each were, using only how often they agreed (the Dawid-Skene method from the truth-discovery literature), and tested a claim only when the combined evidence was unclear. It learned the teachers' reliabilities in the right order (about 0.90 for the honest teacher, 0.81 for the one with random mistakes, 0.67 for the systematically biased one) and matched the earlier targeted rule's accuracy with a few fewer experiments. It did **not** fix the main weakness: against an honest teacher it still lost seven points, because it never became fully sure the teacher was right. The likely reason is subtle and worth understanding: the method assumes the sources make independent mistakes, but the student's own prior and its own experiments tend to fail on the *same* faint links, so their disagreements with a correct teacher look like teacher errors. Also a warning about the method: left alone it can settle on the mirror-image answer (calling true false and false true) and scored below chance until it was told that sources are, on average, better than chance. So "skepticism" here is really a small statistical model of who to believe, with assumptions that must be checked.

**Your ball-drop example, run (2026-10-22).** The agent dropped balls from heights of its choice in simulated worlds (gravity from 1 to 25 m/s^2, with or without air resistance, noisy timing), fitted three candidate laws (free fall, constant-speed fall, and a drag model), and picked by the minimum-description-length rule. After about eight drops it predicted fall times within roughly 1-2% and picked the right family in about 85-90% of the drag-free worlds. Asked about "15 m in 5 s", it refused the claim every time on an Earth-like world (the true time is 1.75 s) and could say what would make it true: free fall on a world with gravity of about 1.2 m/s^2, which is weaker than the Moon's. Claims that were wrong by a factor of 0.4 to 2.5 were rejected with 93-98% accuracy after only three drops and over 99% after five or more. Honest limits: the true law was in the agent's list of candidates (so this is choosing among laws, as the SINDy method does, not inventing one), the planted false claims were far from the truth, and a "curious" experimenter that tests the heights where its candidate laws disagree helped only with very few drops: later it did slightly worse at predicting than random heights, because it stopped covering the range. The next test (Q12b) removes the true law from the list to see whether the agent notices that none of its ideas fit and says "I need more research" instead of confidently giving a wrong answer.

**Does the agent know when its whole set of ideas is wrong? (2026-10-22)** I removed the true law from the agent's list of candidates (air resistance proportional to speed, or gravity that grows along the fall) and gave it a way to measure how noisy its own timing is by repeating a few drops, then a test of whether its best law fits worse than that noise allows. With 3% timing noise it noticed in only about 1 case in 15 (linear drag) to 1 in 4 (changing gravity), which is barely above its false-alarm rate. The reason is not carelessness: its drag model is flexible enough to imitate these laws to within the noise, so there is nothing detectable to notice, and its verdicts on claims stayed 98-99% accurate anyway. When I made the timing 30 times more precise the same test caught the problem 84-98% of the time with almost no false alarms. This is a limit of the mathematics you can rely on: **"knowing that you don't know" has a resolution.** An agent can only detect a gap in its understanding that is larger than its measurement noise relative to how flexible its current ideas are, so the honest statement is not "I am certain" or "I am lost" but "my models cannot be told apart from alternatives within about X%". That resolution statement is the next thing to build (Q12d). A second lesson: when the test did fire, a lazy fix (adding one generic power-law term) failed; expanding a model's vocabulary needs a principled search, not any extra term.

**Is "emotion", structured randomness, what finds deeper connections? (2026-10-22)** I tried to give your idea its best chance in two forms. *Form 1:* a population of learners whose relevance gates are random variations of the learned one (the best chosen by early performance). It lost to the single learned gate in every kind of world, and arbitrary random gates were useless. *Form 2:* random **proposals of new connections** that a tester then checks (here: which pair of inputs, multiplied together, predicts an output, a connection that ordinary correlations cannot see). Proposals drawn according to the learned prior beat uniform guessing by up to 2.6 times at small budgets. But then a boring deterministic rule, "check the pairs among the inputs the prior ranks highest, in order", beat the random sampling (26% against 21% success with 200 proposals, 24% against 13% with 50), so the benefit comes from the **prior's ranking**, not from the randomness. Randomness showed its only real value as insurance: if the prior is confidently *wrong* (ranking backwards), the deterministic rule dropped to 11% while sampling in proportion to the prior stayed at 15%. So a sensible reading of your "emotion" part is a **dial for how adventurous to be**, to be turned up when the learner has reason to distrust its own prior and down when the prior has been right, much like the exploration setting in a decision problem; setting it requires an estimate of how reliable the prior is (the source-reliability idea again). What would still make randomness matter in your sense: hidden structure that no ranking can reveal and that only combinations of many unlikely guesses can find (triples, rare regimes); the pair test also has a ceiling (about 46% success even when checking every pair) set by how much data the test sees. These are the next experiments (Q10e).

**Your quantum-state idea, tested properly (2026-10-22).** The earlier experiments found no benefit from encoding bits as continuous quantum-like states on the car and causal worlds. This time I built the setting where the idea has published support: judgments where the *order* of two yes/no questions changes the answers, as in human survey data (the quantum-cognition programme of Busemeyer and others; their "QQ equality" is a precise prediction that a two-dimensional quantum model makes with no free parameters, which an ordinary order-dependent table of probabilities does not have to satisfy). I generated data three ways and scored three predictors by exact Bayesian code length. When the data came from a quantum process, the quantum predictor won in 257 of 300 runs with 20 trials and in all 300 with 1,000 trials, but its advantage is *thrift*: two parameters instead of six, worth 8% fewer bits per trial at 20 trials and only about 1% at 1,000. When the data came from an ordinary classical process (either kind) the quantum predictor lost badly, by a quarter to nearly half a bit per trial, because the constraint it carries is false for such data. So the honest statement is: **a quantum-style representation is a hypothesis about the structure of the world, not a general improvement to a learner.** It pays when the world really has that structure and costs when it does not. Nothing the project has met so far has that structure. Whether human judgments or some data you care about do is an empirical question, and it is one a learner could answer for itself by comparing the two codes (Q9b). Still unanswered and separate: whether real quantum hardware could speed up the *search* for structure.



## Appendix (iteration 12): real fly wiring as the test world

Q19 (iteration 12): connectome-derived world. 60 FlyWire v783 subgraphs (300 neurons, pairs >= 5 synapses, sign from dominant transmitter), rate dynamics over all 300, agent sees 50 core neurons + 3 injected action channels (hidden: 250 neurons; correction iteration 13: true parents are 25% of inputs in the 40 held-out worlds, 33% in the training worlds). 20 subgraphs train a connectome prior, 40 held out. Plain error 0.870 (ticks 300-1000). Gain over plain: oracle +0.098+-0.005, synthetic-trained prior +0.093+-0.002 (95% of oracle), connectome-trained prior +0.108+-0.002 (110%); improved in 40/40 worlds each. BUT true-parent rate is 0.33 (dense wiring, small headroom) and the AUROC of the priors for the true direct parents is only 0.56 (synthetic) / 0.63 (connectome): the gain is NOT edge recovery. Share >100% means the prior removes inputs that hurt a small-data MLP even if they are true direct parents (hidden neurons make direct parents weak predictors). Limits: one simulator setting (leak 0.3, gain 2, noise 0.05, untuned), synthetic dynamics on real wiring, 50/300 observed, one seed per subgraph.

Plain reading: the self-learning wrapper still helps when the 'world' is wired like a real fly brain piece, about as much as knowing the true parents. That does not mean it found the wiring; it learned which inputs to ignore. The control that would settle whether the real wiring matters (versus any dense random network) is Q19b.


## Appendix (iteration 13): what an independent review of the whole project found

An independent reviewer (Fable 5.1) read all the code, results and documents; the full review is `paper/FABLE_REVIEW.md`. In plain terms:

- **What holds up.** Knowing which inputs matter helps a lot under clutter. A small gate learned from about ten earlier problems recovers a real part of that help and keeps it under mild changes of the problem family. Retraining on its own labels did not collapse, for a reason that is understood. The negative results (curiosity vs random, quantum encoding, randomness as emotion) are clean.
- **What was described wrongly.** The gate is not learning "what affects what" (causes); it learns "what is useful to predict with" for a small network, and mathematically it behaves like a learned per-input learning rate. Some numbers were compared with a reference that is not the best possible, a curiosity result was an artefact, and the constants register no longer passes its own audit. All of these are now corrected or queued.
- **The tension at the heart of your new direction.** A perfect teacher makes the student certain. A certain student has nothing left to be skeptical with. So "taught only truth" and "healthy skepticism" pull in opposite directions unless the student (a) also keeps its own way to check claims (experiments) and (b) learns how reliable each source is, or the oracle also teaches it *about limits* ("here is what you have not been taught"). The next experiments (E1-E5) measure exactly this: is the taught student's confidence honest, does it notice when a question is outside what it was taught, and does it treat a contradicting claim differently from contradicting data.
- **Limits of the math to keep in mind.** (1) Without experiments, watching a system cannot in general tell a cause from something that merely tracks the cause. (2) A learner's best possible accuracy is fixed by the information in what it looks at; experience cannot exceed it. (3) Inside one family of problems, learning-to-learn stops improving once it has learned that family. (4) Being well calibrated is always relative to the kind of data it was calibrated on; there is no detector that notices every kind of novelty, so "never wrong without knowing it" is unattainable in principle; the attainable target is "honest inside what it was taught, and abstaining outside a boundary it can detect, with a stated resolution". (5) A learner trained only on its own outputs gains no new information about the world.


## Appendix (iteration 13, continued): how much of the gain really needs experience?

The reviewer suspected the 'experience' story was mostly a calibration trick. Test: give a no-experience screen the same soft gating, using one simple number (how strongly an input's value correlates with the output's next change), squashed through an S-curve with two dials set on 24 practice worlds. Result: that screen gets +0.082; the learned prior from 100 problems gets +0.117; the difference +0.035 is real (seven error bars). So roughly 70% of the benefit is 'soft-weight inputs by a simple statistic', and roughly 30% is what experience across problems adds. The earlier claim that no-experience methods recover 'at most 7%' was wrong because those methods were all-or-nothing. Lesson for your design: the self-improving part is real but modest here; most of the work is done by turning a statistic into a well-calibrated soft weight.

**A bookkeeping error found on the way:** a change to the world generator in iteration 3b silently made 'world number 9000' a different world in later experiments. Each experiment is internally fair (every arm sees the same worlds), but numbers from early and late experiments come from different samples of worlds. That most likely explains why two runs of the same protocol disagreed (exp N vs exp O).


## Appendix (iteration 14): is the taught student's confidence honest, and does it know when it is out of its depth?

We taught the student the exact truth (which inputs are real causes) on 100 practice worlds, then asked it 'how likely is this input a cause?' on new worlds.

- **On worlds like the ones it was taught on, its confidence is honest.** When it says 2%, about 2% are causes. It ranks real causes well (90% of the time a cause outranks a non-cause).
- **On worlds that differ only mildly, it stays honest.**
- **On worlds built differently (many more causes per variable, or real fly wiring), it is confidently wrong.** On fly-wiring worlds it says ~1% while the truth is ~24%, and its ranking is no better than a coin flip.
- **It cannot feel that it is wrong.** Asking several copies of the student and seeing whether they disagree is the usual 'am I unsure?' test. On exactly the worlds where it is wrong, the copies agree MORE (they all confidently say 'no'). Self-doubt measured from inside the model fails.
- **What does work, mostly, is 'this looks unlike anything I was taught'**: comparing the new world's raw statistics with the practice data flags most of the harmful worlds (about 3 out of 4 correctly ranked), but missed the fly-like ones.
- **A standard statistical safety net (conformal prediction) gave a false sense of safety**: it promises '90% of the time the truth is in my answer set', and it keeps that promise on average by always answering 'not a cause' (true 98% of the time) while missing almost every real cause.

Plain lesson for your design: 'taught the truth' gives honest confidence only inside the territory it was taught. Healthy skepticism needs a separate sense of 'am I in familiar territory?', built from the data rather than from the model's own feelings of certainty, and an experiment to check when that sense says no. This is why the next tests (E8: an oracle that exposes its limits; X1: the student runs its own experiments) matter: they are the two ways to give the student that sense.


## Appendix (iteration 15): the student runs its own experiments, with perfect instruments

The student drops a ball from heights it chooses and times the fall exactly. With just five drops it always worked out which law the world follows. It always noticed when the world followed a law it did not know (linear drag, gravity that changes along the fall), with no false alarms. And it caught claims that were wrong by as little as 1%. It did not matter how it chose the heights: random heights worked as well as clever ones.

With 3% timing noise the picture changes. Claims off by 1% or 10% slip through, laws outside its library mostly go unnoticed, and here choosing heights cleverly helps a bit at small budgets.

What this teaches: with perfect measurements, poking holes in a bad theory about one quantity is easy. The genuinely hard problems are (1) inventing the missing law once it knows none of its laws fit, (2) worlds with many variables where the choice of experiment decides what can be learned at all, and (3) worlds that are random in themselves. Those are the next tests.


## Appendix (iteration 18): the first test under the new rule (predictions written down before the run)

I wrote five predictions about the learned gate's experience curve, then ran 50 versions of it on 96 worlds. **One of five passed.** That is the point of the rule: the four failures taught more than the pass.

- **The earlier claim 'it stops improving after about 10 problems' was wrong.** It was an artefact of fitting each version only once. With 10 versions per level, the gain keeps rising: 0.093 at 10 problems, 0.110 at 30, 0.118 at 100. Each extra tripling of experience adds about half as much as the previous one: diminishing returns, not a wall.
- **One earlier problem makes things worse**, in half the worlds. Learning from a single example over-generalises.
- **With little experience, luck dominates**: which problems it happened to learn from, and how well that matched the new problem. After about 30 problems only the new problem itself matters.


## Appendix (iteration 18): the first theorem to pass a test written down in advance

The gate multiplies each input by a number p between 0 and 1. I derived on paper that this must be exactly the same as two other changes: shrink that input's starting weights by p, and let it learn p times slower. I wrote down the prediction before running, then tested it on 48 worlds. The two versions agreed almost perfectly (correlation 0.996 across worlds). **That is a theorem confirmed by a test it could have failed.**

My second guess failed: I expected 'learn slower' to be the important half. Each half alone gives only a fifth of the benefit; together they give all of it. A useless input has to be both quiet now (small weights) and slow to grow back (low learning rate). Biology may have a parallel: a synapse's strength and its plasticity are separate properties, both regulated (not yet checked).


## Appendix (iteration 19): piece B1, fast memory plus slow memory, like the hippocampus and the cortex

The brain seems to have two memory systems: a fast one that records specific experiences right away (hippocampus), and a slow one that gradually learns general patterns (cortex). During rest and sleep the fast one replays experiences to the slow one. I tested this on a world that secretly switches between three sets of rules, coming back to each one later.

- **Replaying old experiences protected old knowledge:** when a familiar set of rules came back, errors were 23% lower.
- **The fast memory helped at the moment of change:** combined with the slow learner, errors in the first moments of a brand-new situation were 15% lower.
- **Neither alone was best overall**, and over the whole run the benefit was small, because most of the time nothing is changing.

That is exactly the division of labour the theory predicts. One fairness question is still open: the replay versions also got twice as many learning steps, so the next test gives the plain learner the same extra steps.

**Fairness check (iteration 19b):** I gave the plain learner the same extra learning steps the replay version had. Half of replay's benefit disappeared. It still protects old knowledge (12% fewer errors when an old situation returns), but now it costs about 5% on new situations and overall. That is the classic stability-versus-plasticity trade-off: keeping the past costs some ability to fit the present. It also suggests a design question for your sleep idea: replaying only during rest, instead of while working, might keep the protection without the cost. That is a test for piece B9.


## Appendix (iteration 20): piece B7, a learning speed set by surprise

In the brain, noradrenaline seems to signal 'something unexpected just happened, the situation changed'. I gave the learner the same signal: compare its recent errors with its usual errors. **The signal itself works well**: two-thirds of its alarms happened right after the world really changed, about eight times more often than chance.

What to *do* with the alarm was surprising. **Learning faster when surprised made things worse** (noise triggers it too often). **Wiping the learner's short-term learning history made things worse too.** **Doing both together helped**: 7% fewer errors right after a change, 9% fewer when an old situation returned, and no cost when nothing was changing. That is the second time today that two pieces only work as a pair (the gate's theorem was the first). I don't yet know why; finding out is the next test.


## Appendix (iteration 21): piece P5, mastery, or does practice make repeats free?

Ten different problems came up in random order, ten times each. The learner was never told which one it was facing.

- **One brain for everything was a disaster:** about five times worse than starting fresh each time, because each problem overwrote the others.
- **A library of memories, one per problem, works only if you know which one to use.** When I told it which problem it was, the error fell steadily with practice: 31% lower by the second time, 48% lower by the tenth.
- **When it had to recognise the problem itself, it failed:** it picked up the right memory only a quarter of the time, and it kept making new memories (73 of them for 10 problems).

So mastery has two halves: practice and **recognition** (knowing *which* situation you're in). Recognition is the hard part. In the brain this is thought to be a hippocampus job (telling similar situations apart and completing partial cues); that link is not yet checked. Also, even with perfect recognition, cost did not approach zero after 10 repeats; the world's own randomness sets a floor.

**Second try at recognition (iteration 22):** a fairer rule: use an old memory only if it predicts the first moments better than starting from scratch would. Correct re-use doubled (25% to 50%), but it still made 42 memories for 10 problems, and wrong re-use rose to 13%. The likely reason is a vicious circle. Each time recognition misses, a new memory is made, practice gets split between several half-trained copies, and each copy becomes harder to recognise. The brain's apparent answer is consolidation during sleep: merging and tidying memories offline. That is exactly your sleep idea, and it is now a concrete test: does offline merging end the vicious circle?


## Appendix (iterations 23-24): sleep, recognition, and a hypothesis of mine that was wrong

**Sleep as tidying memories (B9).** At 'night' the learner merged memories that predicted each other's experiences. It went from 44 memories to 30, but nearly a third of the merges joined two *different* problems, and mastery did not improve. **How much evidence does recognition need? (P5c)** With well-trained memories, picking the right one of ten from the first 50 moments works 85% of the time, and 92% after 300. So recognition itself is not hopeless. But the right memory, used as it is, predicted the new episode *worse* than a brand-new learner about half the time.

I guessed this was a measuring mistake on my part (each network judging itself on its own scale). I tested that guess, and **it was wrong**: on a common scale the effect is even stronger. The real reason: in these worlds the 'same' problem behaves differently each time it starts, because it starts from a different place and drifts into different territory. So knowing *which* problem you're in is not enough; the stored knowledge has to be **adapted quickly** to this occasion. That matches the earlier oracle result: start from the right memory and keep learning, and practice halves the error. For your idea that a mastered problem should cost zero energy: in a world that never repeats exactly, the floor is not zero; it is the cost of adapting.


## Appendix (iteration 25): correcting my own correction

Last round I concluded that a stored memory, used as is, predicts a new round worse than a fresh learner about half the time, because the 'same' problem behaves differently each time. **That was wrong.** I checked it on an objective yardstick (a fixed scale per problem that only the experimenter knows): the right memory is picked 94% of the time and beats a fresh learner 83% of the time. Letting the world settle first adds only a little. My first guess was the right one: the learner was comparing networks that each measure their errors on their own scale, like comparing prices in different currencies without converting. My test that 'disproved' that guess used a converter that was badly off in the first few moments, which wrecked the comparison.

Lesson for the design: **a mind comparing its own hypotheses needs one stable common yardstick.** The fix the learner can actually use is to judge every candidate on the scale of everything it has experienced so far. That is the next test.

**Next round (iteration 26):** with one fair yardstick, the learner swung the other way. Almost any trained memory beats a brand-new one, because all these worlds share some habits. So it accepted the *wrong* problem's memory 63% of the time. The right question is not 'is this memory better than nothing?' but '**does this situation look like the ones this memory knows?**' That is the same skill the taught student was missing earlier (knowing the edge of what it was taught). Two of your pieces turn out to be one: mastery needs the learner to know what its own competence looks like.


## Appendix (iteration 27): where the mastery piece stands after seven attempts

Seven versions, 24 predictions written in advance, 2 passed. The picture is now clear even though the piece is not solved:

- **Practice works.** With the right memory, errors halve over ten repeats.
- **The hard part is one decision:** 'have I seen this before, and if so, which one is it?' Every fix moved the failure somewhere else. Too strict, and the same problem gets split into many weak copies. Too lenient, and the wrong memory gets used. A strict statistical test stopped wrong re-use (2%), but then it refused to choose between two copies of the same thing.
- **This is a known, deep problem**: when to treat a situation as new versus a variation of an old one. Psychology models it as inferring hidden causes. The brain seems to split the job: a general store that knows what usually happens (cortex), and a specific store for particular episodes (hippocampus). The next attempt uses exactly that split.

I am pausing this piece here so the other pieces get tested too, and will come back to it.


## Appendix (iteration 28): piece P3, what kind of curiosity?

The student had six things to study: two easy, two hard, and two 'noisy TVs' that are pure randomness with nothing to learn. It chose what to study, 600 times.

- **'Study what I understand least'** is a trap. Noise can never be understood, so this student spent 92% of its time watching static and ended up 2.4 times worse than choosing at random.
- **'Study where I'm improving fastest'** (learning progress) avoided the static: only 7% of its time. But it also chased the easy topics, where progress is easy to see, and neglected the hard ones. So it still ended worse than random.
- **Random spreading was hard to beat** at this budget. That also explains the earlier finding that curiosity didn't beat random.

For your design: curiosity's first job is to **not get stuck on the unlearnable**, and learning progress does that well. Its second job, sharing time fairly with hard-but-learnable topics, needs a better way of measuring progress. That is the next test.

**Follow-up (iteration 28b):** I tried better ways of measuring 'where am I improving', a bigger study budget, and more static channels. None beat choosing at random, even when random wasted half its time on static. The problem: 'how fast am I improving?' has to be estimated from noisy mistakes, and that estimate is mostly noise. One version even fell back into watching static, because random noise looks like change. The textbook fix is a different kind of curiosity: not 'where am I wrong?' but 'where would more data change my mind?' A noisy TV never changes your mind. That is the next test.


## Appendix (iterations 29-30): the curiosity piece, summed up

Six kinds of curiosity, four situations, 14 predictions written in advance, 5 passed.

- **'Where do several copies of me disagree?'** also fell for the static (60-80% of its time), because copies that keep learning at a fixed speed never settle on pure noise.
- **'Where will one more look tell me the most?'** (expected information gain) **solved the trap**: only 3-6% of its time on static, in every situation. A noisy channel teaches you little per look, and this measure knows that.
- **But no kind of curiosity beat spreading time evenly** at the end. They all neglected the hard topics, which improve slowly and so look unrewarding in the short run.

For your design: curiosity's proven job is **not wasting time on the unlearnable**. Choosing *well* among learnable things needs something curiosity lacks: a sense of what will matter in the long run, which hard things are worth sticking with. That sounds like your 'motivation' and 'purpose' pieces, the Self deciding what is worth effort. It is a concrete reason the Self has to be more than curiosity.


## Appendix (iteration 31): piece B11, is energy a burden or a help? (your assumption A1)

I charged the learner 'energy' in two brain-like ways: for **firing** (activity) and for **wiring** (how strong its connections are), and turned each charge up step by step.

- My first attempt was too gentle: the charges didn't change anything, so it tested nothing. I said so and turned them up.
- **Charging for firing** saves energy but always makes the learner worse at a changed situation, and worse the more you charge.
- **Charging for wiring helps, up to a point.** At the right level the learner keeps only about 5% of its connection strength, fires half as much, learns 6-8% better, and handles a changed situation 12-13% better. Too much, and it collapses.
- **This one survived the strict rule:** I fixed the setting in advance and re-ran it on 48 new worlds, and the benefit held (-12.8%).

What it means for your idea: a wiring budget is not just a cost the brain puts up with; it pushes the learner to bet that only a few connections matter, and in a world like that the bet pays off. Real fly wiring is extremely sparse too. Mathematically this is a known trick (an L1 penalty, a 'most things don't matter' prior). The brain may be using the same trick, paid for in energy. Firing costs, on their own, did not help here.


## Appendix (iteration 32): piece A3, what would a 'perfect brain scan' have to capture?

I built pretend brains on 40 pieces of the real fruit-fly wiring, giving each one every ingredient: who connects to whom, how strongly, whether each connection excites or inhibits, each neuron's own personality (how easily it fires), and a slowly drifting brain-wide chemical mood. Then I built copies that each lacked one ingredient, fed everyone the same inputs, and asked how closely each copy's activity tracked the real one. All five predictions passed.

| copy missing... | how much of the behaviour it still reproduces |
|---|---|
| nothing (complete scan) | 100% |
| excite-or-inhibit signs | **8%** |
| each neuron's own personality | 42% |
| connection strengths (wiring only) | 51% |
| the brain-wide chemical mood | 71% |
| nothing, but strengths measured with 5% / 20% error | 99% / 94% |

What it means for your third assumption: a wiring map alone gets you about half way. **Whether each connection excites or inhibits is the most important extra fact** (the real fly-brain simulation by Shiu and colleagues also needed it). Next come each neuron's own properties and the strengths. Precision matters less than completeness: a slightly blurry complete scan beats a sharp wiring-only one. Electron-microscope scans, the kind that exist for the fly and for a speck of human cortex, show wiring and synapse sizes, but not signs, neuron personalities or chemical state directly. Those have to come from other measurements. Caveat: I chose how strong each ingredient is in the pretend brain, so trust the ordering more than the exact percentages.

**Checking the ranking (iteration 33):** I re-ran it in 27 kinds of pretend brain (neurons more or less alike, stronger or weaker chemical moods, calmer or more excitable). **Signs came first every single time.** What comes second depends on the brain: if its neurons are all much alike, strengths matter more; if they differ a lot, each neuron's own properties matter more. And in an excitable brain, sloppy measurement hurts a lot more (20% error keeps only 50-70% of the behaviour, against about 90-99% in a calm one). So a real scan project must measure how different real neurons are from each other, and how excitable the circuit is, because those decide how much detail the scan needs.


## Appendix (iteration 34): piece T3, the rune universe (your 'foundational blocks' belief)

I made a tiny universe: strings of symbols, six basic 'runes' that change them (reverse, shift, and so on), and six hidden **blocks**, each a fixed pair of runes. Every problem secretly used two blocks. The solver tries possible rune programs, shortest first, until one explains the examples. One solver always works from the bare runes; the other **keeps any piece it used twice as a new building block**.

- The block-keeping solver **rediscovered 5 of the 6 hidden blocks by itself**, from its own solutions.
- After 60 problems it solved new ones about **4 times cheaper**.
- On harder, never-seen problems (three blocks deep) it was **6.6 times cheaper**: the deeper the problem, the more the blocks pay, because search cost explodes with length and blocks shorten everything.
- Surprise: even in a 'control' universe with random problems, keeping blocks helped (about 1.5 times cheaper). With only six runes, random problems still share pieces by chance. So in a small enough universe, everything relates a little.

This is your belief made precise: **how much past problems help is proportional to how much hidden structure they share**, and the help grows with how deep the new problem is. It also says where it would fail: in a universe with no repeated structure, stored blocks give nothing. Our real world seems to be the first kind.

**The clean version (iteration 34b):** I built a fairer control. It used a bigger rune set, and problems in which no pair of runes ever repeats, so they truly share nothing. Then I mixed in more and more problems built from the shared hidden blocks.

| share of problems built from shared blocks | how much cheaper the block-keeping solver is |
|---|---|
| 0% (truly unrelated) | 1.00x (no help, no harm) |
| 25% | 1.12x |
| 50% | 2.7x |
| 75% | 4.2x |
| 100% | 7.5x |

All three predictions passed. This is the clearest result so far: **learning from past problems helps exactly in proportion to how much hidden structure they share.** If the real universe is built on foundational blocks, as you believe, a learner that keeps its blocks gets steadily faster. If it weren't, nothing would help.


## Appendix (iteration 35): piece P6, a Self that grows its own parts (your main focus)

The world had six hidden zones, each with its own rules, appearing one at a time. The Self started with one part and made a new part whenever its errors stayed unexplained.

- **The idea has huge potential:** six specialists that each handle the right zone made **20 times fewer errors** than one big network of the same size. One network smears the zones together; parts keep them apart.
- **My first Self captured none of that.** It made parts (10 for 6 zones), but it sent each situation to whichever part's 'home' was nearest, and the zones are shaped like pizza slices, not circles. So the parts got mixed experience and learned nothing special.
- It did use only a sixth of the effort, since each moment trains one small part.

The pattern from the mastery piece repeats: the hard part is not having parts, it is **knowing which part a situation belongs to**. Next: a Self that learns its own routing from which part predicted best.

**Self v0.1 (iteration 36):** this time the Self learned its own routing, from which part had predicted each situation best. It made about the right number of parts (4 for 6 zones) and beat the single big network by 13%. But most of the possible 20-times improvement is still missing. Next I measure exactly why: is it sorting situations into the wrong parts, or making too few parts?


## Appendix (iteration 37): the biggest pattern so far

Even when I gave the Self exactly the right number of parts, it did barely better than one big network. Only half the time did it send a situation to the right specialist. So the problem is not how many parts, it is **sorting situations into parts**.

Look across the mastery piece and the Self piece together, 13 versions in all, and one fact keeps coming back:

- given the right memory, practice halves the errors;
- given the right part, specialists are 20 times better;
- but **working out by itself which memory or part fits the current situation** is where every version fails.

Psychology calls this inferring the hidden cause or context ('which situation am I in?'). In the brain it is thought to involve the hippocampus (keeping similar situations apart) and the prefrontal cortex (tracking which task or context is active); that link is not yet checked. For your design this is a strong candidate for **the Self's central job**: not doing the work, but recognising what kind of situation this is and handing it to the right part. That is the next thing to test directly.


## Appendix (iteration 38): how much is knowing the situation worth, and how good does it have to be?

I gave the Self a hint about which zone it was in, true some fraction of the time. More hints helped steadily: with the hint always there, errors almost halved compared with one big network. But it was still 10 times worse than the perfect version. The reason is striking. With perfect hints during training, the Self learned to recognise zones 94% of the time. **The 6% it got wrong caused almost all the remaining error**, because a wrong specialist is wildly wrong (about 160 times worse than the right one).

I tried hedging: blend the specialists' answers when unsure. It made things worse. When two situations need very different answers, the average is wrong for both.

This gives a simple law (theorem T8): **the more different the specialists are, the more accurately the Self must recognise the situation.** Here it needs to be right about 99.4% of the time to come within a factor of 2 of perfect. For your design that says something important about the Self: when it is unsure which situation it is in, it should not average. It should **check**, by looking again, running a quick experiment or asking. That links the Self to the experiment-running student.

**Testing the law on new worlds (iteration 39):** I made worlds where the zones' rules are more or less different, and wrote my forecast down first. **Confirmed:** the more different the zones, the more a recognition mistake costs (from 1.3x to 4x worse than perfect). **My arithmetic was wrong:** I assumed a mistake sends you to an average wrong specialist. Really, mistakes happen at the borders between zones and go to the *neighbouring* specialist, which is less wrong there. Mistakes are near misses, not random. **Asking when unsure works:** asking for the true situation on just the 10% least-certain moments cut errors by up to 44%, and it helps most exactly where mistakes are costly. That is the beginning of a principled answer for when the Self should stop and check.


## Appendix (iteration 40): piece B5, knowing what your own actions do (and a design mistake of mine)

The task: keep one variable of a world at a set value by choosing actions. I compared random actions, learning by reward (trial and error), learning a model of what your own actions do and planning with it, and a perfect planner.

**My mistake:** in more than half the worlds, the variable I picked could not be affected by any action at all, so every strategy tied, and every prediction failed for a boring reason. I report that as a failure and redo it properly.

**Looking only at the 20 worlds where control was possible** (this was decided after seeing the data, so treat it as a hint, not a result): the learner that models its own actions did best late on (error 0.28, against 0.34 for trial and error and 0.57 for random). But early on it was no faster, and both learners were still more than twice the perfect planner's error. The re-test is queued with the restriction written down in advance.

**The proper re-test (iteration 41):** this time I picked only worlds where control is possible, decided in advance. My prediction was about a planner that models the *whole* world and thinks three steps ahead, and it failed: it was no better than plain trial and error. The winner was the simplest model of your own actions: it predicts only the one thing you are trying to control, one step ahead. It did 23% better than trial and error early on, and came within 1.5x of the perfect planner. Modelling everything, including all the irrelevant clutter, lets small mistakes pile up as you plan ahead. So **knowing what your own actions do helps, as long as you only model what matters for your goal.** That is the project's first theme, 'what affects what', showing up again in action.

**Two pieces combined (iteration 42):** I gave the 'model of everything' the wiring-energy cost that worked earlier (charge for connection strength). With that budget it did **as well as the focused model** (P104 passed): it learned by itself to stop modelling the clutter. So the energy budget and the forward model fit together: an energy-limited model of your own actions automatically becomes a model of what matters. One honesty note: two predictions I labelled as replications ran on exactly the same worlds as before, so they only reproduced the earlier numbers. A real replication on new worlds is queued.

**Replicated on new worlds (iteration 43):** all three predictions held. A focused model of your own actions learns control faster than trial and error (23% less error early) and beats a do-everything planner (18% less error late). And an energy-limited model of everything ends up about as good as the focused one. Forward models now join the wiring budget and the building-block library as results that survived a pre-registered replication.


## Appendix (iteration 44): piece P1, how much should a mind be born knowing?

In the rune universe I gave the learner a head start: 2 of the 6 hidden blocks built in from birth. Another got 2 useless blocks, and a third got nothing.

- **The right head start helps early** (37% cheaper on the first problems), but **by the end it barely matters** (4%): a learner that builds its own blocks catches up.
- **A wrong head start hurts about as much as a right one helps** (49% costlier early), and **its harm lasts longer** (still 14% at the end): useless built-in pieces keep adding wrong paths to try.

For your 'taught from literally zero' idea this is encouraging, with a warning attached. Starting from nothing costs some early speed but nothing in the long run, *if* the learner can build its own building blocks. Starting with wrong beliefs is worse than starting with none. It is the Bitter Lesson in miniature: built-in knowledge buys early speed; learning wins later.


## Appendix (iteration 45): piece B12, protecting important memories

The brain seems to 'lock in' connections that mattered for something learned, so that new learning doesn't overwrite them. I tried the standard machine version of this (EWC) on the world that switches between three sets of rules.

- It helped very little when an old situation came back (3% fewer errors), and it made things **13% worse overall**.
- Triggering the locking by surprise instead of telling it when the rules changed kept none of the benefit.

The reason is instructive. Here the three situations need the *same* connections to take *different* strengths. Locking a connection to protect one situation fights the change the next situation needs. Locking works when different skills live in different connections. When they conflict over the same ones, the answer is **separate parts plus knowing which situation you're in**: the same conclusion the mastery and Self pieces reached from another direction.


## Appendix (iteration 46): piece B10, are real neurons smarter than the dots in a neural network?

A real neuron's dendrites (its input branches) do their own small computations. I built 'dendritic' neurons, each adding up four little nonlinear branches, and compared them with ordinary neurons at the same total size.

- **Dendritic neurons learned 4-5% better**, in both kinds of world, and **this held on fresh worlds**.
- **Branches that each listen to only a quarter of the inputs did a bit better still** (6-9% better than ordinary neurons when there was a lot of clutter). That is the 'only a few connections matter' bet again, like the wiring budget.

Small but real. One fair objection remains: a dendritic neuron is like an extra layer, so the gain might come from depth, not from dendrites as such. The next test compares against an ordinary network that is equally deep.

**The fair comparison (iteration 47):** against an ordinary network that is just as deep, dendritic neurons were no better (within about 1%). So their gain came from depth. But **branches that each listen to only a few inputs were still about 4% better than the deep network**. The useful part of dendrites here is not extra cleverness per neuron. It is that each branch only listens to a few inputs. That is now the third piece (after the wiring budget and the focused model of your own actions) where the same bet wins: **only a few connections matter**.


## Appendix (iteration 48): piece B8, learning when the reward comes late

In the brain, a connection that was just active gets a temporary 'tag'. If a reward or surprise signal (like a dopamine burst) arrives while the tag is still there, the connection changes. That lets the brain credit an action whose result shows up later. I tested exactly that rule against perfect bookkeeping, where the learner knows which action earned each reward.

- **No delay:** all three learners are equally good.
- **Reward 3 steps late:** the tag rule keeps **97%** of the perfect bookkeeper's performance. Without tags, the learner learns nothing at all (chance level).
- **Reward 10 steps late:** the tag (which fades over about 5 steps) keeps 48%.

All three predictions passed. Lesson: **a purely local rule, where each connection only knows about itself plus one global 'that was good' signal, solves late credit almost perfectly**, as long as the tag lasts about as long as the delay. Choosing how long tags last is the kind of setting the brain's chemical systems might tune; that is the next test.

**Follow-up (iteration 48b):** I tried giving each connection a mix of short, medium and long tags, for when you don't know how late the reward will come. It did *worse* than a single medium tag. Looking afterwards, my way of mixing them gave the long tag almost no say, so this doesn't settle the question. It leaves an open one: **how should a learner decide how long to remember what it just did?** That is likely a job for the brain's chemical signals, and still untested here.


## Appendix (iteration 49): piece P4, motivation, and the first study strategy that beats random

Curiosity alone always lost to random spreading because it gave up on the hard topics. So I gave the student a **motivation** rule with two steps:
1. ignore whatever can't be learned (curiosity's information-gain test);
2. among the rest, keep working on **whatever you are still worst at**.

**It beat random in all four situations** (8-16% fewer final errors). It spent 26-56% of its time on the hard topics, against 6-13% for pure curiosity. The weak spot: when half the topics were static, my simple 'ignore the bottom half' filter let some static back in.

This is your 'genius = motivation + processing speed' in a small, tested form. Curiosity tells you **what can be learned**; motivation tells you **what is worth the effort**; only together do they beat an even spread. It is also a first concrete job description for the Self: filter by learnability, then commit to the hardest learnable thing. A replication is next.

**Replicated (iteration 50):** on 100 new seeds the motivation rule again beat random in all four situations (8-20% fewer errors). Then I tried a stricter filter to waste less time on static. It did waste less, but it also gave up on the hard topics, because a hard topic, late in learning, also looks like it teaches little per look. So it lost. **Some time spent on things that turn out to be unlearnable is the price of not quitting hard things too early.** That sounds like good advice for people too.


## Appendix (iteration 51): piece B13, does the order of teaching matter?

Same 60 rune problems (20 easy, 20 medium, 20 hard), taught in three orders.

| order | total effort to learn everything |
|---|---|
| easy to hard | **1x** (18,400 tries) |
| random | 2.6x |
| hard to easy | **5.5x** |

Two of three predictions passed. The surprise: the easy-first learner did not find the building blocks any *sooner* than the random one; both had 5 of 6 after 20 problems. The saving is that the blocks were **ready by the time the hard problems arrived**. Easy problems are cheap to solve without blocks, and solving them *produces* the blocks. Hard problems are ruinously expensive without blocks and cheap with them.

For your offline-parent-teacher idea: **how a teacher orders the lessons matters as much as what it teaches.** Teach the simple combinations first, and the learner's own library makes the hard ones cheap.


## Appendix (iteration 52): four more results survive a re-run on new seeds

I re-ran four quick experiments with fresh random seeds, predictions written down first. All four held, with almost the same numbers: the building-block dose-response (1.1x to 8.3x as shared structure rises), the innate-knowledge result (right blocks 32% faster early, wrong ones 50% slower), the teaching-order result (easy-first 0.43x, hard-first 2.06x), and the delayed-reward tags (97% at a delay of 3). These now count as replicated.


## Appendix (iteration 53): the brain-scan result replicates; sleep replay does not beat awake replay

**Brain-scan ingredients, again:** on 20 fly-wiring pieces never used before, the same ranking came out (signs 8%, wiring only 53%, neuron properties 44%, chemical mood 72%). That result now counts as replicated.

**Sleep (your parked idea), tested fairly:** I let the learner replay old experiences only while 'asleep' (world paused, no new input), and compared it with replaying them a little at a time while awake, with exactly the same number of learning steps. They protected old knowledge **equally well**; sleep cost slightly more on new things, because it had fewer waking updates. So, in this setting, **sleep is not useful because of *when* it replays.** If sleep helps a mind, it must be by doing something different: tidying and shrinking connections (to save energy), compressing the day into reusable building blocks, or choosing *what* to replay. Those are the next sleep tests.


## Appendix (iteration 54): five tries at sleep, and why none helped

I tried five versions of sleep: merging memories, replaying experiences, making building blocks only at 'night', re-solving the day's problems with the new blocks, and re-solving plus throwing away unused blocks. **None did better than doing the same work while awake.** Delaying the work to sleep cost nothing, but doing extra work in sleep either bloated the library or threw away good blocks.

The likely reason is a flaw in what I tested, and it points somewhere interesting. In these toy worlds the waking learner has *no limits*: it can tidy up its memory whenever it likes, for free. So moving the work into sleep cannot possibly help. In a real brain, waking learning seems to fill up connections and cost energy until learning gets worse, and sleep resets that (the 'synaptic homeostasis' idea). So **sleep's value may depend on a limit of being awake**. That is the next fair test: give the waking learner a limit, then see whether sleep earns its place. This also matches your original instinct to treat sleep as its own study.

**Sleep with a waking limit (iteration 55):** I made waking learning 'fill up' (each change makes later learning slower) and let only sleep reset it. Then sleep helped, as built in, and the more often the better within the range I tried. The interesting failure was the learner choosing its own bedtime by **pressure** ('sleep when learning has slowed to half'). It slept 56 times and spent almost the whole run asleep, ending worse than never sleeping, because pressure builds back within a few dozen moments. A good bedtime rule has to weigh what sleep gives back against what it costs (time not doing anything). Your NeuroBlock design had both forced sleep after a lot of work and voluntary sleep, but it needs that cost side too.

**A good bedtime rule (iteration 56):** the learner kept a running total of how much learning it had lost to 'filling up' since waking, and went to sleep when that lost learning equalled the time a sleep takes. It came within **1.2% of the best fixed schedule**, picked separately for each world with hindsight, without knowing any schedule. Both predictions passed. So your idea that a mind should choose its own sleep works, if the choice weighs what sleep gives back against what it costs. One caveat: this rests on the 'learning fills up' limit I built in. Whether real brains work that way is the neuroscience question (synaptic homeostasis), not something this test shows.

**Replicated (iteration 57):** on 48 new worlds, and with learning filling up at three different speeds, the cost-aware bedtime rule stayed within 0.4-1.8% of the best fixed schedule. The best schedule itself changes with the filling speed, and the rule followed it by itself, sleeping a little less when learning filled up more slowly. That is exactly what a self-regulating mind should do.


## Appendix (iteration 58): asking when unsure, while learning

Could the Self learn which zone is which by asking a teacher only when it feels unsure (10% of the time)? It did about as well as getting random hints 25% of the time, so less than half the questions bought the same result. But in a typical world it was a bit worse, and still nowhere near the 20-times improvement correct parts could give. The reason echoes an earlier finding: while learning, the Self is often **confidently wrong**, so 'when I feel unsure' is not where help is most needed. A better trigger is next: ask when two of its own signals disagree (its router says one thing, the best-predicting part says another).

**And asking on disagreement (iteration 58b):** no better. The real ceiling is the Self's 'recogniser' itself. Even with the true answer every single time, it sorted situations correctly only 94% of the time, and the law found earlier says about 99.4% is needed. So the next test changes the recogniser: instead of a formula, a **memory** of remembered situations ('this looks like those moments I know were zone 3'), which is closer to how the hippocampus is thought to recognise places and contexts.


## Appendix (iteration 59): a memory-based recogniser

Instead of a formula, the Self recognised which zone it was in by looking up the five most similar moments it remembered (and whose zone it had been told). That is closer to how the hippocampus seems to recognise places.

- It made **half as many recognition mistakes** (3.2% instead of 6.4%), and the errors **halved** with them (5x the perfect version instead of 10x).
- With hints on only a quarter of moments, the memory recogniser was right 94% of the time against 79% for the formula.

The thresholds I set (98% right, within 2x of perfect) were not met. But the result checks the earlier law quantitatively: halve the mistakes, halve the cost. The remaining mistakes sit at the borders between zones, so the next step is a memory that checks exactly there.


## Appendix (iteration 60): check only at the borders

The memory recogniser looks at its five most similar memories. If they all agree, it trusts them. If they disagree, it is probably standing on a border between zones, so it **asks**.

- It asked on fewer than 1 in 11 moments, and got the zone right 99.5% of the time (from 96.8%).
- The error fell from 5x the perfect version to 1.3-1.4x. It held on a fresh set of worlds.
- With hints on only a quarter of training moments, most of what is left is not recognition any more: the specialists themselves learned worse, because during training they were sometimes taught with the wrong zone.

So 'know when your memories disagree, and check there' works. The open part is doing the same thing **while learning**, not just when answering.


## Appendix (iteration 61): checking while learning

This time the checking happened during training. When a moment came without a hint, the learner looked at its five most similar memories. If they agreed, it used their zone. If they disagreed, it asked.

- It asked on only about 7 in 100 moments. Its specialists ended up almost as good as specialists taught with the right zone every time (1.12x), against 2.6x without asking.
- Spending even more hints **at random** (35% of moments instead of the ~32% it used) was clearly worse (1.9x). Where you ask matters more than how much.
- With checking both while learning and while answering, the whole system came within 1.3x of the perfect version. This held on a fresh set of worlds.

**A surprise.** Using the memories' vote *without* asking made far fewer training mistakes (7% of moments sent to the wrong specialist, against 48%), yet the specialists were **no better**. My guess, not yet tested: the old way's many mistakes were harmless. It sent each moment to whichever specialist already predicted it well, so little changed. The vote's few mistakes were confident wrong guesses at the borders, and those did real damage. Being wrong less often is not the same as being wrong harmlessly.


## Appendix (iteration 62): why fewer mistakes did not help

I measured how badly each wrong assignment hurt.

- The old way (send the moment to whichever specialist already predicts it best) made many mistakes, about 2,200 per run. But each one was tiny: the specialist that received the moment already predicted it almost right, so it barely changed.
- The memory vote made only about 320 mistakes, but each was about **40-46 times** more damaging. It confidently sent a moment to a specialist that got it badly wrong, so that specialist was pulled hard in the wrong direction. In total the vote did 6-7 times **more** damage.

Lesson: being wrong rarely but confidently can be worse than being wrong often but harmlessly. The same thing happens when a person (or a neural network) trains on their own confident guesses: they reinforce their own errors ("confirmation bias").

Simply not learning from moments where the memories disagree helped somewhat, but not enough. The worst mistakes happened when all five memories agreed and were still wrong. Next idea: a 'surprise gate'. If a moment is very surprising to the specialist it is about to teach, do not trust the assignment.


## Appendix (iteration 63): a surprise gate, no questions

Rule: a specialist is not allowed to learn from a moment that shocks it, meaning its error on the moment is far bigger than usual for it. That moment goes to whichever specialist already predicts it best (last time those hand-offs turned out to be harmless).

- Without asking anyone anything, the specialists ended at about 1.45x the perfectly taught version, from 2.3-3.0x. That is about three quarters of the way to the learner that asked questions (1.12x).
- Honesty note: my pre-set thresholds passed on the first set of worlds but missed by a hair on a fresh set (for example 1.53 against a limit of 1.5). The improvement itself is clear; my thresholds were too tight to call it replicated.
- The shock level (how much bigger than usual counts as a shock) is a number I chose. Trying 2, 4 and 8, the setting 8 looked best, so I tested 8 on yet another fresh set of worlds that played no part in choosing it, and it passed.

In human terms: 'if this surprises me enormously, maybe I am wrong about what situation I am in' is worth a lot. Asking someone is still better.


## Appendix (iteration 64): which questions are worth asking

On a new set of worlds:

- The surprise gate alone (no questions) again landed at about 1.5x the perfectly taught version. Across four sets of worlds it sits steadily at 1.43-1.53x. My pre-set test missed by 0.01 on a ratio, so by my own rules it is still not 'replicated', even though the effect keeps showing up.
- Asking only when **shocked** used a third as many questions (2.2% of moments) as asking when **memories disagree** (6.9%), and got almost the same quality (1.15x vs 1.08x). A question asked at a moment of shock is worth about three times as much.
- Asking in either case reached 1.02x, essentially perfect, with questions on 7.4% of moments.

All three of my predictions failed by small margins. I keep them as failures, because moving the goalposts afterwards would make the whole register meaningless. The pattern is clear, though: surprise is the best signal for 'I might be in the wrong situation, check'.


## Appendix (iteration 65): how many questions, and a correction

I gave the learner budgets of 1, 2, 5 and 10 questions per 100 moments.

| questions per 100 moments | 0 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|
| ask where memories disagree | 1.48x | 1.40x | 1.35x | 1.21x | 1.07x |
| ask when shocked | 1.48x | 1.38x | 1.22x | 1.20x | 1.20x |
| either | 1.48x | 1.42x | 1.34x | 1.18x | **1.02x** |

(1.00x = as good as specialists that always got the right zone.)

**Correction.** Last time I said a shock-question is worth about three times a disagreement-question. That compared two learners that ended at different quality levels. With the same number of questions, shocks are only a little better (3-7%). And shocks are rare, only about 2 moments in 100, so after those are used up, disagreements are where the remaining value is.

The sensible rule is: first ask when shocked, then where memories disagree. With about 7-10 questions per 100 moments, that reaches essentially perfect specialists, and that result held on a fresh set of worlds.


## Appendix (iteration 66): a test that could not tell us anything

I tried the context tools in a different kind of world. There, the situation changes over **time** (the rules shift every 600 moments), not over **place**.

The problem: in that world, knowing the situation was worth only about 5%. A single learner that ignores the situation was almost as good as three specialists given the right situation every time, because the three situations were too similar. A test of 'how well do you recognise the situation' means little when recognising it barely matters. One of my predictions 'passed' only because it could not have failed.

That is my mistake. Before testing a fix, I should first check that the problem it fixes actually exists in that world (a 'manipulation check'). I made the same mistake once before, with the energy-cost experiments. I've now added it as a standing rule.

Two things still came out:

- Recognising the situation by 'similar-looking moments' worked a little better than guessing (52% against 33%), because the situations do leave some traces in what you see.
- Asking when shocked raised the share of time in the right situation from 85% to 98%, with very few questions.

Next I'll repeat the test with situations whose rules are truly different, and check first that knowing the situation matters.


## Appendix (iteration 67): the transfer test, done properly

This time I first checked that knowing the situation matters. With the world's built-in randomness switched on, it mattered too little (14%), so I did not score the rest. In a noise-free version (the default you asked for), on fresh worlds, it mattered by 26%, enough to test. I set that bar after a first look, and I say so.

| method | error (1.00 = always told the situation) | right situation |
|---|---|---|
| ignore the situation | 1.26 | n/a |
| keep the last hint | **1.55** | 86% |
| similar-looking memories | 1.90 | 49% |
| surprise gate | 1.29 | 91% |
| ask when shocked | **1.00** | 99.9% (asked on 3 in 1,000 moments) |

- **Wrong context is worse than no context.** Being in the wrong situation 14% of the time did more harm than ignoring situations completely. That is the same law as before (T8).
- **Asking when shocked carries over completely.** Even when situations change over time, a shock is the signal that the world changed, and one quick question fixes it.
- **'Similar-looking memories' does not carry over.** When the situation is about *when* rather than *where*, the moments look alike, so similarity cannot tell them apart. A recogniser that works for both kinds would need to remember which specialist explained the *last few moments* well. That is the next test.


## Appendix (iteration 68): two ways to recognise a situation

- **By what you see** ("this looks like the moments I remember from zone 3"). This works when the situation is a place: 93% right in the place world. It fails when the situation is a time period: 47%.
- **By what just worked** ("the specialist that has been right for the last few moments is probably still the right one"). This works when situations last a while: 1.17x the perfect version in the time world, better than keeping the last hint (1.58x). It fails in the place world (21%), because there the next moment can be anywhere.

Each one is a different assumption about the world: that situations show in appearances, or that situations persist. A real mind needs both, and has to learn which one to trust.

I tried a judge that trusts whichever recogniser agreed more often with the hints. In the place world it worked perfectly. In the time world it failed. There the 'what just worked' recogniser makes **good predictions**, yet often uses specialists under a different name than the hint gives (its specialists organised themselves). It was judged by names, not by results. Next I'll judge each recogniser by how well its chosen specialist actually predicted. That needs no hints at all.


## Appendix (iteration 69): trust whatever has been working

The new judge ignores the hints entirely. After every moment, it checks how well each recogniser's chosen specialist actually predicted. It then trusts whichever recogniser has been predicting better lately.

- In the place world it came within 2% of the best method, and in the time world within 1%. It did this without being told what kind of world it was in. This held on fresh worlds.
- The old judge ("trust whoever agrees with the teacher's labels") was 32% worse in the time world.
- One prediction failed: I expected the judge to use 'similar memories' at least 80% of the time in the place world, and it used it 74%. It made no practical difference, because where the two recognisers disagreed the cost was small.

This touches your 'calibration and skepticism' foundation. Trust should be earned by a track record of good predictions, not by agreeing with an authority. It is the same idea as the classic 'follow the expert who has been right' algorithms in machine learning.


## Appendix (iteration 70): a correction, scoring after seeing the answer

While building the next test I found a flaw in my own scoring code for the time-world experiments. At the moment a shock was detected, the learner switched specialists, and I then scored the **new** specialist's prediction. But the shock was only known after seeing the answer, so that was like changing your exam answer after the teacher read the solution out. A fair score uses the prediction the learner had already committed to.

Fixed and re-run: asking when shocked goes from 1.00x to **1.03x** the perfect version, and the surprise gate from 1.29x to 1.31x. The conclusions stand (asking when shocked still nearly matches perfect knowledge and still beats ignoring the situation), but the earlier numbers were a little flattered.


## Appendix (iteration 71): all the tools together, one setting

I put every context tool together with exactly the same settings in both kinds of world: memory recognition, recognition by what just worked, the judge that trusts whatever has been predicting better, the surprise gate, and questions (shocks first). Everything was tested on fresh worlds.

| | place world | time world |
|---|---|---|
| ignore the situation | n/a | 1.33x |
| recognition only | 2.60x | 1.17x |
| + surprise gate | 1.59x | 1.20x |
| + questions (all tools) | **1.20x** (asked 5 in 100) | **1.14x** (asked 2 in 100) |

(1.00x = always told the situation.)

- Two of three predictions passed. The third failed because in the time world the surprise gate and questions added only 5% on top of recognition. I predicted 10% or more.
- Being general has a cost. In the time world, the simpler 'keep the last hint, ask when shocked' gets 1.03x, better than the all-purpose stack's 1.14x. That is a classic trade-off: a tool that works everywhere is usually a bit worse than the tool made for one job.

Where this leaves the context problem: across place-like and time-like situations, the machine can now recognise its situation and get within 0-20% of perfect knowledge, asking only a few questions. Without any questions it gets within roughly 20-60%.


## Appendix (iteration 72): Kepler's law from NASA's own numbers

This was the first test on **real** data. I took NASA's published distances and orbit times of the 8 planets and gave the learner six candidate laws:

- time proportional to distance
- time proportional to distance^1.5 (Kepler)
- time proportional to distance^2
- a power with a free exponent
- a straight line
- an exponential

Nobody told it how precise the numbers were; it estimated that itself.

- **It found Kepler's law.** The best exponent was 1.498, and Kepler's 1.5 beat every wrong law overwhelmingly (odds of about e^40 to 1).
- **It predicted Pluto,** which it had not seen, to within 0.2%.
- **It carried the law over to Jupiter's four big moons.** Same exponent (1.504), different constant. Comparing the two constants gave the Sun as **1,047 times heavier than Jupiter**. The true value is 1,047.35. Newton turned Kepler's law into exactly this kind of mass measurement.
- **It poked a hole in the bad theory** 'one rule for all orbits, Sun and Jupiter alike' (rejected decisively).

**And it poked a hole in NASA's table.** The best exponent sat slightly but clearly below 1.5. The cause turned out to be the data: NASA's summary table lists the *tropical* year for Uranus and Neptune (measured against the seasons) but the *sidereal* one for Pluto (measured against the stars). Kepler's law is about the sidereal period. With the correct sidereal periods the exponent moved closer to 1.5 (1.499). That is exactly what 'poke holes' should mean: a residual that is too large points to something wrong, sometimes in the data.

Honest notes:

- I had a Saturn number in memory (10,759 days) that turned out to be wrong (10,756). I checked every value against NASA's pages before using it.
- With the corrected data, Pluto came out 1% off, because NASA's Pluto page gives a different distance than the summary table. My guess (unchecked) is that one is a snapshot of an orbit that Neptune keeps tugging.
- The near-perfect 1,047 was partly luck: the corrected data give 1,044.5 (0.3% off).
