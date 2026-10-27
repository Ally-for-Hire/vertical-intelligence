---
title: Vertical Intelligence - Research Protocol (program charter and operating manual)
domain: vertical-intelligence
kind: protocol
tags: [research-program, protocol, vertical-intelligence, biology, psychology, documentation, requirements]
updated: 2026-10-26
---

# Vertical Intelligence - Research Protocol

Read this if: you are starting any session on the user's brain/intelligence research
(neuroblock, curious-net, dynamic-model-growth, or a new line). It is the program's charter and
its operating manual. The user's statements are givens. Everything marked (assistant) is
interpretation and can be overruled by the user, never silently.

Five-line summary: the program tries to prove "vertical intelligence" (a learner that gets
better at producing information from information, knows its limits, and grows parts as needed)
by combining biology, psychology and computer science. It is at the foundational stage: many
small, pre-registered experiments on toy worlds, 149 registered predictions so far across two
code lines. Givens U1-U19; agenda Q1-Q10 (Q10 is the current direction). The user supplies theorems, tests and ideas through [[inbox]]; the assistant does
everything else under section 4's autonomy grant, to the requirements in section 5, and never
deletes a result.

## 0. How the user works with this

The user drops **theorems, tests, and ideas** into [[domains/vertical-intelligence/wiki/inbox]]
(or says them in chat; the assistant files them). Nothing else is required of the user. The
assistant intakes each item (section 3), runs it to the requirements (section 5), keeps the
queue moving (section 6), and reports outcome first, evidence second, in plain language.

Naming: user theorems are **U1, U2, ...** (the curious-net register already uses T7, T8 for
*derived* theorems, so T-numbers stay for those). Tests are **P-numbers** in the register.
Ideas are **I-numbers** in the inbox. Pieces keep their existing C/CTX/B/E ids.

## 1. The givens (user, 2026-10-26; verbatim in `raw/2026-10-26-three-theorems.md`)

- **U1. The universe is random. We came from random.**
- **U2. The human brain is the bare minimum processing machine to take over the world. It is
  not necessarily a direct model of how to think, but it is by far the best one we have.**
- **U3. Transformer models are stupid. Anything trained on its own, on a goal, and not on
  entirely self-driven data will inevitably become a single-minded machine.**
- **U4. The brain is divided into sub-parts: the Self, the Sensors, the Emotions. These all
  work in harmony after billions of years of evolution.** (2026-10-26; verbatim in
  `raw/2026-10-26-harmony-theorems-and-questions.md`)
- **U5. The brain throws nothing away; everything works in harmony.** (2026-10-26)
- **U6. Intelligence is the processing and ability to use randomness to your best benefit.**
- **U7. Every part of the brain has intrinsic weights and can take over; emotions are a raw
  reaction before intelligence; the IFS end goal is the Self holding everything as parts.**
- **U8. Simulation inside the mind, and the drive to improve that simulation, are
  foundational.** (U6-U8 from Dialogue 1, 2026-10-26; verbatim in
  `raw/2026-10-26-dialogue-summary.md` (public summary; verbatim kept private); readings and evidence in
  [[dialogue-prediction-is-not-the-whole-story]] s3)
- **U9. Boredom is the want to learn plus the inability to learn; the need to learn is a
  requirement, not a choice.**
- **U10. Two needs fight, to be more and to be all of yourself; being all of yourself is the
  priority, and wholeness multiplies growth.**
- **U11. A trauma part is an ungrown snapshot; it must be fully expressed before it can learn.**
- **U12. The drive is not uncertainty reduction; certainty past a point is harmful.** (U9-U12
  from Dialogue 1 part 2, verbatim in the same raw file; readings in the dialogue page s5)
- **U13. Some aspect of a biological substrate, simulated or not, is part of the solution; the
  missing link is in the representation of knowledge, not the knowledge.** **U14. Time is not
  required for intelligence.** **U15. Genius is drive keeping a problem alive past the wall.**
  **U16. Intuition is a pushed-aside part's warning; it grows if ignored, dims, never stops.**
  **U17. Comfort and growth are one hunger, grown from one 'better or worse for me' signal.**
  (Part 3; dialogue page s6.) **Direction (user, 2026-10-26): Q10, what is the 'for me', is
  the path forward for this research.**
- **U18. Gradient descent is not bad; it is probably the fastest learner and part of the end
  goal. What is vital is something that manages it; it should be intrinsic to everything, but it
  is not the whole answer.** (Part 4.) Consequence: every agent in [[experiment-plan]] may use
  gradient descent inside; the test is what manages it.
- **U19. The Self is formed, not innate: the ancient 'better or worse for me' signal is the seed,
  the child mimics its surroundings, a Self forms and then becomes something greater, not yet named.
  Emotions and intellect are sensor-level tools whose outputs the Self takes as information.** (From
  a conversation with a practicing psychotherapist, recorded with consent for this research (2026-10-27; notes private).)
- **Framing (user, 2026-10-27):** "primarily a deep dive into evolution and the nature of
  intelligence with the express goal of applying it to artificial intelligence", meaning machine
  learning (verbatim: `raw/2026-10-27-program-framing.md`).
- Goal: **prove vertical intelligence** and build it from biology, psychology and good
  computer science. Stage: foundational; running experiments; trying different models of
  thinking; using what is known to work and adding the user's own insight.

Earlier givens still in force unless the user retracts them: the Dream
([[domains/curious-net/wiki/the-dream]]), the end-goal vision
([[domains/curious-net/wiki/vision-and-roadmap]]), the foundations in
`neuroblock-research/FOUNDATIONS.md`. Final exam (user, Oct 22): a model given only
pre-quantum science proposes that quantum theory must exist.

### Testable readings (assistant; for the user to correct)

| Theorem | Testable reading | Evidence so far (curious-net register) |
|---|---|---|
| U1 random origin | Structure must be *discovered* from noise, not assumed; the learner needs a noise floor it refuses to learn below; randomness supplies variants, not reasoning | Randomness as a mechanism was null (C17); curiosity's proven job is avoiding the unlearnable (C40). Randomness as *origin* untested |
| U2 brain as floor, not blueprint | Copy the brain's constraints (energy, sparsity, parts, sleep, delayed credit) but test each; drop any that does not pay | Energy/wiring budgets pay (C41, R); sparsity wins three times (C41/C47/C50); sleep replay null unless waking learning saturates (C54-56); consolidation fights relearning (C49, N) |
| U3 goal-trained models go single-minded | A learner trained only on a fixed objective loses calibration outside it; self-driven data (own experiments, questions) keeps it honest | Student taught only truth is confidently wrong outside it (C31, S); targeted questions cut error 44-73% (CTX6-7, R); motivation + curiosity beats either alone (C52, R). No head-to-head goal-only vs self-driven test yet |
| U4 Self / Sensors / Emotions in harmony | (a) Each part is necessary: removing any one hurts. (b) Harmony = **superadditive**: the joint system gains more than the sum of each part's gain alone. (c) No part is graded on its own goal; all share one score, else they compete (link to U3) | Correct parts are 20x better than one net, but self-assignment captures little (C44, D). Motivation + curiosity (an emotion-like pair) beats either alone (C52, R). Several mechanisms only work in pairs (C36 forget + boost, P). Full context stack (CTX13) is a part ensemble but was never ablated as Self/Sensors/Emotions. The Self and the emotional converter have no direct experiments |
| U5 nothing is thrown away | (a) Experience is never deleted: an archive keeps everything; what changes is what is *active*. (b) An archive-everything learner protects returning knowledge with no loss on new learning, if retrieval is selective. (c) Falsified if a store-everything arm loses to a forgetting arm at equal compute | Replay of old experience protects returning knowledge (-12%) but costs ~5% new learning (C35, P). **Tension:** the wiring budget that prunes to 5% of weight mass *helps* (C41, R), and locking weights fights relearning (C49, N). Reading that keeps U5: prune *connections*, keep *experience* (assistant) |

### The main questions (user, 2026-10-26; Q-numbers, tracked in [[inbox]])

The questions are the program's current research agenda. Every test should say which Q it
moves. Assistant notes on each (what the evidence already says, a handle to attack it):

- **Q1. How does everything work together in harmony?** Handle: harmony as superadditive
  interaction plus shared scoring (U4 reading). First test: ablation lattice over the three
  parts (P-draft 4). Existing hint: paired mechanisms (C36, C52) and "judge by consequences"
  (CTX12b) are small cases of parts cooperating under one score.
- **Q2. What do you grade it on?** and **Q3. What scoring produces vertical intelligence?**
  Same question at two levels: the *external* grade (how we know it is vertically intelligent)
  and the *internal* score (what the system itself optimises). Candidate external grade: the
  definition below. Candidate internal score (proposed in neuroblock, never tested): **learning
  progress per unit energy**. U3 predicts that a fixed external goal used as the internal score
  produces a single-minded machine, so Q3 is the core test of U3 (P-draft 1).
- **Q4. How did we go from inert rocks to sensor devices to thinking machines?** Ties to U1.
  Handle: a toy open-ended world where nothing is designed in, only replication + cost +
  random variation, and see whether sensing and then prediction appear. Prior art to read:
  artificial-life systems (Tierra, Avida) [u]. Largest scope of the six; parked as I4.
- **Q5. What does it mean to be intelligent?** Current working answer (the user's, Oct 22):
  getting better at taking information and producing more information, not at specific
  problems. Q5 is answered when the definition below is confirmed and measurable.
- **Q6. What are the base foundations of intelligence and how can we use them?** Current
  candidates from the register (hypothesis level): sparsity, context inference, paired
  mechanisms, transfer proportional to shared structure, check when unsure (see
  [[domains/curious-net/wiki/synthesis]]). Each "use" is a transformer or architecture change
  tested against a plain baseline with zero-experience parity.

**Definition of vertical intelligence (assistant, proposed; user to confirm):** each new
problem is solved faster and with less energy by connecting it to all previous problems, and the
system knows the edge of what it understands. Measured as error vs problem count (not time),
energy per solved problem, calibration (confident-wrong rate), and transfer proportional to
shared structure (C43 already shows this dose-response).

## 2. Map: where everything lives (one home per kind of record)

| Kind of record | Home | Rule |
|---|---|---|
| The user's words (theorems, redirects) | `domains/<line>/raw/YYYY-MM-DD-*.md` | Verbatim, append-only, dated. Personal reflections: verbatim only in the private wiki; a summary goes in the public tree ([[publishing]] section 5) |
| Intake of new theorems / tests / ideas | [[domains/vertical-intelligence/wiki/inbox]] | Processed top to bottom; status column |
| Open research questions (Q-numbers) | inbox Questions table; notes in section 1 here | A question closes only with ids of the results that answer it |
| Code and experiment scripts | `~/curious-net` (current), `~/neuroblock-research` (older), course repo for growth | One script per experiment, named after its id |
| Per-iteration narrative | repo `LOG.md` (newest first) | One entry per iteration: id, script, seeds, numbers, verdicts, reading, limits, literature ties |
| Pre-registered predictions | repo `paper/PREDICTIONS.md` (P-numbers) | Written BEFORE the run; verdict appended after |
| What to run next | repo `paper/QUEUE.md` | Ordered; each item names the question and the U-theorem it serves |
| Pieces and their status | repo `paper/PIECES.md`; wiki [[domains/curious-net/wiki/pieces-inventory]] | Confidence letter per piece |
| Durable synthesis | [[domains/curious-net/wiki/synthesis]] and this page | Current-state, compact, cites ids |
| Cross-session timeline | `Knowledge-Wiki/log.md` | One entry per session or iteration |
| Literature | repo `paper/LITERATURE.md`; wiki related-work pages | Mark `[u]` if not read in full |
| Independent review | repo `paper/FABLE_REVIEW.md` | Corrections go in LOG as CORRECTION entries |
| Published snapshot (for other agents and readers) | **Public** GitHub repo `Ally-for-Hire/vertical-intelligence` (branch `main`), local clone `~/vertical-learning` (branch `public`; the older private repo `Ally-for-Hire/vertical-learning` keeps the unredacted history); human entry `README.md`, agent entry `AGENTS.md` | Never edited directly: edit the canonical repos and wiki, then `python tools/sync.py` (copies, never deletes, redacts private terms via the gitignored `tools/redact.local.json`, writes `SNAPSHOT.md`); then `python tools/publish.py` (dry run) and `--push "message"` (standing grant, see [[publishing]]). Anything written in the canonical repos can become public: no secrets, account names or third-party drafts outside the redaction list |

Do not create a second home for any of these. If a record does not fit, add a row here first.

## 3. Intake requirements (what happens when the user gives something new)

**A new theorem (U-n).** Within the same session:
1. Capture verbatim in `raw/` with date. Assign the next U-number.
2. Write 1-3 testable readings (what would be observed if true; what would falsify it).
3. Check it against every existing given and every R/S result; list agreements, conflicts, and
   "already supported by" with ids. Conflicts go in section 7, never resolved silently.
4. Propose the first cheapest test (one line: arms, metric, world). Add to the inbox as a P-draft.
5. Update the table in section 1. Log line. Commit.

**A new test (P-draft).** Before running:
1. One-sentence question and the U-theorem (or parked idea) it serves.
2. Arms, including a plain baseline with exact parity at zero experience, and an oracle/ceiling
   arm where one exists.
3. Metric, threshold, direction, and seeds, written down before any run. A manipulation check
   that the varied factor matters in this world.
4. Budget: expected wall time and hardware. Toy-world default: CPU, <= 10 min per arm,
   >= 48 worlds or seeds. GPU default: <= 30 min unattended on the laptop (8 GB).
5. Then it is a registered P-number and may run.

**A new idea (I-n).** Capture in the inbox with: one line, which U-theorem it touches, "worth
testing if ..." (the condition under which it would be pulled), and a cost guess. Ideas are
parked by default; they are promoted to tests only via the P-draft requirements. Nothing is lost
and nothing jumps the queue without a reason written down.

**Default ordering of the queue** (highest first): the user's explicit "do this next"; a test of
a U-theorem that has no evidence row; replication of an S result on fresh seeds (to reach R);
a D piece with a new diagnosis to try; parked ideas whose "worth testing if" condition has been
met; everything else.

## 4. Autonomy grant (do without asking) and stop lines (ask first)

**Do without asking:** run registered toy-world experiments within section 3 budgets; add or
reorder queue items under the default ordering; write and commit wiki pages, LOG entries,
predictions and verdicts; commit code to the research repos; read literature and file it;
replicate S results; write CORRECTION entries; park ideas; pick defaults from section 6.

**Ask first:** any push or public GitHub action other than the standing public-repo publish
(below); any job over the budget defaults or that
needs the user's GPU for more than 30 min; anything that steals the user's screen or focus;
changing, merging or dropping a given; changing the definition of vertical intelligence;
spending money or credits; adding to the course repo's `main`; contacting people; deleting any
file that holds a result.

**Compute (user, 2026-10-27, supersedes the one-thread default and the 30-minute GPU ask-first):**
all machine-usage restrictions are lifted; use as many threads and runs as useful, but keep a 15%
margin so the PC does not crash: total CPU <= 85%, free RAM >= 15% of total, GPU memory <= 85%
(including the desktop's ~11%; NeuroBlock `NB_GPU_FRAC` <= 0.7). `curious-net/vi/budget.py`
computes the worker count and GPU fraction from the live load. The limits on pushing, public
actions, money and focus-stealing (srcds) are unchanged.

**Publishing (user, 2026-10-27, standing):** keep the public repo actively updated with progress.
After each recorded result (or a coherent batch), run `python tools/publish.py` in the public clone
(a dry run), then `--push "message"`. That pushes only to the public remote's `main`, after hard
checks (branch, remote, PUBLIC visibility, and a scan for private terms, secrets, e-mails, local
paths). Still ask first: force pushes or history rewrites, changing a repo's visibility, creating
or deleting repos or branches, any push to the old private archive, and anything that would put
personal or third-party material into the public tree. Procedure and quality bar: [[publishing]].

**Reporting.** Each session ends with: outcome in one paragraph, verdicts with ids, what
changed in belief (or "nothing"), the one question for the user if there is one, and the next
queue head. No checklists narrated.

## 5. Requirements (definition of done) by kind of work

**Foundation (theorem, principle, definition).** Verbatim source in `raw/`; U-number; testable
readings; conflict check against all givens and R/S results; first test proposed; section 1
table updated; log line; committed.

**Experiment.** Registered P-number before running; fresh seeds; manipulation check passed
(else VOID, recorded); baseline with zero-experience parity; held-out ground truth the loop
never trained on; scored as committed (never after seeing the answer); absolute levels and
ratios both reported; medians across worlds plus spread; verdict per clause; confidence letter;
LOG entry with numbers, reading, limits, literature ties; PIECES updated; synthesis updated
only if a belief changed; wiki log line; script named after the id and reproducible by one
command; no hand-tuned constants without a derivation or a measured sweep; committed.

**Code and infrastructure.** Smallest change that serves a registered test; one zero-arg (or
one-flag) command to reproduce; output written to `results/` with the id in the filename;
seeds explicit; a bit-exact parity check against the previous version where behaviour must not
change; documented in the repo README in <= 5 lines; committed.

**Literature.** Paper, year, venue, claim in one line, what it changes for us, `[u]` if not
read in full; filed in `LITERATURE.md` or a related-work page; speedups from large models never
quoted as targets.

**Synthesis / belief update.** Cites the ids that changed it; keeps the confidence letter
scale; marks what was retracted and why; stays under ~30 KB per page (split by pillar above
that); starts with "Read this if" and a five-line summary.

**Course deliverables (dynamic-model-growth).** Course scope rules win inside that repo (keep
`main` minimal, per-area branches). Results that bear on the program are copied into this
domain's pages with ids, not left only in the course repo.

**Every piece of work.** Failures, voids, design flaws and corrections are kept and labelled;
a mistaken number is fixed by a CORRECTION entry, not an edit. Remembered constants are not
used: fetch or measure. Toy-world results are described as principles in miniature, never as
facts about brains or LLMs.

## 6. Speed rules (keeping it moving and on task)

- **Principles, not metric chasing (user, 2026-10-26).** Every test names its section in
  [[findings-map]] and the gradient-descent disconnect it probes. A stalled number means change
  the question, not the knobs.

- **Time boxes.** Exploratory look <= 30 min, then register or park. One registered run per
  iteration. A session with no run ends by registering the next one.
- **Three strikes.** Three consecutive FAILED verdicts on one piece with no new reading: write
  the D diagnosis, park it with the condition that would unpark it, move on.
- **Thresholds are set once.** A borderline fail (CTX9/9c) is reported as S, not re-run with a
  looser threshold. A new threshold means a new P on new seeds.
- **Defaults, not questions.** When a choice is not load-bearing, pick the default and state
  it: 48 worlds; median plus IQR; noise-free first then noise 0.05; seeds from a fresh block of
  100 per iteration; CPU before GPU; smallest world that still passes the manipulation check.
- **One question per session**, at most, and only when the answer changes what runs next.
- **No re-litigating.** Decisions in section 1 and section 7 stand until the user changes them.
- **Batch the bookkeeping.** LOG, PREDICTIONS, PIECES, wiki log and commit happen once at the
  end of the iteration, not after every number.
- **Queue head always filled.** The last act of any session is writing the next item at the
  top of `QUEUE.md` and the "Next up" line at the top of the inbox, so the next session starts
  by running, not by deciding.
- **Scope guard.** Anything that is not a registered test, a required record, or an intake
  from the user is out of scope for the session. Interesting tangents become I-numbers.
- **Weekly compaction.** Once a week (or every ~10 iterations): recount the register by
  script, compact the synthesis table, archive log above 150 KB, confirm every S piece has a
  replication queued.

## 7. Open conflicts and drift (assistant; needs the user's call)

1. **Transformer or not.** The Dream (Oct 22) was "a better transformer". U3 (Oct 26) says
   goal-trained transformers go single-minded. Proposed reconciliation: the transformer is a
   component (a converter), not the Self; the program's claim is about the *training signal*
   (self-driven data) more than the architecture. Needs the user's confirmation.
2. **The Self has zero experiments** despite being the stated main focus. Context inference is
   the closest proxy and is at D/S. Candidate: Self v0 as a controller choosing among explore /
   consolidate / sleep / experiment / ask, paid in energy.
3. **Two halves never joined.** NeuroBlock has the energy-limited brain; curious-net has
   generality, skepticism and the teacher. No single system has both.
4. **Randomness.** U1 says random origin; C17 says randomness as mechanism is null. Compatible
   (origin vs mechanism) but the program should state which it is testing. **Sharpened by U6
   (2026-10-26):** U6 is a mechanism claim (randomness *used well*). Proposed reconciliation:
   C17 tested undirected randomness; U6 is about directed variation plus selection (P-draft 7).
5. **Model growth line.** Maps onto "parts generated on demand" but runs under course scope.
   Decide whether its results feed the program or stay a course deliverable.
6. **Which parts?** U4 names three: Self, Sensors, Emotions. The Oct 22 givens had a fourth,
   the **intellectual converter** (inputs to followable context graphs; makes connections the
   system did not have), and "at least five" parts. Is the intellectual side part of the
   Sensors, part of the Self, or dropped? Also: the Self "has no parts" but generates them on
   demand; are Sensors and Emotions generated by the Self or given at birth?
   **Evidence (2026-10-27, a conversation with a practicing psychotherapist, recorded with consent for this research (2026-10-27; notes private)):** emotions are a part of bodily existence whose output is
   information the Self takes as input (a sensor), and the intellect is likewise a tool or sensor,
   separable from the Self (a small Self can sit with a large intellect). Reading: the
   intellectual converter is sensor-level, not part of the Self; Sensors and Emotions are given at
   birth, the Self is formed later (U19). Still the user's call.
7. **Nothing thrown away vs energy and pruning.** U5 vs the Oct 22 given "store as little as
   possible and reuse it" and vs C41 (pruning to 5% of weight mass helps). Proposed
   reconciliation: experience is never discarded; active wiring is pruned; storage is cheap,
   activation is what costs energy. Needs the user's call.
8. **Uncertainty reduction as the drive (U12 vs Oct 22 design rule).** curious-net's first
   design rule was "uncertainty reduction is the drive". U12 (2026-10-26) says it is not.
   Proposed reconciliation: the drive is learning progress (reducing *reducible* uncertainty,
   staying at the edge), which already matches C40. Pure reduction would sit in the dark room.
   Needs the user's call; P-draft 9 tests it.

## 8. Session checklists

**Start.** Read this page, then the inbox "Next up" line, then the line's domain index, the
repo `LOG.md` top entry and `QUEUE.md` head. State in one line what will be tested and which
U-theorem it serves. If the inbox has unprocessed items, intake them first (section 3).

**End.** Section 5 requirements met for everything touched; inbox statuses updated; queue
head and "Next up" written; wiki and repo committed; if a result was recorded, publish per [[publishing]] (the public repo is pushed under the standing grant; nothing else without asking); report per
section 4. Reading-only sessions log as `Wiki: log-only`.
