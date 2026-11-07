---
title: Vertical Intelligence - Sponsorship and team plan (the program's current priority)
domain: vertical-intelligence
kind: plan
tags: [sponsorship, funding, recruiting, team, pitch, public-repo, priority]
updated: 2026-10-30
---

# Sponsorship and team plan

Read this if: you are starting any session on this program after 2026-10-30. The author's standing
direction from that date is that **getting a sponsor (an advisor of record, compute, and later money)
and people to work on this is the program's first priority**; experiments continue, but every session
also moves one item on this page. The author already has a capstone project; this program needs its
OWN team and resources.

Five-line summary: funders fund momentum, not proposals, so the plan is (1) make the public face
unimpeachable, (2) ask the author's institution for advice before money (advisor of record, GPU time,
research credit), (3) recruit a team of 3-5 with defined first tasks, (4) put a preprint on arXiv, and
(5) stack funding from easiest to hardest, each application citing the previous. Status of every move
is tracked in section 6. People's names never go in this (public) page: they live in the private raw
note `raw/2026-10-30-sponsor-contacts.md`.

## 1. What we want, precisely

| Want | Smallest form | Full form |
|---|---|---|
| A sponsor | one faculty member who will be "advisor of record" and say so in an email | a PI who co-signs grant applications and co-authors the preprint |
| Compute | department GPU hours or a cloud-credit account on a faculty grant | one 24-48 GB GPU for the Self line's real-model stage (inbox I18 rungs 2-4) plus Anthropic AI for Science credits |
| People | two collaborators with a defined first task each (section 4) | a team of 3-5: experiments, adversarial review, theory/literature, ML systems, science data |
| Standing | research credit / independent study so the work is on the author's record | a funded position (REU-style stipend, then a grant line) |
| Money | reimbursed compute | a small grant (Emergent Ventures scale, low five figures) then a foundation grant with the PI |

## 2. What we have to offer (the pitch, in a sponsor's words)

- **A question a department can repeat upward:** what does a learner trained by gradient descent
  alone lack, and can each missing piece be isolated and proven in a fair test?
- **Method that is rare at any level:** 358 pre-registered predictions (169 passed, 171 failed,
  git-timestamped before the runs), 48-world paired bootstrap CIs, tuned baselines, manipulation checks,
  a published self-audit that voided our own results and redid two of them, every component earned
  alone before it is combined, and public Weights & Biases instrumentation in plain language.
- **Results** (toy worlds; stated as principles in miniature): see [[program-brief]] section 3.
- **A ladder with a final exam:** correlation to dependence to cause to functional form to anomaly to
  a missing constant to a principle ([[path-to-pre-quantum]]), ending in the pre-quantum test on real
  historical data.
- **A ready onboarding path:** everything is public and reproducible; the audit's redo list is a set of
  bounded, well-specified first tasks for new people ([[join]]).

## 3. The moves, in order

**Move 1 - Make the public face unimpeachable (before any email goes out).**
Fix the audit-flagged lines in the public README (C52 as R, "less is more" as a general claim, the old
evolution ratios); add a "For sponsors and collaborators" block at the top; refresh the at-a-glance
counts; add one figure for the prediction-only life result (X1.3/X1.4), which is the easiest to grasp
in ten seconds; make the W&B project public (author's call; it is a publishing action).

**Move 2 - Ask the institution for advice before money.** The author's capstone instructor (an
associate chair for undergraduate studies) is the router: the email asks (a) whether this can be
independent study or research credit, (b) which faculty fit, (c) whether department GPU time exists.
Draft: `raw/2026-10-30-capstone-instructor-email-draft.md` (private). Lead with results and a figure;
one sentence on the long goal; a specific meeting request. An earlier, narrower pitch for the
NeuroBlock line exists at `experiments/energy-brain-simulator/publication/PROPOSAL_UNT.md` (2026-10-23; private, not
in the public copy); its routes section is folded into the private contacts note.

**Move 3 - Find the sponsor before needing one.** A short list of faculty whose recent work touches
intrinsic motivation, open-ended or lifelong learning, ML for science, or bio-inspired computing, each
with the one sentence in their work our results speak to. Name them in the email so the router can
introduce rather than search. List: private raw note (section 0 above).

**Move 4 - Recruit the team now.** Start with people already near the work; then the department's ACM
and IEEE chapters and the capstone section. Every recruit gets a first task from [[join]] that is
finishable in two weeks and lands in the register under their name ([[credits]]).

**Move 5 - Preprint.** 8 pages on arXiv (cs.AI / cs.NE), working title "Pre-registered tests of what
gradient descent alone lacks: value, drive and viability in prediction-only agents"; the sponsor as
co-author if they want. Venues after: NeurIPS IMOL workshop (intrinsically motivated open-ended
learning), ALIFE, CogSci. The preprint changes how every funder reads the program.

**Move 6 - One result that is not a toy** (needs the author's go-ahead; it is an experiment). Ladder
rung 3 on real historical data: recover the Balmer/Rydberg relation from published 1880s line
measurements by search outside a fixed library, with a planted null. One real-data result beside
Kepler doubles the credibility of the ladder.

**Move 7 - Stack funding, easiest first, each citing the last.**
1. Institutional: research credit, undergraduate research awards, department GPU via the sponsor.
2. Cloud research credits on the sponsor's faculty account (faculty-only programs).
3. Anthropic AI for Science credits (form reviewed monthly; "researchers attached to a research
   institution"; focus on natural sciences and math): apply once Moves 1-3 are done, with the sponsor
   named. Not before (author, 2026-10-30).
4. Emergent Ventures (Mercatus): fast small grants for ambitious young people with unusual projects;
   verify current terms before applying.
5. Renaissance Philanthropy's Open Source for Science Fund; Schmidt Sciences: PI and preprint required;
   this is the money-and-team tier.

**Move 8 - Start the Self in earnest.** The Self is the author's main focus; the X7 line (2026-10-29/30)
is its first set of experiments. Keep it moving so the program's centre of gravity matches its pitch.

## 4. Team shape (roles, each with a first task)

Defined in [[join]]. Roles: experiment engineer; adversarial reviewer (statistics, attack lists);
theory and literature (cognitive science, neuroscience); ML systems (hooks on real models, GPU stage);
science data (historical measurement sets for the final exam); writing and figures. A role is filled
when its first task is in the register with the person's name.

## 5. Rules that still apply

- Nothing is contacted, submitted or pushed by the assistant: drafts only; the author sends
  ([[research-protocol]] section 4: contacting people and spending money are ask-first).
- People's names and institutions stay in private raw notes, never in synced pages (publishing rule).
- Every claim in a pitch traces to a register id; failures stay visible; toy results are "principles in
  miniature". The pitch is held to "Defend every result" like any experiment.
- The assistant drafts and tracks; the author decides who, when and what to send.

## 6. Status tracker (update every session)

| Move | Item | Status (date) |
|---|---|---|
| 1 | README flagged lines fixed; sponsor block added; counts refreshed | drafted 2026-10-30, push awaiting author |
| 1 | X1.3/X1.4 figure | done 2026-10-30 (`vi_x13_prediction_life`, in the README) |
| 1 | W&B project public | author's call, open |
| 2 | Email to the capstone instructor | draft v3 2026-10-30 (private raw; faculty names filled), not sent |
| 3 | Faculty fit list | done 2026-10-30: 6 candidates + routes in the private contacts note; recommended pairing a CSE continual-learning group (advisor) + a Physics complexity group (co-sponsor); author to check |
| 4 | Collaborator one-pager [[join]] | drafted 2026-10-30 |
| 4 | Contributor portal ([[contributor-portal]]): phone-first web app; key-holders read, ask, propose; queue in `contrib/queue/` | built 2026-10-30 (`portal/`, committed locally); needs the author's Anthropic API key, a first real-call test, a tunnel, and the push |
| 4 | First recruits | none yet |
| 5 | Program brief [[program-brief]] | drafted 2026-10-30 |
| 5 | Preprint | not started |
| 6 | Real-data rung 3 (Balmer) | not registered; needs the author's go-ahead |
| 7 | Funding applications | none; AI for Science deferred by the author (2026-10-30) |
| 8 | Self line (X7.x) | running (other session) |

## 7. Sources

- Anthropic AI for Science: anthropic.com/news/ai-for-science-program (eligibility, criteria,
  application form) and anthropic.com/news/ai-for-science-expands-foundations-research-institutions
  (partners, 2026). Team plan for scientists is PI-only (claude.com/programs/team-plan-for-scientists).
- The earlier course-project plan [[../../dynamic-model-growth/wiki/sponsorship-plan]] (2026-10-26),
  whose "a sponsor funds momentum, not a proposal" and "ask for advice before money" this page keeps.
