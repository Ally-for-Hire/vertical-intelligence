---
title: Contributor portal - a minimal website that lets key-holders queue experiments and ask about the research (spec, 2026-10-30)
domain: vertical-intelligence
kind: spec
tags: [portal, website, contributors, api-keys, queue, sponsorship]
updated: 2026-10-30
---

# Contributor portal (spec)

Read this if: you are building, running or changing `portal/` in the public repo clone
(`~/vertical-learning/portal/`). Author's request (2026-10-30): "a minimal website I can
give to people that gives them limited usage ... to run experiments and add to my research. It should
have literally everything there, automatically updated, and be git-backed via the main repo. If anyone
wants to contribute they need a key from me that I can generate and give ... queue experiments
primarily but also ask questions about the existing research if they don't have an AI subscription."

Five-line summary: one Python process (FastAPI) serving one HTML page. It reads the public repo on
disk (so "everything" is there and updates whenever the repo does), answers grounded questions with
Claude Sonnet over retrieved repo passages, and turns a contributor's experiment idea into a protocol-
shaped P-draft that is written into `contrib/queue/` and committed. Access is by per-person keys the
author mints from a CLI, each with a token budget. The model is paid for through the author's
**Anthropic API key** (Developer Platform), not a claude.ai subscription: consumer subscriptions are
single-user and may not be proxied to others; the API is the supported way, and it is what AI for
Science credits apply to.

## 1. Decisions (defaults; the author may change)
| Choice | Default | Why |
|---|---|---|
| Location | `portal/` at the root of the public clone, like `tools/` (renamed from `site/`, which shadows Python's stdlib module) (committed there, not synced from a canonical repo) | the site IS about the public repo; keeps one deployable unit |
| Stack | Python 3.11, FastAPI + uvicorn, `anthropic` SDK, sqlite3 (stdlib), vanilla HTML/JS (no build) | matches the repo's language; minimal |
| Model | `claude-sonnet-5-5` only, hard-coded allow-list | the author's choice; predictable cost |
| Billing | `ANTHROPIC_API_KEY` env var; budgets enforced in **tokens** (input + output), not dollars | token counts are exact from the API; prices change |
| Keys | minted by `python portal/keys.py new --name <who> --budget <tokens> [--daily <tokens>]`; shown once; stored as SHA-256 in `portal/data/portal.db` (gitignored) | the author controls who; no accounts, no passwords |
| Form factor | **phone first** (author, 2026-10-30): single column under 768 px, bottom tab bar, thumb-sized controls, installable web-app manifest (no service worker) | contributors will mostly use it from a phone |
| Hosting | now: the author's PC (where experiments run anyway), exposed with a Cloudflare Tunnel or similar; **later: a dedicated server** (same command under a process manager, `PORTAL_REPO` pointing at a clone); `0.0.0.0` only when a tunnel/proxy fronts it | no server to pay for yet; the repo is already there |
| Updates | the server re-indexes when the repo's `git rev-parse HEAD` changes (polled every 60 s); optional `PORTAL_GIT_PULL=1` runs `git pull --ff-only` on the same timer | "automatically updated" |
| Runs | the portal NEVER executes experiments. It queues them; the author (or the research session) registers and runs them under the protocol | compute and the register belong to the author |

## 2. What a key-holder can do
1. **Read everything.** Browse and search the repo's markdown (README, program pages, LOG, PREDICTIONS,
   QUEUE, PROVEN_PIECES, audits) rendered in the page.
2. **Ask.** Questions answered by Sonnet with the top retrieved passages as context, citing file paths;
   the system prompt forbids claims not in the passages and states the toy-world limits.
3. **Propose an experiment.** A form (idea, which finding-map section, what would count as success);
   Sonnet drafts it into the protocol's P-draft shape (question + U-theorem, arms incl. baseline and
   ceiling, metric/threshold/direction/seeds, manipulation check, budget, fairness notes). The
   contributor reviews and submits; the server writes `contrib/queue/<date>-<slug>.md` with
   `status: proposed`, the key's name and date, and commits it (`git add`/`git commit`; push only if
   `PORTAL_GIT_PUSH=1`).
4. **See the queue** and the status of their proposals (proposed / registered / running / done /
   declined, edited by the author in the file).
5. **See their budget** (used / total / daily).

## 3. Limits and safety
- Per-key token budget and daily cap; global daily cap (`PORTAL_GLOBAL_DAILY_TOKENS`); per-key rate
  limit (requests/minute); max prompt length; streaming responses.
- Keys revocable (`keys.py revoke`), listable (`keys.py list`), usage shown (`keys.py usage`).
- Proposals and questions are logged (sqlite) with key name; nothing from contributors is executed.
- No secrets in the repo: `portal/data/` and `portal/.env` gitignored; the publish scan still runs.
- The page states what the portal is, that answers are AI-generated from the repo, and the limits.

## 4. Files
```
portal/
  app.py          FastAPI app: routes, auth, budgets, retrieval, Claude calls, queue writer
  keys.py         CLI: new / list / revoke / usage
  index.py        markdown indexing (chunking + BM25), file listing, HEAD polling
  static/index.html  the single page (tabs: Read, Ask, Propose, Queue; phone-first, bottom nav)
  static/manifest.webmanifest, static/apple-touch-icon.png  installable web app
  README.md       run, mint keys, expose with a tunnel, env vars, costs
  requirements.txt
  tests/test_portal.py  auth, budget, queue file shape, retrieval smoke (no network)
contrib/queue/README.md  what a proposal file is and the status vocabulary
```

## 5. Definition of done
Tests pass offline (Claude calls mocked); `python portal/keys.py new` mints a key; the page loads; an
`ask` with a fake key is refused and with a real key returns a cited answer; a `propose` writes a file
in the right shape and commits; budgets decrement and refuse at zero; README explains setup in <= 15
lines; publish scan clean; this page and [[sponsorship-and-team]] (Move 4) updated.

## 6. Build notes (2026-10-30; built by two Sonnet sub-agents, integrated and tested by the coordinator)
- **Folder renamed `site/` -> `portal/`** after the build: `site` is shadowed by Python's stdlib module, so `python -m site.keys` and `uvicorn site.app:app` could not work. Now: `python portal/app.py` (or `python -m uvicorn portal.app:app` from the repo root), `python portal/keys.py new|list|revoke <id|name>|usage`, tests `python -m pytest portal/tests -q` (11 pass offline; model calls mocked).
- Key DB is `<repo>/portal/data/portal.db`; repo = `PORTAL_REPO` or the parent of `portal/`. Extra env vars beyond the spec: `PORTAL_REPO`, `PORTAL_HOST`, `PORTAL_PORT` (runner only; defaults 127.0.0.1:8765).
- `/api/ask` streams SSE: `event: sources` (paths), unnamed `data:` lines carrying JSON-string text chunks, `event: done` ({usage, remaining}), `event: error` ({message}), keep-alive comment every 10 s. The page's parser skips `sources`/`done` and shows `error` (fixed at integration: the first version appended the sources list to the answer).
- `/api/status` keys are `register_counts` and `latest_log`; the page accepts those and the spec's older names.
- Smoke test against the real repo (dummy API key): `/` 200, manifest 200 (`application/manifest+json`), 180 px `apple-touch-icon.png` 200 (iOS ignores SVG icons), `/api/status` 200 with 151 files / 1108 passages and the register line; `/api/me` without or with a bad key 401.
- **Not yet exercised:** a real Anthropic streaming call (written to the SDK's `messages.stream` / `text_stream` / `get_final_message().usage`), the Propose flow end to end in a browser, and the 402/429 paths in the page. First real run: mint a key, ask one question, draft and submit one proposal, check the commit.
- Frontend tested at 390 px (iframe) and 1200 px against a mock API; the file sheet, markdown rendering and streaming were not exercised in a browser.
