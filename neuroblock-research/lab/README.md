# The Lab

Design and run your own experiments on the neuron-block simulator from a web page, using as much of the computer as you allow.

* Open **http://127.0.0.1:8765/lab.html** (or "Open the Lab" on the live dashboard). The server starts at every logon
  (scheduled task "Claude Live Dashboard"). To restart it: `Stop-ScheduledTask 'Claude Live Dashboard'` (also stop the
  `pythonw ... serve.py` process), then `Start-ScheduledTask 'Claude Live Dashboard'`.
* Four experiment types: **Sweep** (vary settings one at a time), **Grid** (two settings together, colour map),
  **Compare** (design A vs design B, paired by seed, including two different brain designs), **Watch a brain** (replay one
  brain growing and tidying itself on a map).
* Every setting of the simulator is editable (see `lab_core.PARAMS`). Hover a setting for a plain-English explanation.
* **Your computer**: CPU threads (up to all of them), priority (below normal / normal / above normal, never real-time), and keep-awake
  while a job runs. Verified: 24 threads drive CPU to ~86-98% and the workers run at above-normal priority; cancel stops a job in
  seconds and leaves no workers behind.
* **Bookmarks and notes**: click ☆ on a result, a sweep chart, a grid cell or a moment in a brain replay. A bookmark keeps its own
  copy of the data, a title, tags and any number of notes (from you or from Claude). Everything is mirrored into
  `bookmarks/BOOKMARKS.md` so a session can read all your bookmarks and notes at once.

## Files

| File | What it is |
|---|---|
| `lab_core.py` | Parameter catalog, validation, the simulation of one lifetime, plugin loader |
| `lab_runner.py` | Runs one job (a folder with `spec.json`) on a process pool; `--plan` validates a spec |
| `lab_selftest.py` | Checks the engine matches the original simulator bit for bit, rejects bad input, runs one job of each type |
| `plugins/` | Your own brain designs and tasks (see `plugins/README.md`) |
| `bookmarks/` | One folder per bookmark (`bookmark.json` + data snapshot) and `BOOKMARKS.md` |
| `~/.claude/tools/live-dashboard/serve.py` | The local server and job queue (127.0.0.1 only) |
| `~\claude-live\lab\jobs\` | One folder per experiment: `spec.json`, `progress.json`, result files |

## Safety

* The server listens on 127.0.0.1 only and rejects other Host headers, foreign Origins, and writes without the `X-Lab` header.
* A job is **data only**: numbers, booleans and names, each validated against the catalog's min/max. The web page can never send code.
  New behaviour enters only through plugin files on disk.
* Limits: 300,000 runs per job, 2,000 repeats, 40 values per setting, 400 grid cells.
* At full power a laptop gets loud and warm. Plug in the charger. Use "Gentle" while working or gaming.

## Using it from Claude

Read `bookmarks/BOOKMARKS.md` to see what you have bookmarked and what you wanted to discuss. Claude can add a note to a bookmark
with `PUT /api/lab/bookmarks/<id>` body `{"add_note": {"text": "...", "by": "claude"}}` (with headers `X-Lab: 1` and
`Origin: http://127.0.0.1:8765`), or queue experiments with `POST /api/lab/jobs`.
