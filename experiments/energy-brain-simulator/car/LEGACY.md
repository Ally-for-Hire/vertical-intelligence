# Legacy index (nothing here is deleted)

Kept for reproducibility; see RESEARCH_LOG.md for what each did and why it was superseded.
- v1 `agent.py`, `train.py`, `results.json`, `run.log`: Brain2 + evolved policy net. Still imported by `voter.py` (pd reference).
- v2 `neuro.py`: Boolean block graph, mutate-and-keep. Still imported (sensor bits).
- v3 `neurogrid.py`, `neurogrid_results.json`: wired grid with local reward rule. Failed (score ~0.12).
- `env.py`: shared car simulator, unchanged.
Run any of them with `python car/<file>.py` from the project root.
