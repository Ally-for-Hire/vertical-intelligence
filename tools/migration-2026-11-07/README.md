# Migration record (2026-11-07)

How the former `curious-net` and `neuroblock-research` repos became this one. `mapping.py` is the old-path -> new-path rule
(`dest(repo, rel)`); `move.py` did the `git mv` and carried untracked files; `patch.py` fixed code paths (bootstrap into `core/vipaths.py`);
`docpaths.py` translated paths inside docs; `famreadme.py` generated the family READMEs. Kept for provenance and for translating old paths
found in older text (`from mapping import dest`). Absolute paths inside the scripts are the author's machine. Not meant to be re-run.
