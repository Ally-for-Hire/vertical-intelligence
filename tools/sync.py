"""Refresh this repo from the canonical working repos. One command, no arguments:

    python tools/sync.py

Copies every tracked or untracked-but-not-ignored file of each source (git ls-files -co --exclude-standard), minus
run debris (EXCLUDE). This repo is public, so every text file passes through REDACT (account names, family names,
local paths) and PRIVATE drafts are left out. Never deletes a result: a file that is in the snapshot but no longer in
its source is listed in SNAPSHOT.md as stale. Writes SNAPSHOT.md with each source's commit and uncommitted-file count.
Commit and push by hand, after reading SNAPSHOT.md.
"""
import json, os, subprocess, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
WIKI = HOME / 'Documents' / 'Knowledge-Wiki'
# (destination folder in this repo, source repo root, sub-path inside the source or '')
SOURCES = [
    ('curious-net', HOME / 'curious-net', ''),
    ('neuroblock-research', HOME / 'neuroblock-research', ''),
    ('program/wiki-domains/vertical-intelligence', WIKI, 'domains/vertical-intelligence'),
    ('program/wiki-domains/curious-net', WIKI, 'domains/curious-net'),
    ('program/wiki-domains/neuroblock-research', WIKI, 'domains/neuroblock-research'),
]
EXCLUDE_EXT = {'.err', '.out', '.pt', '.tmp', '.pyc'}           # run stderr/stdout dumps, resume checkpoints, temp files, bytecode
EXCLUDE_DIRS = {'__pycache__', 'wandb', 'data'}                  # caches; curious-net/data holds third-party raw data (not redistributed)
MAX_BYTES = 95 * 1024 * 1024                                     # GitHub rejects files over 100 MB
# This repo is public. The private terms (account names, family names) live in tools/redact.local.json, which is gitignored
# and never published: {"replace": [["term", "replacement"], ...], "private": ["path/in/this/repo", ...]}. The sync refuses to run without it.
LOCAL = Path(__file__).with_name('redact.local.json')
if not LOCAL.exists(): sys.exit(f'missing {LOCAL.name}: the private redaction list (see the comment above); refusing to publish unredacted files')
_cfg = json.loads(LOCAL.read_text(encoding='utf-8'))
_home = str(HOME)
REDACT = sorted([tuple(x) for x in _cfg['replace']] + [(_home.replace('\\', '\\\\'), '~'), (_home, '~'), (_home.replace('\\', '/'), '~'),
                 ('/' + _home[0].lower() + _home[2:].replace('\\', '/'), '~'),
                 (_home[:8] + _home[8:].replace('\\', '/'), '~')], key=lambda ab: -len(ab[0]))      # local paths become ~ (last entry: the mixed form drive:\Users/name that Windows tools sometimes print)
PRIVATE = set(_cfg['private'])                                   # drafts kept out of the public copy
TEXT_EXT = {'.md', '.py', '.txt', '.json', '.log', '.csv', '.html', '.js', '.ps1', '.stdout', '.gitignore', '.toml', '.yaml', '.yml', '.wb', '.done', ''}


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=True).stdout


def wanted(rel):
    parts = rel.split('/')
    return not (set(parts[:-1]) & EXCLUDE_DIRS) and os.path.splitext(rel)[1].lower() not in EXCLUDE_EXT


def public_bytes(src, rel):
    data = src.read_bytes()
    if os.path.splitext(rel)[1].lower() not in TEXT_EXT or b'\0' in data: return data, 0
    try: s = data.decode('utf-8')
    except UnicodeDecodeError: return data, 0
    n = 0
    for a, b in REDACT: n += s.count(a); s = s.replace(a, b)
    return s.encode('utf-8'), n


def sync_one(dest, repo, sub):
    files = [f for f in git(repo, 'ls-files', '-co', '--exclude-standard', '-z', '--', sub or '.').decode('utf-8').split('\0') if f]
    head = git(repo, 'rev-parse', '--short', 'HEAD').decode().strip()
    dirty = len([l for l in git(repo, 'status', '--porcelain', '--', sub or '.').decode('utf-8').splitlines() if l])
    out = ROOT / dest; seen, new, changed, big, redacted = set(), 0, 0, [], 0
    for f in files:
        rel = f[len(sub) + 1:] if sub else f
        if not wanted(rel) or f'{dest}/{rel}' in PRIVATE: continue
        src = repo / f
        if not src.is_file(): continue                           # deleted in the working tree but still tracked
        if src.stat().st_size > MAX_BYTES: big.append(rel); continue
        dst = out / rel; seen.add(rel); data, n = public_bytes(src, rel); redacted += n
        if dst.exists() and dst.read_bytes() == data: continue
        new += not dst.exists(); changed += dst.exists()
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
    for p in PRIVATE:                                            # a private draft never stays in the public copy (the canonical file is untouched)
        if p.startswith(dest + '/') and (ROOT / p).exists(): (ROOT / p).unlink(); seen.add(p[len(dest) + 1:])
    stale = sorted(str(p.relative_to(out)).replace('\\', '/') for p in out.rglob('*') if p.is_file()
                   and str(p.relative_to(out)).replace('\\', '/') not in seen and wanted(str(p.relative_to(out)).replace('\\', '/')))
    return dict(dest=dest, source=f'{repo.name}/{sub}' if sub else repo.name, head=head, dirty=dirty, new=new, changed=changed, big=big, stale=stale, redacted=redacted)


def main():
    rows = [sync_one(*s) for s in SOURCES]
    lines = ['# Snapshot', '', f'Refreshed {datetime.date.today().isoformat()} by `python tools/sync.py` from the canonical local repos (edit there, not here).', '',
             '| Folder here | Source | Source commit | Uncommitted files in source | New | Updated |', '|---|---|---|---|---|---|']
    lines += [f"| `{r['dest']}/` | {r['source']} | `{r['head']}` | {r['dirty']} | {r['new']} | {r['changed']} |" for r in rows]
    lines += ['', 'Uncommitted files are copied as they were on disk; runs that were live may be partial.',
              'Excluded: ' + ', '.join(sorted(EXCLUDE_EXT)) + ' files and ' + ', '.join(sorted(EXCLUDE_DIRS)) + '/ folders, and ' + str(len(PRIVATE)) + ' private draft(s).',
              f"Public copy: {sum(r['redacted'] for r in rows)} account names, family names and local paths replaced (local paths become `~`)."]
    for r in rows:
        if r['big']: lines += ['', f"Skipped (over 95 MB) in `{r['dest']}/`: " + ', '.join(f'`{b}`' for b in r['big'])]
        if r['stale']: lines += ['', f"Stale in `{r['dest']}/` (no longer in the source; kept, never deleted): " + ', '.join(f'`{s}`' for s in r['stale'])]
    (ROOT / 'SNAPSHOT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for r in rows: print(f"{r['dest']}: {r['new']} new, {r['changed']} updated, {len(r['stale'])} stale, {len(r['big'])} too big (source {r['head']}, {r['dirty']} uncommitted)")


if __name__ == '__main__':
    sys.exit(main())
