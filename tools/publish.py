"""Publish: refresh this PUBLIC snapshot, scan it, and (only with --push) commit and push it.

    python tools/publish.py                       # dry run: sync, scan, summary (nothing committed)
    python tools/publish.py --push "message"      # sync, scan, commit, push to the public remote's main

Safety, in order. It refuses to continue unless:
  1. the current branch is `public` and its upstream is `public/main`;
  2. the `public` remote is the vertical-intelligence repo and GitHub reports it as PUBLIC;
  3. the private redaction list (tools/redact.local.json, gitignored) exists and is NOT tracked;
  4. the scan of every new or changed file finds none of: a private term from the redaction list, a secret-looking
     value, an e-mail address (apart from an allow-list), a local Windows path, or a file over 95 MB.
It never force-pushes. A rejected push (someone else pushed) stops with a message; fix by hand.
The private terms are read at run time and never printed (findings show the file only).
Never push the `main` branch here: that branch tracks the old private archive and holds unredacted history."""
import json, os, re, subprocess, sys, time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
EXPECT_BRANCH, EXPECT_REMOTE, EXPECT_UPSTREAM = 'public', 'public', 'public/main'
EXPECT_REPO = 'Ally-for-Hire/vertical-intelligence'
MAX_BYTES = 95 * 1024 * 1024
TRAILER = 'Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>'
SECRETS = [re.compile(p) for p in (
    r'gh[pousr]_[A-Za-z0-9]{20,}', r'sk-[A-Za-z0-9]{20,}', r'-----BEGIN [A-Z ]*PRIVATE KEY-----', r'AKIA[0-9A-Z]{16}',
    r'(?i)(api[_-]?key|secret|passw(or)?d|auth[_-]?token|access[_-]?token)\s*[:=]\s*[\'"]?[A-Za-z0-9_\-]{16,}',
    r'(?i)wandb[^\n]{0,40}[0-9a-f]{40}')]
EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+')
EMAIL_OK = ('anthropic.com', 'example.com', 'users.noreply.github.com')
LOCAL_PATH = re.compile(r'(?i)(c:[\\/]+' + 'users[\\/]|/c/' + 'users/)')   # built in pieces so this file does not match itself


def git(*a, check=True):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f'git {" ".join(a)} failed: {r.stderr.strip()[:300]}')
    return r.stdout.strip()


def preflight():
    branch = git('branch', '--show-current')
    if branch != EXPECT_BRANCH: sys.exit(f'STOP: on branch {branch!r}, expected {EXPECT_BRANCH!r}')
    up = git('rev-parse', '--abbrev-ref', f'{EXPECT_BRANCH}@{{upstream}}', check=False)
    if up != EXPECT_UPSTREAM: sys.exit(f'STOP: upstream is {up!r}, expected {EXPECT_UPSTREAM!r}')
    url = git('remote', 'get-url', EXPECT_REMOTE)
    if EXPECT_REPO not in url: sys.exit(f'STOP: remote {EXPECT_REMOTE!r} is {url!r}, expected the {EXPECT_REPO} repo')
    try:
        vis = subprocess.run(['gh', 'repo', 'view', EXPECT_REPO, '--json', 'visibility', '-q', '.visibility'], capture_output=True, text=True, timeout=60).stdout.strip()
        if vis != 'PUBLIC': sys.exit(f'STOP: GitHub reports {EXPECT_REPO} as {vis!r}, expected PUBLIC')
    except (OSError, subprocess.TimeoutExpired):
        print('warning: gh unavailable, visibility not verified')
    local = TOOLS / 'redact.local.json'
    if not local.exists(): sys.exit('STOP: tools/redact.local.json is missing (the private redaction list)')
    if git('ls-files', 'tools/redact.local.json'): sys.exit('STOP: tools/redact.local.json is TRACKED; remove it from the index before publishing')
    return [a for a, _ in json.loads(local.read_text(encoding='utf-8'))['replace'] if len(a) >= 3]


def changed_files():
    out = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '-uall', '-z'], capture_output=True).stdout.decode('utf-8', 'replace')
    files = []
    for rec in out.split('\0'):
        if len(rec) > 3 and rec[:2].strip() not in ('D',): files.append(rec[3:])
    return files


def scan(files, terms):
    import sync   # reuses the text-extension list; importing runs no sync
    findings = []
    low = [t.lower() for t in terms]
    for rel in files:
        p = ROOT / rel
        if not p.is_file(): continue
        if any(t in rel.lower() for t in low): findings.append((rel, 'private term in the file name'))
        if p.stat().st_size > MAX_BYTES: findings.append((rel, 'over 95 MB')); continue
        if os.path.splitext(rel)[1].lower() not in sync.TEXT_EXT: continue
        try: s = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError): continue
        sl = s.lower()
        if any(t in sl for t in low): findings.append((rel, 'private term in the content'))
        if any(r.search(s) for r in SECRETS): findings.append((rel, 'secret-looking value'))
        if any(not m.group(0).lower().endswith(EMAIL_OK) for m in EMAIL.finditer(s)): findings.append((rel, 'e-mail address'))
        if LOCAL_PATH.search(s): findings.append((rel, 'local Windows path'))
    return findings


def main():
    args = sys.argv[1:]
    push = bool(args and args[0] == '--push')
    if push and len(args) < 2: sys.exit('usage: publish.py --push "commit message"')
    terms = preflight()
    subprocess.run([sys.executable, str(TOOLS / 'sync.py')], check=True)
    files = changed_files()
    findings = scan(files, terms)
    print(f'\n{len(files)} new or changed files; scan: {len(findings)} finding(s)')
    for rel, why in findings: print(f'  BLOCKED {rel}: {why}')
    if findings: sys.exit('STOP: fix the findings (edit the canonical source or add a redaction rule), then run again. Nothing was committed.')
    print(git('diff', '--stat', 'HEAD', '--', '.', ':!SNAPSHOT.md')[-600:] or '(no tracked-file changes)')
    if not push:
        print('dry run: nothing committed. To publish: python tools/publish.py --push "message"'); return
    git('add', '-A')
    if not git('diff', '--cached', '--name-only'): print('nothing to publish'); return
    git('commit', '-q', '-m', f'{args[1]}\n\n{TRAILER}')
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        r = subprocess.run(['git', '-C', str(ROOT), 'push', EXPECT_REMOTE, f'{EXPECT_BRANCH}:main'], capture_output=True, text=True)
        if r.returncode == 0: print('pushed:', git('log', '--oneline', '-1')); return
        if re.search(r'rejected|non-fast-forward|fetch first', r.stderr): sys.exit('STOP: push rejected (remote has new commits). Not forcing. Fetch and reconcile by hand.')
        print('push failed (network?), retrying:', r.stderr.strip()[:120])
    sys.exit('STOP: push failed after retries; the commit is local only')


if __name__ == '__main__':
    main()
