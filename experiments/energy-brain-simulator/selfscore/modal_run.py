"""Massive runs on Modal's cloud GPUs, detached from this PC (user, 2026-10-26: "use their computers for my massive runs").

Zero-argument launch (10,000 brains, 3,000 generations, one L4 GPU, 12 h cap):      modal run --detach selfscore/modal_run.py
Look at it:                                                                        python selfscore/modal_peek.py
Options (all optional):  --tag NAME  --gens N  --P N  --seed N  --extra "--cull_power 6 ..."  --wait   (--wait blocks until finished: for smoke tests)
Environment: NB_GPU (default L4, about $0.80/h; H100 is about $3.95/h), NB_HOURS (default 12 = hard cost cap).
Results land in the Modal volume 'neuroblock-results' under /<tag>/ (story, summary, json) and are committed every minute.
"""
import os, time
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU = os.environ.get('NB_GPU', 'L4'); HOURS = float(os.environ.get('NB_HOURS', '12'))
SKIP = ('/.git/', '__pycache__', '/selfscore/runs/', '/lab/', '/sweeps/', '/universe/', '/probnet/', '/wandb/', '/notes/', '/paper/')


def _skip(path):
    p = str(path).replace(chr(92), '/')
    if any(k in p for k in SKIP) or p.endswith(('.pyc', '.log', '.npz', '.stdout', '.out', '.err', '.png', '.zip')): return True
    try: return os.path.isfile(path) and os.path.getsize(path) > 3_000_000
    except OSError: return False


image = (modal.Image.debian_slim(python_version='3.11').pip_install('numpy', 'scipy', 'torch').add_local_dir(ROOT, '/root/nb', ignore=_skip))
app = modal.App('neuroblock-selfscore', image=image)
vol = modal.Volume.from_name('neuroblock-results', create_if_missing=True)


@app.function(gpu=GPU, volumes={'/results': vol}, timeout=int(HOURS * 3600))
def run_big(tag: str, args: list):
    import subprocess, threading
    out = f'/results/{tag}'; os.makedirs(out, exist_ok=True)
    env = dict(os.environ, NB_RUNS=out, PYTHONUNBUFFERED='1')
    stop = threading.Event()
    def committer():
        while not stop.wait(60):
            try: vol.commit()
            except Exception: pass
    threading.Thread(target=committer, daemon=True).start()
    p = subprocess.run(['python', '-u', 'selfscore/multi_big.py', '--tag', tag] + args, cwd='/root/nb', env=env)
    stop.set(); vol.commit()
    return p.returncode


@app.local_entrypoint()
def main(tag: str = '', gens: int = 3000, P: int = 10000, seed: int = 0, extra: str = '', wait: bool = False):
    tag = tag or time.strftime('refill%m%d_%H%M') + f'_s{seed}'
    args = ['--P', str(P), '--gens', str(gens), '--seed', str(seed)] + extra.split()
    print(f'launching {tag}: {P} brains x {gens} generations on {GPU} (cap {HOURS:.0f} h) -> volume neuroblock-results:/{tag}/')
    if wait: print('finished with return code', run_big.remote(tag, args))
    else: call = run_big.spawn(tag, args); print('spawned', call.object_id, '| it keeps running if you close this window; read it with: python selfscore/modal_peek.py')
