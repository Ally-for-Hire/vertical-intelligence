"""Zero-argument: print the newest cloud run's summary headline and the last story lines (from the Modal volume), and save the files to selfscore/runs/modal/<tag>/.
python selfscore/modal_peek.py [tag]"""
import os, sys, modal
vol = modal.Volume.from_name('neuroblock-results')
tags = sorted(e.path.strip('/') for e in vol.listdir('/')) if len(sys.argv) < 2 else [sys.argv[1]]
if not tags: print('no cloud runs yet'); sys.exit()
tag = tags[-1]; dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'modal', tag); os.makedirs(dest, exist_ok=True)
for e in vol.listdir(f'/{tag}'):
    name = os.path.basename(e.path)
    with open(os.path.join(dest, name), 'wb') as f:
        for chunk in vol.read_file(f'/{tag}/{name}'): f.write(chunk)
print(f'run {tag}  (saved to {dest})')
st = [x for x in os.listdir(dest) if x.endswith('.story.txt')]
if st: print('\n'.join(open(os.path.join(dest, st[0]), encoding='utf-8').read().splitlines()[-12:]))
sm = [x for x in os.listdir(dest) if x.endswith('_summary.md')]
if sm: print('\n' + '\n'.join(open(os.path.join(dest, sm[0]), encoding='utf-8').read().splitlines()[:6]))
