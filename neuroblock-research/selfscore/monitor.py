"""Read-only live status page for phone monitoring.

python selfscore/monitor.py [--port 8800]
Serves ONE self-refreshing mobile page and its JSON, at an unguessable path: http://<pc-ip>:<port>/<token>/
Safety: read-only (GET only, two files), refuses any client that is not on a private/loopback address, token stored in selfscore/monitor_token.txt.
Delete that file and restart to rotate the token. Stop with Ctrl+C or by killing the process. No data leaves the PC.
"""
import os, sys, re, json, time, glob, socket, secrets, threading, subprocess, ipaddress, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, 'runs'); CAR = os.path.join(ROOT, 'car')
LINE = re.compile(r'g\s*(\d+) test acc mean ([\d.]+) max ([\d.]+).*?\| C ([\d.]+)')
NAME = re.compile(r'(c\d+)_(\w+?)_(none|grow_and_pred|grow|pred|nonreg)_(nocull|ext|lin|step|Q|C|p\d\d|one)_s(\d)')
STATE = {'t': 0, 'data': {}}


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return ''


def gather():
    d = {'time': time.strftime('%H:%M:%S'), 'host': socket.gethostname()}
    g = sh('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits').split(',')
    if len(g) == 4: d['gpu'] = dict(util=g[0].strip(), mem=g[1].strip(), mem_total=g[2].strip(), temp=g[3].strip())
    d['python_procs'] = sum(1 for l in sh('tasklist').splitlines() if l.lower().startswith('python'))
    # grid progress
    grids = []
    for f in sorted(glob.glob(os.path.join(RUNS, 'grid_*.out'))):
        txt = open(f, errors='ignore').read().strip().splitlines()
        m = re.findall(r'\[(\d+)/(\d+)\] .*?\(([\d.]+) min\)', '\n'.join(txt))
        if m:
            done, tot, mins = int(m[-1][0]), int(m[-1][1]), float(m[-1][2])
            grids.append(dict(name=os.path.basename(f)[:-4], done=done, total=tot, minutes=mins, eta=(mins / done * (tot - done)) if done else None))
        else:
            grids.append(dict(name=os.path.basename(f)[:-4], done=0, total=0, minutes=0, eta=None))
    d['grids'] = grids
    # runs: finished (json) and running (log newer than 5 min without json)
    runs, board = [], {}
    for lg in glob.glob(os.path.join(RUNS, '*.log')):
        base = lg[:-4]; nm = os.path.basename(base)
        mt = os.path.getmtime(lg); fin = os.path.exists(base + '.json')
        last = ''
        try:
            lines = open(lg, errors='ignore').read().strip().splitlines(); last = lines[-1] if lines else ''
        except Exception: pass
        m = LINE.search(last)
        r = dict(name=nm, state='done' if fin else ('running' if time.time() - mt < 300 else 'stalled/failed'), age=int(time.time() - mt))
        if m: r.update(g=int(m.group(1)), acc=float(m.group(2)), accmax=float(m.group(3)), C=float(m.group(4)))
        if 'Traceback' in last or 'Error' in last: r['state'] = 'error'
        runs.append(r)
        n = NAME.match(nm)
        if n and m and fin:
            k = f'{n.group(3)} | cull={n.group(4)}'; board.setdefault(k, []).append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    runs.sort(key=lambda r: (r['state'] != 'running', r['age']))
    d['running'] = [r for r in runs if r['state'] == 'running'][:12]
    d['errors'] = [r['name'] for r in runs if r['state'] == 'error'][:8]
    d['n_done'] = sum(1 for r in runs if r['state'] == 'done')
    d['board'] = sorted(({'cfg': k, 'n': len(v), 'acc_mean': sum(x[0] for x in v) / len(v), 'acc_max': sum(x[1] for x in v) / len(v), 'C': sum(x[2] for x in v) / len(v)} for k, v in board.items()),
                        key=lambda r: -r['acc_mean'])[:30]
    # learning curves per problem and configuration, averaged over seeds (running and finished runs)
    curves = {}
    for lg in glob.glob(os.path.join(RUNS, 'c*_*.log')):
        n = NAME.match(os.path.basename(lg)[:-4])
        if not n: continue
        pts = []
        try:
            for ln in open(lg, errors='ignore').read().splitlines():
                m2 = LINE.search(ln)
                if m2: pts.append((int(m2.group(1)), float(m2.group(2)), float(m2.group(3)), float(m2.group(4))))
        except Exception: pass
        if pts: curves.setdefault(n.group(2), {}).setdefault(f'{n.group(3)}|{n.group(4)}', []).append(pts)
    out = {}
    for prob, cfgs in curves.items():
        out[prob] = {}
        for cfg, runs_ in cfgs.items():
            L = min(len(r) for r in runs_); k = max(1, L // 80)
            idx = list(range(0, L, k)) + ([L - 1] if (L - 1) % k else [])
            out[prob][cfg] = dict(g=[runs_[0][i][0] for i in idx], acc=[round(sum(r[i][1] for r in runs_) / len(runs_), 4) for i in idx],
                                  accmax=[round(sum(r[i][2] for r in runs_) / len(runs_), 4) for i in idx], C=[round(sum(r[i][3] for r in runs_) / len(runs_), 1) for i in idx], n=len(runs_))
    d['curves'] = out
    # plain-English story and at-a-glance summary of the newest random-cull run (multi.py)
    try:
        st = sorted(glob.glob(os.path.join(RUNS, 'multi_*.story.txt')), key=os.path.getmtime)
        if st: d['story'] = open(st[-1], encoding='utf-8', errors='ignore').read().splitlines()[-18:]
        sm = sorted(glob.glob(os.path.join(RUNS, 'multi_*_summary.md')), key=os.path.getmtime)
        if sm: d['summary'] = open(sm[-1], encoding='utf-8', errors='ignore').read()
    except Exception: pass
    # car sweep (older job)
    sw = glob.glob(os.path.join(CAR, 'stability_sweep_ms*_F_seed*.json'))
    if sw: d['car_sweep'] = dict(done=len(sw), total=72)
    return d


def loop():
    while True:
        try: STATE['data'] = gather(); STATE['t'] = time.time()
        except Exception as e: STATE['data'] = {'error': str(e)}
        time.sleep(5)


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>NeuroBlock live</title><style>
:root{color-scheme:dark}body{margin:0;background:#0d1117;color:#e6edf3;font:15px -apple-system,system-ui,sans-serif;padding:12px}
h1{font-size:17px;margin:0 0 8px}h2{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin:16px 0 6px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;margin-bottom:8px}
.bar{height:10px;background:#21262d;border-radius:6px;overflow:hidden;margin:6px 0}.bar>i{display:block;height:100%;background:#2f81f7}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;text-align:left;border-bottom:1px solid #21262d}th{color:#8b949e;font-weight:500}
.r{text-align:right;font-variant-numeric:tabular-nums}.ok{color:#3fb950}.bad{color:#f85149}.dim{color:#8b949e;font-size:12px}
</style></head><body><h1>NeuroBlock live <span class=dim id=t></span></h1><div id=app>loading…</div>
<script>
const p=location.pathname.replace(/\/?$/,'/');
async function tick(){try{const d=await (await fetch(p+'status.json?'+Date.now())).json();render(d)}catch(e){document.getElementById('t').textContent='offline'}}
function render(d){document.getElementById('t').textContent=d.time+' · '+(d.host||'');let h='';
if(d.gpu)h+=`<div class=card>GPU <b>${d.gpu.util}%</b> · ${d.gpu.mem}/${d.gpu.mem_total} MiB · ${d.gpu.temp}°C · python procs ${d.python_procs}</div>`;
for(const g of d.grids||[]){const f=g.total?100*g.done/g.total:0;h+=`<div class=card><b>${g.name}</b> ${g.done}/${g.total}<div class=bar><i style="width:${f}%"></i></div><span class=dim>${g.minutes.toFixed(0)} min elapsed${g.eta!=null?' · ~'+g.eta.toFixed(0)+' min left':''}</span></div>`}
if(d.car_sweep)h+=`<div class=card>car sweep ${d.car_sweep.done}/${d.car_sweep.total}<div class=bar><i style="width:${100*d.car_sweep.done/d.car_sweep.total}%"></i></div></div>`;
if((d.errors||[]).length)h+=`<div class="card bad">errors: ${d.errors.join(', ')}</div>`;
h+=(d.summary?`<h2>At a glance</h2><div class=card><pre style="white-space:pre-wrap;font-size:11px;margin:0">${d.summary.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</pre></div>`:'')+(d.story?`<h2>Story (latest lines)</h2><div class=card><pre style="white-space:pre-wrap;font-size:11px;margin:0">${d.story.join('
').replace(/&/g,'&amp;').replace(/</g,'&lt;')}</pre></div>`:'')+`<div id=charts></div><h2>Leaderboard (finished runs, mean over problems/seeds)</h2><div class=card><table><tr><th>config</th><th class=r>n</th><th class=r>acc</th><th class=r>max</th><th class=r>C</th></tr>`;
for(const r of d.board||[])h+=`<tr><td>${r.cfg}</td><td class=r>${r.n}</td><td class=r>${r.acc_mean.toFixed(3)}</td><td class=r>${r.acc_max.toFixed(3)}</td><td class=r>${r.C.toFixed(0)}</td></tr>`;
h+=`</table></div><h2>Running now</h2><div class=card><table>`;
for(const r of d.running||[])h+=`<tr><td>${r.name.replace(/^c\d+_/,'')}</td><td class=r>g${r.g??'-'}</td><td class=r>${r.acc!=null?r.acc.toFixed(3):''}</td><td class=r>C ${r.C??''}</td></tr>`;
h+=`</table></div><div class=dim>${d.n_done||0} runs finished</div>`;document.getElementById('app').innerHTML=h;charts(d.curves||{})}
const COL={none:'#8b949e',grow:'#2f81f7',grow_and_pred:'#f0883e'},DASH={nocull:[2,3],ext:[],lin:[8,4],step:[8,3,2,3],Q:[1,3],C:[4,2,1,2]};
function line(cv,series,key,logy,lo,hi){const dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;const c=cv.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);
let gx=1;for(const s of series)gx=Math.max(gx,s.g[s.g.length-1]);let ys=[];for(const s of series)ys.push(...s[key]);
if(lo==null){lo=Math.min(...ys);hi=Math.max(...ys);if(logy){lo=Math.max(lo,1)}}
const tr=v=>logy?Math.log(Math.max(v,1)):v,L=tr(lo),H=tr(hi)||1,pad=22;
c.strokeStyle='#21262d';c.lineWidth=1;c.fillStyle='#8b949e';c.font='10px sans-serif';
for(let i=0;i<=4;i++){const y=pad/2+(h-pad)*(1-i/4);c.beginPath();c.moveTo(28,y);c.lineTo(w,y);c.stroke();const v=logy?Math.exp(L+(H-L)*i/4):lo+(hi-lo)*i/4;c.fillText(logy?v.toFixed(0):v.toFixed(2),0,y+3)}
c.fillText('gen '+gx,w-48,h-2);
for(const s of series){const m=/^p(\d+)$/.exec(s.cull);if(m||s.cull==='one'){const f=s.cull==='one'?100:+m[1];c.strokeStyle=`hsl(${220-f*2.2},85%,60%)`;c.setLineDash(s.inner==='none'?[5,3]:[])}else{c.strokeStyle=COL[s.inner]||'#fff';c.setLineDash(DASH[s.cull]||[])}c.lineWidth=1.6;c.beginPath();
s[key].forEach((v,i)=>{const x=28+(w-30)*s.g[i]/gx,y=pad/2+(h-pad)*(1-(tr(v)-L)/((H-L)||1));i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()}c.setLineDash([])}
function charts(cs){const el=document.getElementById('charts');if(!el)return;let h='';for(const p of Object.keys(cs))h+=`<h2>${p}</h2><div class=card><div class=dim>best-individual accuracy (held-out) vs generation</div><canvas id=a_${p} style="width:100%;height:150px"></canvas><div class=dim>mean complexity (log)</div><canvas id=c_${p} style="width:100%;height:110px"></canvas></div>`;
h+=`<div class=card dim style="font-size:12px"><span style="color:#8b949e">grey</span> inner none · <span style="color:#2f81f7">blue</span> inner grow · <span style="color:#f0883e">orange</span> grow+pred · dashes: solid=ext cull, dashed=lin, dash-dot=step, dotted=nocull/Q, mixed=C. Cull sweep (c2): colour = cull % (blue 0 to red 99, magenta = single winner), solid = grow, dashed = no growth</div>`;
if(el.dataset.sig!==h){el.innerHTML=h;el.dataset.sig=h}
for(const p of Object.keys(cs)){const ser=Object.entries(cs[p]).map(([k,v])=>({inner:k.split('|')[0],cull:k.split('|')[1],...v}));line(document.getElementById('a_'+p),ser,'accmax',false,0.45,1.0);line(document.getElementById('c_'+p),ser,'C',true)}}
tick();setInterval(tick,5000);
</script></body></html>"""


def private(ip):
    try:
        a = ipaddress.ip_address(ip.split('%')[0]); return a.is_private or a.is_loopback
    except Exception:
        return False


class H(BaseHTTPRequestHandler):
    token = ''
    def log_message(s, *a): pass
    def do_GET(s):
        if not private(s.client_address[0]): s.send_error(403); return
        path = s.path.split('?')[0]
        if path == f'/{s.token}/' or path == f'/{s.token}': body, ct = PAGE.encode(), 'text/html; charset=utf-8'
        elif path == f'/{s.token}/status.json': body, ct = json.dumps(STATE['data']).encode(), 'application/json'
        else: s.send_error(404); return
        s.send_response(200); s.send_header('Content-Type', ct); s.send_header('Cache-Control', 'no-store'); s.send_header('X-Robots-Tag', 'noindex, nofollow'); s.send_header('Content-Length', str(len(body))); s.end_headers(); s.wfile.write(body)
    do_HEAD = do_GET


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8800); ap.add_argument('--bind', default='127.0.0.1', help='127.0.0.1 = this PC only (the tunnel connects here)'); a = ap.parse_args()
    tf = os.path.join(HERE, 'monitor_token.txt')
    if not os.path.exists(tf): open(tf, 'w').write(secrets.token_urlsafe(12))
    H.token = open(tf).read().strip()
    threading.Thread(target=loop, daemon=True).start()
    print(f'serving read-only status page on {a.bind}:{a.port} at /{H.token}/', flush=True)
    ThreadingHTTPServer((a.bind, a.port), H).serve_forever()
