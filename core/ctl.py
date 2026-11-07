"""Run control for long experiments: python core/ctl.py status|ping|stop|start|restart [name]   (name = x1_29, x1_28b, ...; default: all running)
status/ping: processes, CPU/RAM, progress file (results/<name>_progress.json), log size/age, result present. stop: kills only that experiment's
process tree (matched by script name; never python in general). start: detached launch, log to results/<name>_run.log. restart = stop + start
(experiments with a per-world cache, like x1_29, resume where they stopped)."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, subprocess
R = os.path.join(str(vipaths.RESULTS)); V = os.path.dirname(os.path.abspath(__file__))
PS = lambda c: subprocess.run(["powershell", "-NoProfile", "-Command", c], capture_output=True, text=True).stdout


def procs(name):
    out = PS(f"Get-CimInstance Win32_Process | ?{{$_.CommandLine -match '{name}\.py' -and $_.CommandLine -notmatch 'ctl\.py'}} | % {{\"$($_.ProcessId)|$($_.ParentProcessId)|$($_.Name)\"}}")
    return [tuple(l.split("|")) for l in out.split() if "|" in l]


def _rd(p):
    try: return json.load(open(p))
    except (OSError, ValueError): return None     # a heartbeat caught mid-write


def machine():
    o = PS("$c=[int](Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue;$m=Get-CimInstance Win32_OperatingSystem;\"$c $([int](100*$m.FreePhysicalMemory/$m.TotalVisibleMemorySize))\"").split()
    return f"CPU {o[0]}%  RAM free {o[1]}%" if len(o) == 2 else "machine: n/a"


def status(name):
    P = procs(name); n_work = len(PS("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ?{$_.CommandLine -match 'multiprocessing'} | % {$_.ProcessId}").split())
    res = os.path.exists(os.path.join(R, f"{name.replace('x1_', 'x1_')}_run.json")); lg = os.path.join(R, f"{name}_run.log"); pg = os.path.join(R, f"{name}_progress.json")
    print(f"[{name}] {'RUNNING' if P else ('FINISHED (result present)' if res else 'NOT RUNNING')}  main/launcher procs {len(P)}, pool workers on machine {n_work}  |  {machine()}")
    if os.path.exists(lg): print(f"  log {os.path.getsize(lg)} bytes, last write {int(time.time() - os.path.getmtime(lg))} s ago")
    if os.path.exists(pg):
        j = json.load(open(pg)); age = int(time.time() - j["updated"]); eta = (j["seconds"] / max(j["done"], 1)) * (j["total"] - j["done"])
        print(f"  progress {j['done']}/{j['total']} worlds, {j['seconds']} s elapsed, last update {age} s ago, naive ETA {int(eta)} s")
    pd = os.path.join(R, "progress")
    if os.path.isdir(pd):
        now = time.time(); rows = [j for j in (_rd(os.path.join(pd, f)) for f in os.listdir(pd)) if j]
        for j in sorted((j for j in rows if now - j["updated"] < 900), key=lambda j: j["seed"]):
            print(f"  [{j.get('exp', '?')}] world {j['seed']}: arm {j['arm']}, problem {j['t']}/{j['T']}, running R^2 {j['r2']:.3f}, {int(now - j['updated'])} s ago")
    return bool(P)


def stop(name):
    for pid, ppid, nm in procs(name): subprocess.run(["taskkill", "/T", "/F", "/PID", pid], capture_output=True)
    print(f"[{name}] stopped"); time.sleep(1)


def start(name):
    lg = open(os.path.join(R, f"{name}_run.log"), "a")
    subprocess.Popen([sys.executable, "-u", str(vipaths.find(name)), "run"], cwd=str(vipaths.find(name).parent), stdout=lg, stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x00000200)
    print(f"[{name}] started (log results/{name}_run.log)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"; names = sys.argv[2:] or ["x1_42"]
    for n in names:
        {"status": status, "ping": status, "stop": stop, "start": start, "restart": lambda x: (stop(x), start(x))}[cmd](n)
