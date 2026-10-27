"""Machine budget for experiments (user rule, 2026-10-27): all usage restrictions are lifted, but keep a
15% margin so the PC does not crash. Use this instead of hard-coding a worker count.

  CPU   total utilisation <= 85%        -> workers = floor(0.85 * cores - cores busy right now)
  RAM   free >= 15% of total            -> workers also limited by (free - 15% of total) / per-worker need
  GPU   memory used <= 85% of total     -> gpu_fraction() = 0.85 - the share already in use (cap 0.7)

Usage:  from budget import workers, pmap, gpu_fraction, report
        n = workers(per_worker_gb=0.3)
        results = pmap(fn, args_list, n)       # process pool; fn must be a top-level function
Numbers here are the user's margin (USER) or measurements of this machine."""
import os
import subprocess
import multiprocessing as mp

from constants import C

MARGIN = C("margin")    # USER: keep at least this share of CPU, RAM and GPU memory free
GPU_CAP = C("gpu_cap")  # NeuroBlock NB_GPU_FRAC upper bound (see constants.py)


def _mem_gb():
    """(total, free) physical memory in GB (Windows via ctypes; no psutil dependency)."""
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30


def _cpu_busy_cores(seconds=2.0):
    """Cores currently busy (system-wide), sampled over a short window."""
    import ctypes, time

    class FT(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]

    def t():
        i, k, u = FT(), FT(), FT()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        f = lambda x: (x.hi << C("filetime_shift")) | x.lo
        return f(i), f(k) + f(u)   # idle, kernel (includes idle) + user
    i0, s0 = t()
    time.sleep(seconds)
    i1, s1 = t()
    busy = 1 - (i1 - i0) / max(1, (s1 - s0))
    return busy * os.cpu_count()


def gpu_used_fraction():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], text=True).split(",")
        return float(out[0]) / float(out[1])
    except Exception:
        return None


def workers(per_worker_gb=0.5, reserve_cores=0):
    """How many worker processes may start now without breaking the 15% margin."""
    cores = os.cpu_count()
    by_cpu = int(cores * (1 - MARGIN) - _cpu_busy_cores() - reserve_cores)
    total, free = _mem_gb()
    by_ram = int((free - MARGIN * total) / per_worker_gb)
    return max(1, min(by_cpu, by_ram))


def gpu_fraction():
    """Fraction of GPU memory a new job may take (for NB_GPU_FRAC)."""
    used = gpu_used_fraction() or 0.0
    return max(C("gpu_min"), min(GPU_CAP, (1 - MARGIN) - used))


def report():
    total, free = _mem_gb()
    return {"cores": os.cpu_count(), "busy_cores": round(_cpu_busy_cores(), 1), "ram_total_gb": round(total, 1),
            "ram_free_gb": round(free, 1), "gpu_used_frac": gpu_used_fraction(), "workers_now": workers(),
            "gpu_fraction_now": gpu_fraction()}


def pmap(fn, args_list, n=None):
    """Process-pool map with single-threaded BLAS in every worker; order preserved."""
    n = n or workers()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    with mp.get_context("spawn").Pool(n) as p:
        return p.starmap(fn, args_list)


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1))
