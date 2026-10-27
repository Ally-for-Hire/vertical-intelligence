"""Shared setup. Default = user's compute rule for this PC (1 thread, below-normal priority).
Set CN_FULL=1 (only when the user has said the PC is free) for high priority; parallelism is across processes (1 BLAS thread each)."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"


def set_priority():
    try:
        import ctypes
        cls = 0x80 if os.environ.get("CN_FULL") == "1" else 0x4000   # HIGH vs BELOW_NORMAL
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), cls)
    except Exception:
        pass


def n_workers():
    return (os.cpu_count() or 2) if os.environ.get("CN_FULL") == "1" else 1


set_priority()
