"""SIVP revision experiments for BMAS.

Every ``python -m experiments.<module>`` run prints its peak resident memory on exit,
so an out-of-memory kill in a Slurm job can be traced to the stage that grew.
"""
import atexit
import resource
import sys


def _report_peak_memory() -> None:
    try:
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"[mem] peak RSS of this stage: {peak_mb:.0f} MB", file=sys.stderr, flush=True)
    except Exception:
        pass


atexit.register(_report_peak_memory)
