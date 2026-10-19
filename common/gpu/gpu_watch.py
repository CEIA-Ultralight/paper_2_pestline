#!/usr/bin/env python
"""Login-node watchdog: scancel a job the moment one of its processes holds a GPU
outside the job's Slurm allocation (second net besides the in-job gpu_jail/auditor).

  gpu_watch.py --jobs 34400,34401 --every 3        # prints only events; exits when the jobs end

Read-only except for ``scancel`` of the watched jobs.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Dict, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_audit import parse_pmon  # noqa: E402


def parse_idx(text: str) -> Set[int]:
    """``scontrol show job -d`` GRES ``(IDX:0-1,5)`` -> {0, 1, 5}."""
    out: Set[int] = set()
    for m in re.finditer(r"IDX:([0-9,\-]+)", text):
        for part in m.group(1).split(","):
            if "-" in part:
                a, b = part.split("-")
                out.update(range(int(a), int(b) + 1))
            elif part:
                out.add(int(part))
    return out


def job_of(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for kv in f.read().split(b"\0"):
                if kv.startswith(b"SLURM_JOB_ID="):
                    return kv.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


def job_state(job: str) -> str:
    r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--every", type=float, default=3)
    a = ap.parse_args()
    jobs = [j for j in a.jobs.split(",") if j]
    alloc: Dict[str, Set[int]] = {}
    while jobs:
        for j in list(jobs):
            st = job_state(j)
            if st not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
                print(f"gpu_watch: job {j} ended ({st or 'gone'})", flush=True)
                jobs.remove(j)
            elif st == "RUNNING" and j not in alloc:
                d = subprocess.run(["scontrol", "show", "job", "-d", j], capture_output=True, text=True)
                alloc[j] = parse_idx(d.stdout)
                print(f"gpu_watch: job {j} running on GPU(s) {sorted(alloc[j])}", flush=True)
        pm = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True).stdout
        for gpu, pid, kind, cmd in parse_pmon(pm):
            j = job_of(pid)
            if j in alloc and gpu not in alloc[j]:
                print(f"gpu_watch: VIOLATION job {j} pid={pid} ({cmd},{kind}) on GPU {gpu}; "
                      f"allowed={sorted(alloc[j])} -> scancel", flush=True)
                subprocess.run(["scancel", j])
        time.sleep(a.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
