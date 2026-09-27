#!/usr/bin/env python3
"""Run a list of independent GPU jobs in parallel, one job per GPU, on a shared multi-GPU server.

- You choose the GPUs (--gpus 0,2,3) or let it pick every GPU with enough free memory (--gpus auto).
- Before a job starts on a GPU, the GPU's free memory is checked again (other users may have taken it);
  a busy GPU is skipped until it frees up.
- Each job sees exactly one GPU (CUDA_VISIBLE_DEVICES, PCI bus order, same numbering as nvidia-smi).
- Progress is written to <log_dir>/status.json; each job's output to <log_dir>/<name>.log.
- Jobs that already finished successfully (per status.json) are skipped on a rerun; --force reruns them.

Job file: one job per line, "name | shell command"; blank lines and lines starting with # are ignored.
The command runs from the repository root; {name} in the command is replaced by the job name.

Usage:
  python tools/local/run_jobs.py tools/local/jobs/m0_v2.txt --gpus 0,1,2,3 --min_free_gb 40
  python tools/local/run_jobs.py tools/local/jobs/m0_v2.txt --gpus auto --dry_run
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def gpu_table():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu,name",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.strip().splitlines():
        i, free, total, util, name = [x.strip() for x in line.split(",", 4)]
        rows[int(i)] = {"free_gb": int(free) / 1024, "total_gb": int(total) / 1024, "util": int(util), "name": name}
    return rows


def read_jobs(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, cmd = [x.strip() for x in line.split("|", 1)]
        jobs.append({"name": name, "cmd": cmd.replace("{name}", name)})
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), f"duplicate job names in {path}"
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="auto", help='comma-separated GPU indices as in nvidia-smi, or "auto"')
    ap.add_argument("--min_free_gb", type=float, default=40.0, help="start a job only if the GPU has this much free")
    ap.add_argument("--log_dir", default=None, help="default: logs/<job file name>")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--only", default=None, help="comma-separated job names to run (default: all)")
    ap.add_argument("--force", action="store_true", help="rerun jobs that already succeeded")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    jobs = read_jobs(args.jobs)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["name"] in keep]
    log_dir = args.log_dir or os.path.join(ROOT, "logs", os.path.splitext(os.path.basename(args.jobs))[0])
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(log_dir, "status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}

    table = gpu_table()
    gpus = sorted(table) if args.gpus == "auto" else [int(g) for g in args.gpus.split(",")]
    for g in gpus:
        assert g in table, f"GPU {g} not found; nvidia-smi shows {sorted(table)}"
    print("GPUs:")
    for g in sorted(table):
        t = table[g]
        mark = "use" if g in gpus else "   "
        print(f"  [{mark}] {g}: {t['name']}  free {t['free_gb']:.1f}/{t['total_gb']:.1f} GB  util {t['util']}%")

    pending = [j for j in jobs if args.force or status.get(j["name"], {}).get("state") != "done"]
    skipped = [j["name"] for j in jobs if j not in pending]
    if skipped:
        print(f"already done (use --force to rerun): {', '.join(skipped)}")
    print(f"{len(pending)} job(s) to run on GPU(s) {gpus}, one job per GPU, min free {args.min_free_gb} GB")
    if args.dry_run:
        for j in pending:
            print(f"  {j['name']}: {j['cmd']}")
        return

    running = {}  # gpu -> (job, Popen, t0, logfile)

    def save():
        json.dump(status, open(status_path, "w"), indent=1)

    def stop_all(*_):
        for g, (j, p, _, _) in running.items():
            p.terminate()
            status[j["name"]].update(state="killed", end=time.strftime("%Y-%m-%d %H:%M:%S"))
        save()
        sys.exit(1)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    waiting_note = set()
    while pending or running:
        for g, (j, p, t0, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            mins = (time.time() - t0) / 60
            status[j["name"]].update(state="done" if rc == 0 else "failed", rc=rc, minutes=round(mins, 1),
                                     end=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"[{time.strftime('%H:%M:%S')}] {j['name']} on GPU {g}: {'OK' if rc == 0 else f'FAILED rc={rc}'}"
                  f" ({mins:.1f} min)", flush=True)
            del running[g]
            save()
        if pending:
            table = gpu_table()
            for g in gpus:
                if not pending or g in running:
                    continue
                if table[g]["free_gb"] < args.min_free_gb:
                    if g not in waiting_note:
                        print(f"[{time.strftime('%H:%M:%S')}] GPU {g} busy ({table[g]['free_gb']:.1f} GB free), "
                              f"waiting", flush=True)
                        waiting_note.add(g)
                    continue
                waiting_note.discard(g)
                j = pending.pop(0)
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g))
                lf = open(os.path.join(log_dir, f"{j['name']}.log"), "w")
                p = subprocess.Popen(j["cmd"], shell=True, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
                running[g] = (j, p, time.time(), lf)
                status[j["name"]] = {"state": "running", "gpu": g, "cmd": j["cmd"],
                                     "start": time.strftime("%Y-%m-%d %H:%M:%S")}
                save()
                print(f"[{time.strftime('%H:%M:%S')}] start {j['name']} on GPU {g}", flush=True)
                time.sleep(20)  # let the job allocate before the next free-memory check
        time.sleep(args.poll)

    failed = [n for n, s in status.items() if s.get("state") == "failed"]
    print("all jobs finished" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    save()


if __name__ == "__main__":
    main()
