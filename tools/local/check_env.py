#!/usr/bin/env python3
"""Environment check for the local server. Prints what Claude needs to know when something fails.

  python tools/local/check_env.py                 # GPUs, driver, torch/CUDA, a bf16 matmul + SDPA on every GPU
  python tools/local/check_env.py --gpus 1,3      # only these GPUs
Writes the same report to logs/check_env.txt (include it when reporting a problem).
"""

import argparse
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # GPU numbers as in nvidia-smi
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


ap = argparse.ArgumentParser()
ap.add_argument("--gpus", default="all")
args = ap.parse_args()

say(f"host {platform.node()}  python {sys.version.split()[0]}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True)
say("nvidia-smi:\n" + (smi.stdout.strip() or smi.stderr.strip()))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

say(f"torch {torch.__version__}  built for CUDA {torch.version.cuda}  cuda available: {torch.cuda.is_available()}")
try:
    import transformers

    say(f"transformers {transformers.__version__}")
except Exception as e:  # noqa: BLE001
    say(f"transformers import FAILED: {e}")
for k in ("HF_HOME", "COCO_DIR", "MLP_CONNECTOR", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES"):
    say(f"{k}={os.environ.get(k)}")
for k in ("COCO_DIR", "MLP_CONNECTOR"):
    p = os.environ.get(k)
    if p:
        say(f"  {k} exists: {os.path.exists(p)}")

ok = torch.cuda.is_available()
if ok:
    n = torch.cuda.device_count()
    ids = range(n) if args.gpus == "all" else [int(g) for g in args.gpus.split(",")]
    for i in ids:
        try:
            with torch.cuda.device(i):
                a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(20):
                    a @ a
                torch.cuda.synchronize()
                tflops = 20 * 2 * 4096**3 / (time.time() - t0) / 1e12
                q = torch.randn(1, 32, 768, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                F.scaled_dot_product_attention(q, q, q).sum().backward()
                free, total = torch.cuda.mem_get_info()
                say(f"GPU {i} {torch.cuda.get_device_name(i)}: bf16 matmul {tflops:.0f} TFLOP/s, SDPA fwd+bwd ok, "
                    f"free {free / 2**30:.1f}/{total / 2**30:.1f} GB")
        except Exception as e:  # noqa: BLE001
            ok = False
            say(f"GPU {i}: FAILED {type(e).__name__}: {e}")
say("ENV_OK" if ok else "ENV_PROBLEM")
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
open(os.path.join(ROOT, "logs", "check_env.txt"), "w").write("\n".join(lines) + "\n")
