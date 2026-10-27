"""Verify environment (torch/cuda) and print recommended run commands."""
import sys

missing = []
for mod in ["yaml", "torch", "omegaconf", "transformers", "tqdm"]:
    try:
        __import__(mod)
    except Exception as e:
        missing.append((mod, str(e)))

if missing:
    print("Missing modules:")
    for m, e in missing:
        print(" -", m, "->", e)
    print()
    print("Install all (CPU):")
    print("  python -m pip install pyyaml torch --index-url https://download.pytorch.org/whl/cpu")
    print("  python -m pip install omegaconf transformers tqdm")
    print("Install all (CUDA 12.1):")
    print("  python -m pip install pyyaml torch --index-url https://download.pytorch.org/whl/cu121")
    print("  python -m pip install omegaconf transformers tqdm")
    sys.exit(1)

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("-> use --amp")
else:
    print("-> CPU mode: drop --amp, use tiny config, small --max-chunks")
print("env ok")
