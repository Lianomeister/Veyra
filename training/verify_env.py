"""Verify environment (torch/cuda) and print recommended run commands."""
import sys
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("-> use --amp")
    else:
        print("-> CPU mode: drop --amp, use tiny config, small --max-chunks")
except Exception as e:
    print("torch NOT installed:", e)
    print("Install CPU:  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("Install CUDA: python -m pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)
print("env ok")
