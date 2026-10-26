"""M1 helper: run NCM eval harness on a trained checkpoint and print a gate summary.

Usage:
    python training/run_eval.py --config configs/eval/lm1b-small.yaml \
        --checkpoint-dir artifacts/lm1b-small --data-dir data/lm1b-small
"""
import argparse, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--task", default="")
    args = ap.parse_args()

    cmd = [sys.executable, "evaluate.py", "--config", args.config,
           "--checkpoint-dir", args.checkpoint_dir]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd)
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
