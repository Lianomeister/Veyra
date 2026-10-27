"""Pruning + INT8 quantization for deployment (less memory, faster inference).

Usage:
    python training/prune_quantize.py --config configs/eval/lm1b-tiny.yaml \
        --checkpoint artifacts/tiny/generator.pt --out artifacts/tiny/generator_pq.pt \
        --sparsity 0.3
"""
import argparse, os, yaml, torch, torch.nn as nn, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
import types

def to_ns(o):
    if isinstance(o, dict): return types.SimpleNamespace(**{k: to_ns(v) for k, v in o.items()})
    if isinstance(o, list): return o
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="generator_pq.pt")
    ap.add_argument("--sparsity", type=float, default=0.3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    vqvae = VQVAE(**cfg["tokenizer"]["vqvae"])
    model = NextConceptTransformer(vqvae, to_ns(cfg["model"]))
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    # Magnitude pruning on linear weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                thr = torch.quantile(param.abs().flatten(), args.sparsity)
                mask = param.abs() > thr
                param.mul_(mask)

    # Dynamic INT8 quantization (CPU)
    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    # Size comparison
    torch.save({"model": model.state_dict()}, args.out.replace(".pt", "_pruned.pt"))
    torch.save({"model": qmodel.state_dict()}, args.out)
    orig = os.path.getsize(args.checkpoint)
    new = os.path.getsize(args.out)
    print(f"Original: {orig/1e6:.2f} MB")
    print(f"Quantized: {new/1e6:.2f} MB")
    print(f"Reduction: {(1-new/orig)*100:.1f}%")
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
