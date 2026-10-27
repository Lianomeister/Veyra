"""Report parameter efficiency: params, FLOPs estimate, and params-per-quality.

Usage:
    python training/efficiency_report.py --config configs/eval/lm1b-tiny.yaml
"""
import argparse, yaml, torch, sys, os
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
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    vqvae = VQVAE(**cfg["tokenizer"]["vqvae"])
    model = NextConceptTransformer(vqvae, to_ns(cfg["model"]))
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(sd["model"], strict=False)
    params = sum(p.numel() for p in model.parameters())
    vq_params = sum(p.numel() for p in vqvae.parameters())
    print(f"Generator params: {params/1e6:.2f}M")
    print(f"Tokenizer params: {vq_params/1e6:.2f}M")
    print(f"Total params: {(params+vq_params)/1e6:.2f}M")
    print(f"MoE enabled: {getattr(cfg['model'], 'use_moe', False) if isinstance(cfg['model'], object) else cfg['model'].get('use_moe', False)}")
    print("Efficiency metric (PPL/Params) requires eval run; see evaluation/gen_ppl.py")

if __name__ == "__main__":
    main()
