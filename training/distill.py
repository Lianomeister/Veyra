"""Knowledge Distillation: train a small student NCM from a large teacher NCM.

Loss = alpha * CE(student, hard labels) + (1-alpha) * KL(student, teacher) * T^2

Usage:
    python training/distill.py --teacher-config ... --student-config ... \
        --teacher-ckpt ... --student-out ... --data ...
"""
import argparse, os, glob, yaml, torch, torch.nn as nn, torch.nn.functional as F
import random, math, types
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from omegaconf import OmegaConf
from tqdm import tqdm

def to_ns(o):
    if isinstance(o, dict): return types.SimpleNamespace(**{k: to_ns(v) for k, v in o.items()})
    if isinstance(o, list): return o
    return o

def build(cfg, ckpt=None, device="cpu"):
    vqvae = VQVAE(**cfg["tokenizer"]["vqvae"])
    model = NextConceptTransformer(vqvae, to_ns(cfg["model"]))
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
    return model.to(device), vqvae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-config", required=True)
    ap.add_argument("--student-config", required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--student-out", default="student.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight for hard CE")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--max-chunks", type=int, default=2000)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    tcfg = yaml.safe_load(open(args.teacher_config))
    scfg = yaml.safe_load(open(args.student_config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher, _ = build(tcfg, args.teacher_ckpt, device)
    student, _ = build(scfg, None, device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    pre = scfg["tokenizer"]["pre_tokenizer"]
    hf_tok = AutoTokenizer.from_pretrained(pre["model_id"])
    if pre.get("special_tokens"): hf_tok.add_special_tokens(pre["special_tokens"])

    # simple chunk dataset -> code ints not needed; reuse dummy text chunks as code via tiny mapping
    paths = [args.data] if os.path.isfile(args.data) else sorted(__import__("glob").glob(os.path.join(args.data, "**", "*"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p)]
    toks = []
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            toks.extend(hf_tok.encode(f.read()))
    ctx = scfg["tokenizer"]["vqvae"]["context_length"]
    chunks = [toks[i:i+ctx] for i in range(0, len(toks)-ctx, ctx)][:args.max_chunks]
    if not chunks: raise ValueError("No chunks")

    # Build tokenizer for student to convert text -> ncm inputs
    from models.vqvae import VQVAE
    svq = VQVAE(**scfg["tokenizer"]["vqvae"])
    # load student tokenizer if present alongside? fallback: random (smoke only)
    svq.eval()
    svq.to(device)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    T = args.temperature

    student.train()
    for ep in range(args.epochs):
        pbar = tqdm(chunks, desc=f"distill epoch {ep+1}/{args.epochs}")
        for c in pbar:
            x = torch.tensor([c], dtype=torch.long, device=device)
            with torch.no_grad():
                inp_s, tgt_s = svq.get_ncm_inputs(x)
                # teacher uses its own tokenizer ideally; here reuse student inp as proxy for smoke
                t_logits = teacher.forward(inp_s)
            s_logits = student.forward(inp_s)
            # KL on logits (temperature)
            kl = F.kl_div(F.log_softmax(s_logits/T, dim=-1), F.softmax(t_logits/T, dim=-1), reduction="batchmean") * (T*T)
            ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), tgt_s.reshape(-1))
            loss = args.alpha * ce + (1-args.alpha) * kl
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}", kl=f"{kl.item():.3f}")
        torch.save({"model": student.state_dict()}, args.student_out)
        print(f"epoch {ep+1} done -> {args.student_out}")

if __name__ == "__main__":
    main()
