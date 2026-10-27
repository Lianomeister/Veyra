"""M1: train the Next-Concept generator (robust)."""
import argparse, os, glob, yaml, torch, torch.nn as nn, torch.nn.functional as F
import random, math, types
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from omegaconf import OmegaConf
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class NCMChunkDataset(Dataset):
    def __init__(self, paths, hf_tok, vqvae, model, cfg, context_length, max_chunks=5000):
        toks = []
        for p in paths:
            with open(p, encoding="utf-8", errors="ignore") as f:
                toks.extend(hf_tok.encode(f.read()))
        chunks = [toks[i:i+context_length] for i in range(0, len(toks)-context_length, context_length)]
        chunks = chunks[:max_chunks]
        self.samples = []
        vqvae.eval()
        prefix_id = hf_tok.cls_token_id if hf_tok.cls_token_id is not None else 0
        P = cfg["model"]["prefix_len"]
        with torch.no_grad():
            for c in chunks:
                dev = next(vqvae.parameters()).device
                x = torch.tensor([c], dtype=torch.long, device=dev)
                inp, target = vqvae.get_ncm_inputs(x)  # (1, L-levels0, D), (1, L-levels0)
                dev = next(vqvae.parameters()).device
                pref_emb = vqvae.encoder.wte(torch.tensor([[prefix_id]*P], device=dev))  # (1, P, D)
                parts = [pref_emb]
                if bool(cfg["model"].get("query_level0", False)):
                    l0q = model.l0_queries.detach().to(dev)  # (1, levels0, D)
                    parts.append(l0q)
                parts.append(inp)
                inp = torch.cat(parts, dim=1)  # (1, P + (levels0) + (L-levels0), D)
                self.samples.append((inp.squeeze(0).cpu(), target.squeeze(0).cpu()))
        if not self.samples: raise ValueError("No samples produced.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

def build_vqvae(cfg, ckpt):
    tcfg = cfg["tokenizer"]["vqvae"]
    vqcfg = OmegaConf.create(tcfg["vector_quantizer_config"])
    vqvae = VQVAE(in_vocab_size=tcfg["in_vocab_size"], context_length=tcfg["context_length"],
        embed_dim=tcfg["embed_dim"], vqvae_vocab_size=tcfg["vqvae_vocab_size"],
        n_layers=tcfg["n_layers"], vector_quantizer_config=vqcfg,
        compression_factor=tcfg.get("compression_factor", 1), dropout=tcfg.get("dropout", 0.1),
        pre_quant_groupnorm=tcfg.get("pre_quant_groupnorm", 4), pre_quant_dropout=tcfg.get("pre_quant_dropout", 0.0),
        use_qk_norm=tcfg.get("use_qk_norm", False), enc_positional=tcfg.get("enc_positional", False),
        dec_positional=tcfg.get("dec_positional", True), dec_layers=tcfg.get("dec_layers", 4))
    sd = torch.load(ckpt, map_location="cpu")
    vqvae.load_state_dict(sd["model"], strict=True)
    for p in vqvae.parameters(): p.requires_grad = False
    return vqvae

def to_ns(o):
    if isinstance(o, dict): return types.SimpleNamespace(**{k: to_ns(v) for k, v in o.items()})
    if isinstance(o, list): return o
    return o

def build_model(cfg, vqvae):
    return NextConceptTransformer(vqvae, to_ns(cfg["model"]))

def cosine_lr(step, total, base, warmup):
    if step < warmup: return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + __import__("math").cos(__import__("math").pi * prog))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="generator.pt")
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--max-chunks", type=int, default=2000)
    ap.add_argument("--grad-accum", type=int, default=1); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.config): raise FileNotFoundError(args.config)
    with open(args.config) as f: cfg = yaml.safe_load(f)
    pre = cfg["tokenizer"]["pre_tokenizer"]
    hf_tok = AutoTokenizer.from_pretrained(pre["model_id"])
    if pre.get("special_tokens"): hf_tok.add_special_tokens(pre["special_tokens"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae = build_vqvae(cfg, args.tokenizer).to(device)
    model = build_model(cfg, vqvae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    start_epoch, global_step, best_loss = 0, 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0); global_step = ck.get("step", 0)
        best_loss = ck.get("best_loss", best_loss)
        print(f"Resumed epoch {start_epoch}")
    paths = [args.data] if os.path.isfile(args.data) else sorted(glob.glob(os.path.join(args.data, "**", "*"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths: raise FileNotFoundError(f"No data files at {args.data}")
    ds = NCMChunkDataset(paths, hf_tok, vqvae, model, cfg, cfg["tokenizer"]["vqvae"]["context_length"], args.max_chunks)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=True)
    print(f"Dataset: {len(ds)} samples, {len(dl)} batches/epoch")
    total_steps = args.epochs * len(dl)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.train()
    for ep in range(start_epoch, args.epochs):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
        running = 0.0; opt.zero_grad()
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                logits = model.forward(x)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                aux = model.moe_aux_loss() if hasattr(model, "moe_aux_loss") else 0.0
                if torch.is_tensor(aux):
                    loss = (ce + 0.01 * aux) / args.grad_accum
                else:
                    loss = ce / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = cosine_lr(global_step, total_steps, args.lr, args.warmup)
                for g in opt.param_groups: g["lr"] = lr
                scaler.step(opt); scaler.update(); opt.zero_grad(); global_step += 1
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.3f}")
        epoch_loss = running / len(dl)
        print(f"epoch {ep+1}: avg_loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep+1, "step": global_step, "best_loss": best_loss}, args.out)
            print(f"  -> best ckpt saved ({best_loss:.4f})")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep+1, "step": global_step, "best_loss": best_loss},
                   os.path.join(os.path.dirname(args.out) or ".", "generator_last.pt"))
    print(f"Done. best_loss={best_loss:.4f} -> {args.out}")

if __name__ == "__main__":
    main()
