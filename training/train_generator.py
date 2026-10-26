"""M1: train the Next-Concept generator (CE over levels).

Usage:
    python training/train_generator.py --config configs/eval/lm1b-small.yaml \
        --tokenizer artifacts/lm1b-small/tokenizer.pt --data /path/to/lm1b.txt \
        --out artifacts/lm1b-small/generator.pt
"""
import argparse, os, glob, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from omegaconf import OmegaConf
import types


class NCMChunkDataset(Dataset):
    def __init__(self, paths, hf_tok, vqvae, cfg, context_length, max_chunks=5000):
        toks = []
        for p in paths:
            with open(p, encoding="utf-8", errors="ignore") as f:
                toks.extend(hf_tok.encode(f.read()))
        chunks = [toks[i:i+context_length] for i in range(0, len(toks)-context_length, context_length)]
        chunks = chunks[:max_chunks]
        self.samples = []
        vqvae.eval()
        prefix_id = hf_tok.cls_token_id if hf_tok.cls_token_id is not None else 0
        with torch.no_grad():
            for c in chunks:
                x = torch.tensor([c], dtype=torch.long)
                inp, target = vqvae.get_ncm_inputs(x)  # inp: (1, L-levels0, D)
                # prepend prefix embedding (raw token -> encoder.wte) for P=prefix_len positions
                P = cfg["model"]["prefix_len"]
                pref_emb = vqvae.encoder.wte(torch.tensor([[prefix_id]*P]))  # (1, P, D)
                inp = torch.cat([pref_emb, inp], dim=1)  # (1, P+L-levels0, D)
                self.samples.append((inp.squeeze(0).cpu(), target.squeeze(0).cpu()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def build_vqvae(cfg, ckpt):
    tcfg = cfg["tokenizer"]["vqvae"]
    vqcfg = OmegaConf.create(tcfg["vector_quantizer_config"])
    vqvae = VQVAE(
        in_vocab_size=tcfg["in_vocab_size"],
        context_length=tcfg["context_length"],
        embed_dim=tcfg["embed_dim"],
        vqvae_vocab_size=tcfg["vqvae_vocab_size"],
        n_layers=tcfg["n_layers"],
        vector_quantizer_config=vqcfg,
        compression_factor=tcfg.get("compression_factor", 1),
        dropout=tcfg.get("dropout", 0.1),
        pre_quant_groupnorm=tcfg.get("pre_quant_groupnorm", 4),
        pre_quant_dropout=tcfg.get("pre_quant_dropout", 0.0),
        use_qk_norm=tcfg.get("use_qk_norm", False),
        enc_positional=tcfg.get("enc_positional", False),
        dec_positional=tcfg.get("dec_positional", True),
        dec_layers=tcfg.get("dec_layers", 4),
    )
    sd = torch.load(ckpt, map_location="cpu")
    vqvae.load_state_dict(sd)
    for p in vqvae.parameters():
        p.requires_grad = False
    return vqvae


def to_config(ns):
    return types.SimpleNamespace(**ns)


def build_model(cfg, vqvae):
    mc = cfg["model"]
    # OmegaConf -> SimpleNamespace recursively
    def convert(o):
        if isinstance(o, dict):
            return types.SimpleNamespace(**{k: convert(v) for k, v in o.items()})
        if isinstance(o, list):
            return o
        return o
    config = convert(mc)
    # vocab_size list ok
    model = NextConceptTransformer(vqvae, config)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="generator.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-chunks", type=int, default=2000)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pre = cfg["tokenizer"]["pre_tokenizer"]
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(pre["model_id"])
    if pre.get("special_tokens"):
        hf_tok.add_special_tokens(pre["special_tokens"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae = build_vqvae(cfg, args.tokenizer).to(device)
    model = build_model(cfg, vqvae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    paths = [args.data] if os.path.isfile(args.data) else sorted(glob.glob(os.path.join(args.data, "**", "*"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p)]
    ds = NCMChunkDataset(paths, hf_tok, vqvae, cfg, cfg["tokenizer"]["vqvae"]["context_length"], args.max_chunks)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0)

    model.train()
    for ep in range(args.epochs):
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
