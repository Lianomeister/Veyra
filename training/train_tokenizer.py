"""M1: train the multiscale residual VQ-VAE tokenizer.

Usage (on cluster with GPU):
    python training/train_tokenizer.py --config configs/eval/lm1b-small.yaml \
        --data /path/to/lm1b.txt --out artifacts/lm1b-small/tokenizer.pt
"""
import argparse, os, glob, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.vqvae import VQVAE
from omegaconf import OmegaConf


class TextChunkDataset(Dataset):
    def __init__(self, paths, hf_tok, context_length, max_chunks=100000):
        self.ctx = context_length
        toks = []
        for p in paths:
            with open(p, encoding="utf-8", errors="ignore") as f:
                toks.extend(hf_tok.encode(f.read()))
        self.chunks = [toks[i:i+context_length] for i in range(0, len(toks)-context_length, context_length)]
        self.chunks = self.chunks[:max_chunks]
        self.pad = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 0
        for c in self.chunks:
            if len(c) < context_length:
                c += [self.pad] * (context_length - len(c))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return torch.tensor(self.chunks[i], dtype=torch.long)


def build_vqvae(cfg):
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
    return vqvae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="file or dir of text")
    ap.add_argument("--out", default="tokenizer.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-chunks", type=int, default=20000)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["tokenizer"]
    pre = tcfg["pre_tokenizer"]
    hf_tok = AutoTokenizer.from_pretrained(pre["model_id"])
    if pre.get("special_tokens"):
        hf_tok.add_special_tokens(pre["special_tokens"])

    paths = [args.data] if os.path.isfile(args.data) else sorted(glob.glob(os.path.join(args.data, "**", "*"), recursive=True))
    paths = [p for p in paths if os.path.isfile(p)]
    ds = TextChunkDataset(paths, hf_tok, tcfg["vqvae"]["context_length"], args.max_chunks)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae = build_vqvae(cfg).to(device)
    opt = torch.optim.AdamW(vqvae.parameters(), lr=args.lr)

    vqvae.train()
    for ep in range(args.epochs):
        for step, batch in enumerate(dl):
            batch = batch.to(device)
            recon, vq_loss, *_ = vqvae(batch)
            ce = F.cross_entropy(recon.reshape(-1, recon.size(-1)), batch.reshape(-1))
            loss = ce + (vq_loss if torch.is_tensor(vq_loss) else 0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                print(f"ep{ep} step{step} ce={ce.item():.4f} vq={float(vq_loss):.4f}")
    torch.save({"model": vqvae.state_dict()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
