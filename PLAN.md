# Veyra One - Plan: The First Large Next-Concept Model (NCM)

> Status: Draft plan. Working directory: C:\Users\Liam\Documents\veyra (empty repo, starting from scratch).
> Owner: Veyra AI (veyra-ai on Hugging Face).
> Goal: Train and release Veyra One, the first large (multi-billion-parameter) Next-Concept Model (NCM), pushing NCM beyond the research-scale checkpoints (~1B params) from Hazy Research.

## 1. Vision & Positioning

- NCM (Next-Concept Modeling) is a two-stage, coarse-to-fine non-autoregressive LM: a multiscale residual VQ-VAE tokenizer turns text into a hierarchy of discrete codes; a transformer predicts one level at a time, all positions in a level in parallel.
- Released NCMs top out at ~1B params (LM1B-large: 36L/20H/d=1280) and are research demos on LM1B/OpenWebText/XSum/WMT14/Sudoku. No production-grade, instruction-tuned, large NCM exists.
- Veyra One fills that gap: a large (target 7B-13B) NCM trained on a broad, high-quality corpus, instruction-tuned, released with weights + tokenizer + eval harness under the Veyra brand.
- Branding: new generation named "Veyra One": Veyra-One-<size>-Base / -Instruct, plus Veyra-One-Tokenizer.

## 2. Success Criteria

1. Trained multiscale tokenizer + generator pair at >=7B params, released on HF.
2. Quality on standard eval competitive with an AR baseline of similar size, retaining NCM speed (target >=5x fewer NFE vs diffusion).
3. Instruction-tuned variant passing basic chat/tool-use eval.
4. Reproducible training code + configs + model card in this repo.
5. Public, defensible claim: "first large known NCM" with numbers.

## 3. Architecture (reuse, then scale)

Reuse released NCM design (HazyResearch/ncm), then scale.

### Stage 1 - Multiscale Residual VQ-VAE Tokenizer
- Pre-tokenizer: HF BPE (start bert-base-uncased-style or Veyra BPE; decide Phase 0).
- Encoder/decoder transformer VQ-VAE; embed_dim 384 -> scale to 768/1024.
- Codebook: 12 levels, 8192 codes per level (residual), manual levels [1,8,16,24,32,40,52,64,76,92,108,128] for 128 context.
- For large: increase context (512/1024) and levels; key research lever ("adapt computation to difficulty").

### Stage 2 - Next-Concept Generator
- Transformer with block-diagonal attention per level, RoPE, QK-norm, level + positional encodings.
- Shared output head across levels, per-level bias.
- Post-upsample conv (k=3) between levels. CFG for conditional tasks.

### Scaling knobs
- Width/depth: 7B (32L/4096d), 13B (40L/5120d).
- Context: 128 -> 512 -> 1024. Levels: keep 12 initially; experiment adaptive stopping.

## 4. Data Strategy

- Tokenizer training data: diverse text matching target domain (web, code, books). Reuse OpenWebText-style + Veyra corpora.
- Generator pretraining: filtered web (FineWeb-edu-style), code (Stack), books, plus Veyra corpora. Target >=1T tokens (NCM uses 4-5x fewer tokens than diffusion; "large" implies >=500B-1T).
- Instruction tuning: Veyra instruct datasets + public (OpenHermes-style, tool-calling traces). Keep Veyra tool-calling focus.
- Eval data: hold-out slices; reuse NCM eval protocol (genppl under GPT-2-large, BLEU/ROUGE, Sudoku-style reasoning).


## 5. Training Recipe

NCM repo ships eval only - training code must be implemented by us. Biggest engineering item.

### Phase A - Reproduce (small)
- Reimplement training loop for tokenizer (VQ-VAE reconstruction + commitment + nested dropout + token-level recon) and generator (next-concept CE per level, code corruption / fine-dropout aux).
- Reproduce LM1B-small numbers to validate (gate: must match published genppl before scaling).

### Phase B - Tokenizer at scale
- Train tokenizer on target corpus; freeze; release Veyra-One-Tokenizer.
- Validate reconstruction fidelity (token recovery rate) >= target.

### Phase C - Generator pretraining
- Train generator on tokenized corpus; cosine LR, warmup; NCM lower token budget -> fewer passes.
- Track genppl + unigram entropy per scale (NCM scaling law).

### Phase D - Instruction tuning
- SFT on chat/tool data; optional DPO/ORPO.

### Phase E - Test-time scaling
- MBR decoding over multiple candidates (NCM showed gains); ship as default sampling option.

## 6. Infrastructure & Compute

- Multi-GPU cluster (H100s). Estimate: 7B pretraining on ~1T tokens at NCM efficiency ~ thousands H100-hours; 13B proportionally. Plan 8-64 H100s.
- Storage: tokenized datasets (TB scale), checkpoints (tens of GB each), keep last 3 + best.
- Mixed precision (bf16), gradient checkpointing, FSDP/DeepSpeed. Implement eval sharding like NCM scripts/eval_shard.py.

## 7. Evaluation Plan

- Reproduce NCM metrics: genppl (GPT-2-large scorer), unigram entropy, 5 seeds x 256 samples.
- Conditional: XSum ROUGE, WMT14 BLEU/chrF.
- Reasoning: Sudoku solved-rate (or harder reasoning suite).
- Chat: MT-Bench / IFEval for instruct variant.
- Speed: wall-clock per sample on H100; NFE count.
- Publish a Micro-Model-Bench-style leaderboard entry for Veyra One.

## 8. Release Plan

- HF repos under veyra-ai:
  - veyra-ai/Veyra-One-Tokenizer
  - veyra-ai/Veyra-One-7B-Base, -7B-Instruct
  - (optional) veyra-ai/Veyra-One-13B-Base
- Ship config.yaml, tokenizer.pt, generator.pt, model card, eval configs.
- Code: this repo (training + eval), MIT license (match NCM).

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Training code not public (NCM eval-only) | Reimplement from paper + configs; reproduce small first as gate |
| Tokenizer reconstruction quality at scale | Validate recovery rate; consider joint training (NCM "what's next") |
| NCM scaling unknown beyond ~1B | Run scaling ablations at 1B/3B before committing 7B+ |
| Compute budget | Start with 7B; defer 13B; seek sponsorship |
| Claim "first large NCM" challenged | Publish numbers, code, checkpoints; be precise about "large" (>=7B) |
| Context/levels design | Ablate levels and context in Phase A |

## 10. Milestones

- M0: Plan + repo scaffold + environment. (done)
- M1 (code-complete): Reproduce LM1B-small training & eval (gate). Training/eval code + docs ready; execution & gate validation on cluster. See docs/M1.md.
- M2 (in progress): Tokenizer at scale. See docs/M2.md.
- M3: 7B generator pretrained; scaling-law report.
- M4: Instruction-tuned variant.
- M5: Eval suite + model cards + release.
- M6: Blog/paper note claiming "first large NCM"; 13B follow-up.

## 11. Open Decisions (need user input)

1. Target size for Veyra One: **decided -> 7B first** (13B deferred).
2. Context length: 512 vs 1024 (recommend 512 for v1).
3. Base tokenizer: reuse bert-base-uncased (NCM default) vs train Veyra BPE.
4. Training framework: from-scratch PyTorch+FSDP vs adapt a trainer (recommend from scratch given NCM custom levels).
5. Compute provider / budget.
6. Whether to attempt joint tokenizer+generator training in v1 (recommend no).

## 12. Immediate Next Steps

1. Scaffold repo: models/, tokenization/, training/, configs/, eval/, scripts/, docs/.
2. Pin dependencies (requirements.txt from NCM + training extras).
3. Implement tokenizer training; reproduce reconstruction on small corpus.
4. Set up experiment tracking (wandb) and checkpointing.
5. Decide open questions above, then begin Phase A.

---

*This plan is a living document. Update as decisions are made.*

- Announcement: blog post + paper note + "first large NCM" claim with numbers.
