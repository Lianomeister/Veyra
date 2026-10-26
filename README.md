# Veyra One

Veyra One is the plan and (eventually) the codebase for the **first large Next-Concept Model (NCM)** — a multi-billion-parameter, instruction-tuned NCM released under the Veyra brand.

See [PLAN.md](./PLAN.md) for the full roadmap, architecture, data, training, evaluation, release plan, risks, milestones, and open decisions.

## Status

Planning phase. No training code yet. NCM (Hazy Research) ships eval-only; training must be reimplemented (see PLAN.md Phase A).

## Layout

- `models/` — generator + tokenizer model code
- `tokenization/` — BPE + VQ-VAE tokenizer
- `training/` — training loops (to be implemented)
- `configs/` — model/eval configs
- `eval/` — evaluation harness
- `scripts/` — sharding/merge utilities
- `docs/` — notes, decisions

## License

MIT (intended, matching NCM).
