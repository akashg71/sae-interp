#!/usr/bin/env python
"""Phase 1 — feature exploration with a pretrained SAE.

Harvest activations over a corpus, push them through the pretrained SAE, and collect
max-activating examples for a batch of features so you can pick 3-5 clearly
interpretable ones to carry into the causal phase.

Outputs:
  - prints max-activating examples + density + logit-lens hints per feature,
  - writes a markdown table to results/features.md you can annotate by hand.

Run:
    python scripts/01_explore_pretrained_sae.py
    python scripts/01_explore_pretrained_sae.py --features 0 5 12 100 --top-k 12
    python scripts/01_explore_pretrained_sae.py --auto 20   # auto-pick 20 active features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sae_interp import net

net.bootstrap()

from sae_interp.config import load_config  # noqa: E402
from sae_interp.device import get_device  # noqa: E402
from sae_interp import features as F  # noqa: E402
from sae_interp.activations import load_corpus_tokens  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 — explore a pretrained SAE")
    p.add_argument("--config", default="configs/sae_gpt2small.yaml")
    p.add_argument("--features", type=int, nargs="*", default=None,
                   help="explicit feature indices to inspect")
    p.add_argument("--auto", type=int, default=15,
                   help="if --features not given, auto-pick this many active features")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-seq", type=int, default=None,
                   help="override data.n_sequences for a quicker run")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.n_seq:
        cfg.data.n_sequences = args.n_seq
    device = get_device()
    print(f"[phase1] device={device}  hook={cfg.model.hook_name}")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(cfg.model.name, device=str(device))
    model.eval()
    sae = SAE.from_pretrained(cfg.pretrained_sae.release, cfg.pretrained_sae.sae_id, device=str(device))
    if isinstance(sae, tuple):
        sae = sae[0]
    print(f"[phase1] SAE d_sae={sae.cfg.d_sae}")

    tokens = load_corpus_tokens(cfg, model)
    print(f"[phase1] corpus: {tokens.shape[0]} sequences x {tokens.shape[1]} tokens")

    # Decide which features to inspect.
    if args.features:
        feat_indices = args.features
    else:
        # Auto-pick: score a sample of features, keep ones with healthy (not extreme) density.
        sample = list(range(0, sae.cfg.d_sae, max(1, sae.cfg.d_sae // 400)))[:400]
        acts = F.collect_feature_activations(model, sae, tokens[:200], cfg.model.hook_name, sample, device)
        density = (acts > 0).float().mean(dim=(0, 1))
        # interpretable-ish band: fires sometimes but not on almost everything
        good = [(sample[i], density[i].item()) for i in range(len(sample))
                if 0.002 < density[i].item() < 0.15]
        good.sort(key=lambda t: -t[1])
        feat_indices = [i for i, _ in good[: args.auto]] or sample[: args.auto]
        print(f"[phase1] auto-picked {len(feat_indices)} features: {feat_indices}")

    # Full scoring pass for the chosen features over the whole corpus.
    feat_acts = F.collect_feature_activations(model, sae, tokens, cfg.model.hook_name, feat_indices, device)

    rows = []
    for col, fi in enumerate(feat_indices):
        density = F.feature_density(feat_acts, col)
        examples = F.max_activating_examples(model, feat_acts, tokens, col, top_k=args.top_k)
        lens = F.logit_lens_tokens(model, sae, fi, top_k=8)
        print(f"\n=== feature #{fi} | density {density:.3%} | "
              f"logit-lens top tokens: {[t for t, _ in lens]} ===")
        for ex in examples:
            print("   " + F.format_example(ex))
        top_ctx = F.format_example(examples[0]) if examples else "(never fired)"
        rows.append((fi, density, [t for t, _ in lens][:5], top_ctx))

    # Write an annotatable markdown table.
    out = Path(cfg.results_dir) / "features.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# Feature catalogue (Phase 1)\n\n")
        fh.write(f"Model `{cfg.model.name}` · hook `{cfg.model.hook_name}` · "
                 f"SAE `{cfg.pretrained_sae.release}`\n\n")
        fh.write("Fill in the **Label** column after reading the examples.\n\n")
        fh.write("| Feature | Density | Logit-lens hint | Top example | Label (you fill in) |\n")
        fh.write("|--------:|--------:|-----------------|-------------|---------------------|\n")
        for fi, density, lens, top_ctx in rows:
            safe = top_ctx.replace("|", "\\|")
            fh.write(f"| {fi} | {density:.2%} | {', '.join(lens)} | `{safe}` |  |\n")
    print(f"\n[phase1] wrote {out} — annotate it, then pick 3-5 features for Phase 2:")
    print("  python scripts/02_causal_intervention.py --features <i> <j> <k>")


if __name__ == "__main__":
    main()
