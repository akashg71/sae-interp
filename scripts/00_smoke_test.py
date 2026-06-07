#!/usr/bin/env python
"""Phase 0 — smoke test. The gate for getting started.

Proves the whole stack works on your machine:
  1. loads GPT-2-small via TransformerLens,
  2. loads ONE pretrained GPT-2-small residual-stream SAE via SAELens,
  3. runs ~5 sentences through the model and encodes the activations with the SAE,
  4. prints, for a few features, their top-activating tokens with a little context.

Run:
    python scripts/00_smoke_test.py
    python scripts/00_smoke_test.py --config configs/sae_gpt2small.yaml

If this prints feature examples without error, you're ready for Phase 1.

NOTE: this script downloads GPT-2 (~500 MB) and one SAE (tens of MB) from the
Hugging Face Hub on first run. If your network blocks Hugging Face (e.g. a corporate
web filter returning HTTP 403), it cannot complete — see README "Network / corporate
proxy". The download is cached, so subsequent runs are fast and offline-friendly.
"""

from __future__ import annotations

import argparse

import torch

# net.bootstrap() loads .env and makes SSL use the OS trust store (corporate proxies).
from sae_interp import net

net.bootstrap()

from sae_interp.config import load_config  # noqa: E402  (after bootstrap)
from sae_interp.device import get_device  # noqa: E402


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2023, researchers published a new method for training language models.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The Eiffel Tower is located in Paris, the capital of France.",
    "She poured the coffee, sat down, and opened her laptop to start the day.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 smoke test")
    parser.add_argument("--config", default="configs/sae_gpt2small.yaml")
    parser.add_argument("--n-features", type=int, default=3, help="how many features to inspect")
    parser.add_argument("--top-k", type=int, default=6, help="top activating tokens per feature")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"[smoke] device = {device}")
    print(f"[smoke] model  = {cfg.model.name}  hook = {cfg.model.hook_name}")
    print(f"[smoke] SAE    = release '{cfg.pretrained_sae.release}' / id '{cfg.pretrained_sae.sae_id}'")

    # --- 1. Load the model -------------------------------------------------
    from transformer_lens import HookedTransformer

    print("\n[smoke] loading GPT-2-small (first run downloads ~500 MB)...")
    model = HookedTransformer.from_pretrained(cfg.model.name, device=str(device))
    model.eval()

    # Sanity: the configured hook must exist on this model.
    if cfg.model.hook_name not in model.hook_dict:
        raise SystemExit(
            f"Hook '{cfg.model.hook_name}' not found. Available resid hooks include: "
            + ", ".join(h for h in model.hook_dict if "resid" in h)[:300]
        )

    # --- 2. Load the pretrained SAE ---------------------------------------
    from sae_lens import SAE

    print("[smoke] loading pretrained SAE (first run downloads tens of MB)...")
    sae = SAE.from_pretrained(
        release=cfg.pretrained_sae.release,
        sae_id=cfg.pretrained_sae.sae_id,
        device=str(device),
    )
    # sae-lens >= 4 returns just the SAE object (older versions returned a tuple).
    if isinstance(sae, tuple):
        sae = sae[0]
    d_sae = sae.cfg.d_sae
    print(f"[smoke] SAE loaded: d_in={sae.cfg.d_in}, d_sae={d_sae}")

    # --- 3. Run sentences through model + SAE -----------------------------
    print(f"\n[smoke] running {len(SENTENCES)} sentences through the model...")
    # Tokenise to a common length via padding so we can batch; track real lengths.
    all_feats = []   # list of (tokens_1d, feats_2d) per sentence
    with torch.no_grad():
        for text in SENTENCES:
            tokens = model.to_tokens(text).to(device)            # (1, seq)
            _, cache = model.run_with_cache(
                tokens, names_filter=cfg.model.hook_name, return_type=None
            )
            acts = cache[cfg.model.hook_name][0]                  # (seq, d_model)
            feats = sae.encode(acts.to(torch.float32))            # (seq, d_sae)
            all_feats.append((tokens[0], feats.cpu()))

    # --- 4. Pick a few active features and show top-activating tokens -----
    # Choose features that actually fire across our sentences (skip dead ones).
    stacked = torch.cat([f for _, f in all_feats], dim=0)          # (total_tokens, d_sae)
    max_per_feature = stacked.max(dim=0).values                    # (d_sae,)
    top_features = torch.topk(max_per_feature, args.n_features).indices.tolist()

    print(f"\n[smoke] top-activating tokens for {args.n_features} of the most active features:\n")
    for fi in top_features:
        print(f"=== feature #{fi} (max act {max_per_feature[fi]:.2f}) ===")
        # Gather every (sentence, position, activation) for this feature, take top-k.
        hits = []
        for s_idx, (toks, feats) in enumerate(all_feats):
            col = feats[:, fi]
            for pos in range(col.shape[0]):
                hits.append((col[pos].item(), s_idx, pos))
        hits.sort(reverse=True)
        for act, s_idx, pos in hits[: args.top_k]:
            if act <= 0:
                continue
            toks = all_feats[s_idx][0]
            lo, hi = max(0, pos - 4), min(toks.shape[0], pos + 5)
            ctx = model.to_str_tokens(toks[lo:hi])
            peak = pos - lo
            rendered = "".join(
                f"【{t}】" if i == peak else t for i, t in enumerate(ctx)
            ).replace("\n", "\\n")
            print(f"   act={act:6.2f} | {rendered}")
        print()

    print("=" * 70)
    print("[smoke] SUCCESS — the stack works. Next:")
    print("  Phase 1:  python scripts/01_explore_pretrained_sae.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
