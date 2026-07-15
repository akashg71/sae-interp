#!/usr/bin/env python
"""Phase 2 — causal validation of features.

For chosen features, ABLATE (remove) and CLAMP (steer) them via TransformerLens hooks
and observe the effect on generations and next-token logits. This turns a
correlational claim ("feature fires on X") into a causal one ("forcing the feature on
makes the model produce X").

Run:
    python scripts/02_causal_intervention.py --features 12 100
    python scripts/02_causal_intervention.py --features 12 --alpha 12 --prompt "I went to the"
"""

from __future__ import annotations

import argparse

from sae_interp import net

net.bootstrap()

from sae_interp.config import load_config  # noqa: E402
from sae_interp.device import get_device  # noqa: E402
from sae_interp import interventions as IV  # noqa: E402


DEFAULT_PROMPTS = [
    "The weather today is",
    "I went to the shop to buy",
    "The police arrested the",
    "After surgery, patients with severe",
    "At the border, the officials saw",
    "The new policy will affect",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 — causal interventions")
    p.add_argument("--config", default="configs/sae_gpt2small.yaml")
    p.add_argument("--features", type=int, nargs="+", required=True,
                   help="feature indices to intervene on (from Phase 1)")
    p.add_argument("--prompt", action="append", default=None,
                   help="prompt(s) to test; repeatable. Defaults to a built-in set.")
    p.add_argument("--alpha", type=float, default=8.0, help="clamp/steer strength")
    p.add_argument("--max-new-tokens", type=int, default=30)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    prompts = args.prompt or DEFAULT_PROMPTS

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(cfg.model.name, device=str(device))
    model.eval()
    sae = SAE.from_pretrained(cfg.pretrained_sae.release, cfg.pretrained_sae.sae_id, device=str(device))
    if isinstance(sae, tuple):
        sae = sae[0]

    hook = cfg.model.hook_name
    for fi in args.features:
        print("\n" + "#" * 72)
        print(f"# FEATURE {fi}")
        print("#" * 72)

        # Logit-diff view for EVERY prompt (A1: multi-prompt steering table).
        print(f"\n{'─'*60}")
        print(f"  BOOSTED-TOKEN TABLE  (clamp α={args.alpha})")
        print(f"{'─'*60}")
        for prompt in prompts:
            ld = IV.logit_diff_from_intervention(
                model, sae, prompt, hook, fi, mode="clamp", alpha=args.alpha, device=device
            )
            print(f"\n  prompt: {prompt!r}")
            for tok, val in ld["top_boosted_tokens"]:
                print(f"    {val:+6.2f}  {tok!r}")

        # Generation view: clamp vs baseline on each prompt.
        print(f"\n{'─'*60}")
        print(f"  GENERATION DIFFS")
        print(f"{'─'*60}")
        for prompt in prompts:
            res = IV.generate_with_intervention(
                model, sae, prompt, hook, fi, mode="clamp",
                alpha=args.alpha, max_new_tokens=args.max_new_tokens, device=device
            )
            print(f"\n  prompt:   {prompt!r}")
            print(f"  baseline: {res.baseline_text!r}")
            print(f"  clamped:  {res.intervened_text!r}")

        # Ablation view on the first prompt.
        abl = IV.generate_with_intervention(
            model, sae, prompts[0], hook, fi, mode="ablate",
            max_new_tokens=args.max_new_tokens, device=device
        )
        print(f"\n  [ablate] prompt: {prompts[0]!r}")
        print(f"  baseline: {abl.baseline_text!r}")
        print(f"  ablated:  {abl.intervened_text!r}")

    print("\n[phase2] done. If clamping steers generations toward the feature's concept,")
    print("         that's your causal result. Capture good before/after pairs for the writeup.")


if __name__ == "__main__":
    main()
