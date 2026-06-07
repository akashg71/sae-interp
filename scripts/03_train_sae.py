#!/usr/bin/env python
"""Phase 3 — train your own SAE.

Harvest residual-stream activations from GPT-2-small, then train the custom L1 ReLU
SAE in src/sae_interp/sae.py on them. This is the part that demonstrates you
understand SAEs (the model and training loop are hand-written, not from a library).

This is the slow phase. On a Mac (MPS/CPU) it's tolerable for a small run; for a real
run use a free GPU (Colab/Kaggle) — the code is device-agnostic, just set the device
to cuda there. See notebooks/ for a Colab-runnable path.

Run (local quick check):
    python scripts/03_train_sae.py --steps 2000 --sanity
Run (full):
    python scripts/03_train_sae.py --wandb
"""

from __future__ import annotations

import argparse

from sae_interp import net

net.bootstrap()

from sae_interp.config import load_config  # noqa: E402
from sae_interp.device import get_device  # noqa: E402
from sae_interp.activations import harvest_activations  # noqa: E402
from sae_interp.train import train_sae, overfit_sanity_check  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 — train a custom SAE")
    p.add_argument("--config", default="configs/sae_gpt2small.yaml")
    p.add_argument("--steps", type=int, default=None, help="override train.steps")
    p.add_argument("--l1", type=float, default=None, help="override sae.l1_coeff (lambda)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--sanity", action="store_true",
                   help="run the overfit-a-few-batches sanity check first")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"[phase3] device={device}  d_sae={cfg.d_sae}  l1={args.l1 or cfg.sae.l1_coeff}")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(cfg.model.name, device=str(device))
    model.eval()

    print("[phase3] harvesting activations (cached after first run)...")
    activations = harvest_activations(cfg, model, device)
    print(f"[phase3] activation pool: {tuple(activations.shape)}")

    # Free the model's memory before training the SAE (we don't need it during training).
    del model
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.sanity:
        overfit_sanity_check(cfg, activations, device)

    sae, history = train_sae(
        cfg, activations, device, use_wandb=args.wandb,
        l1_coeff=args.l1, max_steps=args.steps,
    )
    print("[phase3] training complete. Final metrics:")
    if history.step:
        print(f"   L0={history.l0[-1]:.1f}  VE={history.variance_explained[-1]:.3f}  "
              f"recon={history.recon_loss[-1]:.3f}  dead={history.dead_fraction[-1]:.1%}")
    print("\nNext: evaluate the frontier across lambdas:")
    print("  python scripts/04_eval_frontier.py")


if __name__ == "__main__":
    main()
