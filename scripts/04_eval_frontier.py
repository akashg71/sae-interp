#!/usr/bin/env python
"""Phase 4 — metrics & the sparsity↔fidelity frontier (the headline result).

Trains a small SAE at each lambda in metrics.l1_sweep (or evaluates existing
checkpoints), measures L0 + variance-explained + (optionally) CE-loss-recovered, and
plots L0 vs fidelity. The up-and-to-the-left this curve sits, the better the SAE.

Run (train + sweep; reduce --steps for a quick pass):
    python scripts/04_eval_frontier.py --steps 3000
Run with CE-loss-recovered (slower; needs the model in memory):
    python scripts/04_eval_frontier.py --steps 3000 --ce
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sae_interp import net

net.bootstrap()

from sae_interp.config import load_config  # noqa: E402
from sae_interp.device import get_device  # noqa: E402
from sae_interp.activations import harvest_activations, load_corpus_tokens  # noqa: E402
from sae_interp.train import train_sae  # noqa: E402
from sae_interp import metrics as M  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 — frontier sweep")
    p.add_argument("--config", default="configs/sae_gpt2small.yaml")
    p.add_argument("--steps", type=int, default=3000, help="training steps per lambda")
    p.add_argument("--ce", action="store_true", help="also compute CE-loss-recovered (slower)")
    p.add_argument("--lambdas", type=float, nargs="*", default=None,
                   help="override the lambda sweep from the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    lambdas = args.lambdas or cfg.metrics.l1_sweep
    print(f"[phase4] device={device}  lambdas={lambdas}  steps/lambda={args.steps}")

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(cfg.model.name, device=str(device))
    model.eval()

    activations = harvest_activations(cfg, model, device)
    # Hold out a slice for evaluation so we don't report train-set numbers.
    n = activations.shape[0]
    n_eval = min(50_000, n // 5)
    eval_acts = activations[-n_eval:]
    train_acts = activations[:-n_eval]
    eval_tokens = load_corpus_tokens(cfg, model)[: cfg.metrics.eval_batches] if args.ce else None

    points = []
    for lam in lambdas:
        print(f"\n[phase4] === lambda = {lam:.0e} ===")
        sae, _ = train_sae(cfg, train_acts, device, l1_coeff=lam, max_steps=args.steps)
        recon = M.reconstruction_metrics(sae, eval_acts, device)
        ce = None
        if args.ce:
            ce = M.ce_loss_recovered(model, sae, eval_tokens, cfg.model.hook_name, device)
        pt = M.frontier_point(recon, ce, lam)
        points.append(pt)
        ce_str = f"  CE-recovered={ce['ce_loss_recovered']:.3f}" if ce else ""
        print(f"[phase4] lambda {lam:.0e}: L0={recon.l0:.1f}  VE={recon.variance_explained:.3f}{ce_str}")

    # Save raw points + plots.
    results_dir = Path(cfg.results_dir)
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (results_dir / "frontier_points.json").open("w") as fh:
        json.dump(points, fh, indent=2)
    M.plot_frontier(points, str(figures / "frontier.png"), y_key="variance_explained")
    if args.ce:
        M.plot_frontier(points, str(figures / "frontier_ce.png"), y_key="ce_loss_recovered")

    print(f"\n[phase4] wrote results/frontier_points.json and figures/frontier.png")
    print("This frontier plot is the headline figure for the writeup.")


if __name__ == "__main__":
    main()
