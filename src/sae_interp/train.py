"""Custom SAE training loop.

We train our own :class:`~sae_interp.sae.SparseAutoencoder` rather than using a
black-box trainer — the point of the project is to show the mechanics. The loop:

- streams activation batches (from ``activations.iterate_activation_batches``),
- optimises ``MSE + λ·L1`` with Adam,
- keeps the decoder columns unit-norm via gradient projection + renormalisation,
- periodically **resamples dead features** (a known SAE failure mode: some features
  stop firing entirely and waste capacity; we re-initialise them toward activations
  the SAE currently reconstructs poorly),
- logs recon / L1 / L0 / variance-explained, optionally to Weights & Biases,
- checkpoints to ``results/checkpoints`` (gitignored).

Everything is seeded and device-agnostic. On a Mac this trains slowly; the same code
runs on a Colab/Kaggle T4 unchanged (just point it at a CUDA device).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .activations import iterate_activation_batches
from .config import Config
from .sae import SparseAutoencoder


@dataclass
class TrainHistory:
    """Lightweight in-memory log of training metrics (for plotting/notebooks)."""

    step: list[int] = field(default_factory=list)
    recon_loss: list[float] = field(default_factory=list)
    l1_loss: list[float] = field(default_factory=list)
    l0: list[float] = field(default_factory=list)
    variance_explained: list[float] = field(default_factory=list)
    dead_fraction: list[float] = field(default_factory=list)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _variance_explained(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    resid_var = (x - x_hat).var(dim=0, unbiased=False).sum()
    total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-8)
    return (1.0 - resid_var / total_var).item()


@torch.no_grad()
def _resample_dead_features(
    sae: SparseAutoencoder,
    dead_mask: torch.Tensor,
    recent_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Re-initialise dead features toward poorly-reconstructed inputs (Anthropic's recipe).

    For each dead feature we pick an input vector the SAE currently reconstructs badly
    (sampled ∝ squared error), point that feature's encoder/decoder at it, and reset
    its Adam optimiser state so it can learn afresh. Returns the number resampled.
    """
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    x_hat = sae.decode(sae.encode(recent_batch))
    sq_err = (recent_batch - x_hat).pow(2).sum(-1)        # (batch,)
    if sq_err.sum() <= 0:
        return 0
    probs = sq_err / sq_err.sum()
    choice = torch.multinomial(probs, num_samples=n_dead, replacement=True)
    new_dirs = recent_batch[choice]                        # (n_dead, d_model)

    dead_idx = torch.nonzero(dead_mask).flatten()
    # Decoder columns <- unit-normalised chosen inputs (new dictionary directions).
    unit = new_dirs / new_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sae.W_dec.data[:, dead_idx] = unit.T
    # Encoder rows <- same direction, scaled up a bit so the feature can fire.
    # (Scale by the norm of living encoder rows so it's in a sensible range.)
    alive_norm = sae.W_enc.data[~dead_mask].norm(dim=-1).mean().clamp_min(1e-3)
    sae.W_enc.data[dead_idx] = unit * alive_norm * 0.2
    sae.b_enc.data[dead_idx] = 0.0

    # Reset Adam moments for the resampled params so stale momentum doesn't fight us.
    for p in (sae.W_enc, sae.W_dec, sae.b_enc):
        state = optimizer.state.get(p, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                if p is sae.W_dec:
                    state[key][:, dead_idx] = 0.0
                else:
                    state[key][dead_idx] = 0.0
    return n_dead


def train_sae(
    cfg: Config,
    activations: torch.Tensor,
    device: torch.device,
    use_wandb: bool = False,
    l1_coeff: float | None = None,
    max_steps: int | None = None,
) -> tuple[SparseAutoencoder, TrainHistory]:
    """Train an SAE on a pool of activations and return (sae, history).

    ``l1_coeff`` / ``max_steps`` override the config — handy for the frontier sweep
    (same data, different λ) and for quick smoke runs (few steps).
    """
    set_seed(cfg.train.seed)
    l1 = cfg.sae.l1_coeff if l1_coeff is None else l1_coeff
    steps = cfg.train.steps if max_steps is None else max_steps

    sae = SparseAutoencoder(d_in=cfg.model.d_model, d_sae=cfg.d_sae, l1_coeff=l1).to(device)

    if cfg.sae.init_b_dec_to_mean:
        from .activations import estimate_mean_activation

        sae.set_decoder_bias(estimate_mean_activation(activations).to(device))

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.train.lr)
    batches = iterate_activation_batches(
        activations, cfg.train.batch_size, device, shuffle=True, seed=cfg.train.seed
    )

    history = TrainHistory()
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project="sae-interp",
            config={"l1_coeff": l1, "steps": steps, "d_sae": cfg.d_sae,
                    "lr": cfg.train.lr, "hook": cfg.model.hook_name},
        )

    # Track which features have fired recently, for dead-feature detection.
    last_fired = torch.zeros(cfg.d_sae, dtype=torch.long, device=device)
    ckpt_dir = Path(cfg.results_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, steps + 1):
        x = next(batches)
        out = sae(x)

        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        sae.remove_parallel_decoder_grad()   # keep decoder unit-norm: project grad...
        optimizer.step()
        sae.normalize_decoder()               # ...then correct any residual drift.

        # Update dead-feature tracker.
        fired = (out.feats > 0).any(dim=0)
        last_fired[fired] = step
        dead_mask = (step - last_fired) > cfg.train.dead_feature_window

        if cfg.train.resample_dead and step % cfg.train.dead_feature_window == 0 and step > 0:
            n = _resample_dead_features(sae, dead_mask, x, optimizer)
            if n:
                sae.normalize_decoder()
                last_fired[dead_mask] = step  # give them a fresh window
                print(f"[train] step {step}: resampled {n} dead features")

        if step % cfg.train.log_every == 0 or step == 1:
            ve = _variance_explained(x, out.x_hat)
            dead_frac = dead_mask.float().mean().item()
            history.step.append(step)
            history.recon_loss.append(out.recon_loss.item())
            history.l1_loss.append(out.l1_loss.item())
            history.l0.append(out.l0.item())
            history.variance_explained.append(ve)
            history.dead_fraction.append(dead_frac)
            rate = step / (time.time() - t0)
            print(
                f"[train] step {step:>6}/{steps} | recon {out.recon_loss.item():8.3f} "
                f"| L1 {out.l1_loss.item():7.4f} | L0 {out.l0.item():6.1f} "
                f"| VE {ve:5.3f} | dead {dead_frac:4.1%} | {rate:5.1f} it/s"
            )
            if run is not None:
                run.log({"recon_loss": out.recon_loss.item(), "l1_loss": out.l1_loss.item(),
                         "l0": out.l0.item(), "variance_explained": ve,
                         "dead_fraction": dead_frac}, step=step)

        if cfg.train.ckpt_every and step % cfg.train.ckpt_every == 0:
            sae.save(str(ckpt_dir / f"sae_l1{l1:.0e}_step{step}.pt"))

    final_path = ckpt_dir / f"sae_l1{l1:.0e}_final.pt"
    sae.save(str(final_path))
    print(f"[train] done in {time.time() - t0:.1f}s — saved {final_path}")
    if run is not None:
        run.finish()
    return sae, history


def overfit_sanity_check(cfg: Config, activations: torch.Tensor, device: torch.device,
                         n_batches: int = 4, steps: int = 500) -> float:
    """Sanity check: an SAE should drive recon loss ~0 on a few fixed batches.

    Returns the final reconstruction loss. If it doesn't drop a lot, something is
    wrong with the model/optimiser wiring before you spend money on a real run.
    """
    set_seed(cfg.train.seed)
    pool = activations[: n_batches * cfg.train.batch_size].to(device, dtype=torch.float32)
    sae = SparseAutoencoder(cfg.model.d_model, cfg.d_sae, l1_coeff=0.0).to(device)  # no L1: pure recon
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        out = sae(pool)
        opt.zero_grad(set_to_none=True)
        out.recon_loss.backward()
        sae.remove_parallel_decoder_grad()
        opt.step()
        sae.normalize_decoder()
        loss = out.recon_loss
    print(f"[sanity] overfit recon loss after {steps} steps: {loss.item():.4f}")
    return loss.item()
