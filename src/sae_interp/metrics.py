"""Metrics — the part that makes the project credible.

These quantify *how good* an SAE is along the two axes that matter:

- **Sparsity** (L0): how many features fire per token. Lower = sparser = more
  monosemantic-friendly.
- **Fidelity**: how well the reconstruction preserves the activation
  (variance-explained / MSE) AND, more importantly, the model's *behaviour*
  (CE-loss-recovered).

The headline deliverable is the **frontier**: sweep the L1 coefficient λ and plot
L0 (x) vs variance-explained or CE-recovered (y). A good SAE pushes that curve up-left.

These functions work for BOTH our custom ``SparseAutoencoder`` and a SAELens SAE,
because they only rely on ``.encode(x)`` / ``.decode(f)`` which both expose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class ReconMetrics:
    l0: float                    # mean active features per token
    mse: float                   # mean squared reconstruction error (summed over d, mean over batch)
    variance_explained: float    # 1 - Var(x - x_hat)/Var(x), in [~0, 1]
    n_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def reconstruction_metrics(sae, activations: torch.Tensor, device, batch_size: int = 4096) -> ReconMetrics:
    """Compute L0, MSE and variance-explained over a pool of activations.

    ``sae`` is anything with ``.encode`` and ``.decode``. ``activations`` is a
    (n_tokens, d_model) CPU tensor; we stream it through in batches.
    """
    n = activations.shape[0]
    total_l0 = 0.0
    total_mse = 0.0
    # For variance-explained we accumulate sum of squared residuals and the total
    # variance of x. We use the standard "fraction of variance unexplained" form.
    sum_sq_resid = 0.0
    sum_sq_centered = 0.0
    # Need the global mean of x for centering; estimate from a first pass-free trick:
    # accumulate sum and sumsq, then variance = E[x^2] - E[x]^2. Do it in one pass.
    feat_sum = torch.zeros(activations.shape[1], dtype=torch.float64)
    feat_sumsq = torch.zeros(activations.shape[1], dtype=torch.float64)

    seen = 0
    for start in range(0, n, batch_size):
        x = activations[start : start + batch_size].to(device, dtype=torch.float32)
        f = sae.encode(x)
        x_hat = sae.decode(f)

        total_l0 += (f > 0).float().sum(-1).sum().item()
        total_mse += (x_hat - x).pow(2).sum(-1).sum().item()
        sum_sq_resid += (x_hat - x).pow(2).sum().item()

        # Move to CPU *before* float64 — MPS doesn't support double.
        xb = x.detach().cpu().double()
        feat_sum += xb.sum(dim=0)
        feat_sumsq += xb.pow(2).sum(dim=0)
        seen += x.shape[0]

    mean = feat_sum / seen
    # Total variance summed over dims = sum_d (E[x_d^2] - mean_d^2) * N
    total_var = ((feat_sumsq / seen) - mean.pow(2)).clamp_min(0).sum().item() * seen
    variance_explained = 1.0 - (sum_sq_resid / total_var) if total_var > 0 else float("nan")

    return ReconMetrics(
        l0=total_l0 / seen,
        mse=total_mse / seen,
        variance_explained=variance_explained,
        n_tokens=seen,
    )


@torch.no_grad()
def ce_loss_recovered(
    model,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    device,
    batch_size: int = 8,
) -> dict:
    """CE-loss-recovered: does patching x_hat back into the model preserve behaviour?

    We measure next-token cross-entropy under three conditions at the SAE's layer:

    - **clean**: the model untouched (lower bound on loss).
    - **sae**:   replace the residual at ``hook_name`` with the SAE reconstruction.
    - **ablate**: replace it with the *mean* activation (a behaviour-destroying baseline).

    Then::

        recovered = (loss_ablate - loss_sae) / (loss_ablate - loss_clean)

    1.0 means the SAE reconstruction is as good as the real activations; 0.0 means it's
    no better than mean-ablation. This is the metric that shows the SAE captures
    *function*, not just numbers.

    ``tokens`` is (n_seq, ctx_len). Loss is computed by TransformerLens with
    ``return_type="loss"`` (mean CE over predicted positions).
    """
    model.eval()

    def run_clean(batch):
        return model(batch, return_type="loss").item()

    def run_with_replacement(batch, replace_fn):
        def hook(act, hook):  # noqa: A002 - TransformerLens passes a `hook` arg
            return replace_fn(act)
        return model.run_with_hooks(
            batch, return_type="loss", fwd_hooks=[(hook_name, hook)]
        ).item()

    # Estimate the mean activation for the ablation baseline from the first batch.
    first = tokens[:batch_size].to(device)
    _, cache = model.run_with_cache(first, names_filter=hook_name, return_type=None)
    mean_act = cache[hook_name].reshape(-1, cache[hook_name].shape[-1]).mean(0)

    def sae_replace(act):
        shape = act.shape
        flat = act.reshape(-1, shape[-1]).to(torch.float32)
        recon = sae.decode(sae.encode(flat)).to(act.dtype)
        return recon.reshape(shape)

    def mean_replace(act):
        return mean_act.to(act.dtype).expand_as(act)

    n = tokens.shape[0]
    losses = {"clean": 0.0, "sae": 0.0, "ablate": 0.0}
    nb = 0
    for start in range(0, n, batch_size):
        batch = tokens[start : start + batch_size].to(device)
        losses["clean"] += run_clean(batch)
        losses["sae"] += run_with_replacement(batch, sae_replace)
        losses["ablate"] += run_with_replacement(batch, mean_replace)
        nb += 1

    for k in losses:
        losses[k] /= max(nb, 1)

    denom = losses["ablate"] - losses["clean"]
    recovered = (losses["ablate"] - losses["sae"]) / denom if denom != 0 else float("nan")
    return {
        "loss_clean": losses["clean"],
        "loss_sae": losses["sae"],
        "loss_ablate": losses["ablate"],
        "ce_loss_recovered": recovered,
    }


def frontier_point(recon: ReconMetrics, ce: dict | None, l1_coeff: float) -> dict:
    """Bundle one (λ) sweep point for plotting/saving."""
    point = {"l1_coeff": l1_coeff, **recon.as_dict()}
    if ce is not None:
        point.update(ce)
    return point


def plot_frontier(points: list[dict], out_path: str, y_key: str = "variance_explained") -> None:
    """Plot L0 (x) vs a fidelity metric (y) and save to ``out_path``.

    ``points`` is a list of dicts as produced by :func:`frontier_point`. Each point is
    one trained SAE at a different λ. We sort by L0 so the curve reads left-to-right.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless-safe (no display needed)
    import matplotlib.pyplot as plt

    pts = sorted(points, key=lambda p: p["l0"])
    xs = [p["l0"] for p in pts]
    ys = [p[y_key] for p in pts]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, "o-", color="#2b6cb0")
    for p in pts:
        ax.annotate(f"λ={p['l1_coeff']:.0e}", (p["l0"], p[y_key]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("L0 (mean active features per token)  →  sparser is left")
    ax.set_ylabel(y_key.replace("_", " "))
    ax.set_title("Sparsity ↔ fidelity frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] saved frontier plot to {out_path}")
