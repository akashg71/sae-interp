"""Causal interventions — does a feature actually *do* something?

Finding that a feature fires on "Python code" is correlational. To show it's causal we
edit the residual stream at the SAE's layer *during a forward pass* and watch the
effect on the model's outputs:

- **Ablate** feature j: remove its contribution ``f_j * W_dec[:, j]`` from the residual.
  If the feature mattered, downstream logits / generations change in a predictable way.
- **Clamp / steer** feature j: add ``alpha * W_dec[:, j]`` to the residual (i.e. force
  the feature on strongly). The model should steer toward the feature's concept.

Both are implemented as TransformerLens forward hooks on ``hook_name``. We edit in the
SAE's input space (the raw residual) using the SAE's own encode/decode so the edit is
consistent with how the feature is defined.

All functions take a SAELens SAE *or* our custom SAE — they only use ``.encode`` and
``W_dec``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InterventionResult:
    prompt: str
    baseline_text: str
    intervened_text: str
    feature_index: int
    mode: str            # "ablate" | "clamp"
    alpha: float | None  # steering strength for clamp


def _feature_direction(sae, feature_index: int, device) -> torch.Tensor:
    """Decoder direction for the feature, shape (d_model,).

    Handles both layouts:
      - Our custom SAE: W_dec is (d_model, d_sae) → column j
      - SAELens SAE:    W_dec is (d_sae, d_model)  → row j
    """
    W = sae.W_dec
    if W.shape[0] < W.shape[1]:   # (d_model, d_sae): custom SAE
        direction = W[:, feature_index]
    else:                          # (d_sae, d_model): SAELens SAE
        direction = W[feature_index]
    return direction.to(torch.float32).to(device)


def make_ablation_hook(sae, feature_index: int):
    """Hook that zeroes feature j's contribution to the residual.

    We re-encode the incoming residual to get f_j at each position, then subtract
    ``f_j * direction`` so only that feature's contribution is removed (other features
    and the reconstruction error are left untouched).
    """
    def hook(act, hook):  # noqa: A002
        shape = act.shape
        flat = act.reshape(-1, shape[-1]).to(torch.float32)
        f = sae.encode(flat)                                  # (n, d_sae)
        f_j = f[:, feature_index : feature_index + 1]         # (n, 1)
        direction = _feature_direction(sae, feature_index, flat.device)  # (d_model,)
        edited = flat - f_j * direction                       # remove its contribution
        return edited.reshape(shape).to(act.dtype)

    return hook


def make_clamp_hook(sae, feature_index: int, alpha: float):
    """Hook that steers by adding ``alpha * direction`` to the residual at every position.

    Positive ``alpha`` pushes the model toward the feature's concept. ``alpha`` is in
    units of the (unit-norm) decoder direction; typical useful values are a few× the
    feature's usual activation scale — sweep it.
    """
    def hook(act, hook):  # noqa: A002
        direction = _feature_direction(sae, feature_index, act.device)
        return (act.to(torch.float32) + alpha * direction).to(act.dtype)

    return hook


@torch.no_grad()
def generate_with_intervention(
    model,
    sae,
    prompt: str,
    hook_name: str,
    feature_index: int,
    mode: str = "clamp",
    alpha: float = 8.0,
    max_new_tokens: int = 30,
    device=None,
) -> InterventionResult:
    """Generate text with and without the intervention and return both.

    ``mode`` is "clamp" (steer toward the feature) or "ablate" (remove the feature).
    """
    model.eval()
    tokens = model.to_tokens(prompt)
    if device is not None:
        tokens = tokens.to(device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, verbose=False, do_sample=False)

    baseline = model.generate(tokens, **gen_kwargs)
    baseline_text = model.to_string(baseline[0])

    if mode == "ablate":
        hook = make_ablation_hook(sae, feature_index)
    elif mode == "clamp":
        hook = make_clamp_hook(sae, feature_index, alpha)
    else:
        raise ValueError(f"mode must be 'ablate' or 'clamp', got {mode!r}")

    with model.hooks(fwd_hooks=[(hook_name, hook)]):
        intervened = model.generate(tokens, **gen_kwargs)
    intervened_text = model.to_string(intervened[0])

    return InterventionResult(
        prompt=prompt,
        baseline_text=baseline_text,
        intervened_text=intervened_text,
        feature_index=feature_index,
        mode=mode,
        alpha=alpha if mode == "clamp" else None,
    )


@torch.no_grad()
def logit_diff_from_intervention(
    model,
    sae,
    prompt: str,
    hook_name: str,
    feature_index: int,
    mode: str = "clamp",
    alpha: float = 8.0,
    top_k: int = 10,
    device=None,
) -> dict:
    """Measure how the next-token distribution changes under the intervention.

    Returns the top-k tokens whose logit *increased* the most, which is a crisp,
    quantitative companion to the qualitative generation diff.
    """
    model.eval()
    tokens = model.to_tokens(prompt)
    if device is not None:
        tokens = tokens.to(device)

    clean_logits = model(tokens, return_type="logits")[0, -1].to(torch.float32)

    if mode == "ablate":
        hook = make_ablation_hook(sae, feature_index)
    else:
        hook = make_clamp_hook(sae, feature_index, alpha)

    with model.hooks(fwd_hooks=[(hook_name, hook)]):
        edited_logits = model(tokens, return_type="logits")[0, -1].to(torch.float32)

    delta = edited_logits - clean_logits
    vals, idx = torch.topk(delta, top_k)
    boosted = [(model.to_string(torch.tensor([i])), v) for i, v in zip(idx.tolist(), vals.tolist())]

    return {
        "prompt": prompt,
        "feature_index": feature_index,
        "mode": mode,
        "alpha": alpha if mode == "clamp" else None,
        "top_boosted_tokens": boosted,
    }
