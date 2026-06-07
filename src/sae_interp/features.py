"""Feature analysis — find out what an SAE feature *means*.

Three complementary tools:

1. :func:`max_activating_examples` — the workhorse. For a feature index j, run a
   corpus through model+SAE, record f_j at every token, and return the token
   positions where it fires hardest, with surrounding context. Reading these is how
   you label a feature ("fires on Python code", "fires on the token ' the' after a
   preposition", etc.).

2. :func:`feature_density` — what fraction of tokens a feature fires on. Very dense
   features (>~10%) are usually not monosemantic; density 0 means a dead feature.

3. :func:`logit_lens_tokens` — project a feature's decoder direction through the
   unembedding (W_U) to see which output tokens it most increases. A quick, cheap
   hint at meaning that complements the max-activating examples.

Works with both our custom SAE and a SAELens SAE (only needs ``.encode`` and a
``W_dec`` of shape (d_model, d_sae)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class ActivatingExample:
    seq_index: int            # which sequence in the corpus
    token_index: int          # position of the peak-activating token in that sequence
    activation: float         # the feature activation value at the peak
    tokens: list[str]         # the context window (decoded tokens)
    peak_in_window: int       # index of the peak token within `tokens`


@torch.no_grad()
def collect_feature_activations(
    model,
    sae,
    tokens: torch.Tensor,
    hook_name: str,
    feature_indices: list[int],
    device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Return a (n_seq, ctx_len, n_features) tensor of activations for the chosen features.

    We run the model once, grab the residual at ``hook_name``, encode with the SAE,
    and keep only the requested feature columns (keeps memory bounded even for a
    16k-wide SAE).
    """
    feat_idx = torch.tensor(feature_indices, device=device)
    n_seq, ctx = tokens.shape
    out = torch.zeros(n_seq, ctx, len(feature_indices), dtype=torch.float32)

    model.eval()
    for start in tqdm(range(0, n_seq, batch_size), desc="scoring features"):
        batch = tokens[start : start + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name, return_type=None)
        acts = cache[hook_name].to(torch.float32)         # (b, ctx, d_model)
        f = sae.encode(acts.reshape(-1, acts.shape[-1]))  # (b*ctx, d_sae)
        f = f.reshape(acts.shape[0], ctx, -1)             # (b, ctx, d_sae)
        out[start : start + batch.shape[0]] = f.index_select(-1, feat_idx).cpu()
    return out


def max_activating_examples(
    model,
    feature_acts: torch.Tensor,   # (n_seq, ctx, n_features) from collect_feature_activations
    tokens: torch.Tensor,         # (n_seq, ctx)
    feature_col: int,             # which column of feature_acts (0..n_features-1)
    top_k: int = 10,
    window: int = 8,
) -> list[ActivatingExample]:
    """Top-k token positions where the chosen feature fires hardest, with context.

    ``window`` is how many tokens of context to show on each side of the peak.
    """
    col = feature_acts[:, :, feature_col]                 # (n_seq, ctx)
    n_seq, ctx = col.shape
    flat = col.reshape(-1)
    k = min(top_k, flat.numel())
    top_vals, top_flat_idx = torch.topk(flat, k)

    examples: list[ActivatingExample] = []
    for val, flat_i in zip(top_vals.tolist(), top_flat_idx.tolist()):
        if val <= 0:
            continue  # feature never fired this hard — skip dead/inactive hits
        s = flat_i // ctx
        t = flat_i % ctx
        lo = max(0, t - window)
        hi = min(ctx, t + window + 1)
        ctx_tokens = model.to_str_tokens(tokens[s, lo:hi])
        examples.append(
            ActivatingExample(
                seq_index=s,
                token_index=t,
                activation=val,
                tokens=ctx_tokens,
                peak_in_window=t - lo,
            )
        )
    return examples


def format_example(ex: ActivatingExample) -> str:
    """Render one example as a one-line string with the peak token wrapped in 【…】."""
    parts = []
    for i, tok in enumerate(ex.tokens):
        tok = tok.replace("\n", "\\n")
        parts.append(f"【{tok}】" if i == ex.peak_in_window else tok)
    return f"act={ex.activation:6.2f} | " + "".join(parts)


def feature_density(feature_acts: torch.Tensor, feature_col: int) -> float:
    """Fraction of tokens on which the feature is active (> 0)."""
    col = feature_acts[:, :, feature_col]
    return (col > 0).float().mean().item()


@torch.no_grad()
def logit_lens_tokens(model, sae, feature_index: int, top_k: int = 10) -> list[tuple[str, float]]:
    """Project a feature's decoder direction through the unembedding.

    ``W_dec[:, j]`` is the direction this feature writes into the residual stream.
    Pushing it through ``W_U`` (and the final layernorm's scale, approximately) tells
    us which vocabulary logits it most increases — a cheap semantic hint.

    Returns the top-k (token_string, logit_contribution) pairs.
    """
    # W_dec is (d_model, d_sae) for both our SAE and SAELens SAEs.
    direction = sae.W_dec[:, feature_index].to(torch.float32)  # (d_model,)
    # Approximate logit-lens: direction @ W_U. (We skip the final LN scale; this is a
    # hint, not an exact attribution.) W_U is (d_model, d_vocab) in TransformerLens.
    W_U = model.W_U.to(torch.float32)
    logits = direction @ W_U                                    # (d_vocab,)
    vals, idx = torch.topk(logits, top_k)
    toks = [model.to_string(torch.tensor([i])) for i in idx.tolist()]
    return list(zip(toks, vals.tolist()))


def find_dead_features(feature_acts: torch.Tensor, threshold: float = 0.0) -> list[int]:
    """Indices (into the feature_acts columns) of features that never fire."""
    never = (feature_acts.amax(dim=(0, 1)) <= threshold)
    return torch.nonzero(never).flatten().tolist()
