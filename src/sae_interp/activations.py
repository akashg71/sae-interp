"""Harvest and cache model activations at a hook point over a corpus.

The SAE never sees text directly — it sees the model's *internal activations* at one
site (e.g. ``blocks.8.hook_resid_pre``). This module:

1. loads a text corpus (default ``NeelNanda/pile-10k``),
2. tokenises it to a fixed context length,
3. runs it through the model with ``run_with_cache``, grabbing the activation at the
   configured hook point,
4. flattens (batch, pos, d_model) -> (n_tokens, d_model) and caches to disk.

We deliberately roll a simple, readable harvester on top of TransformerLens rather
than using SAELens' ``ActivationsStore`` so the data path is transparent (you can see
exactly what the SAE trains on). SAELens' store is a fine drop-in later if you want
streaming at larger scale.

Everything is device-agnostic and the on-disk cache is keyed by
(model, hook, ctx_len, n_sequences, dataset) so re-runs are instant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm

from .config import Config


def _cache_key(cfg: Config) -> str:
    """Stable hash of the settings that determine the activation contents."""
    payload = {
        "model": cfg.model.name,
        "hook": cfg.model.hook_name,
        "ctx_len": cfg.data.ctx_len,
        "n_sequences": cfg.data.n_sequences,
        "dataset": cfg.data.dataset,
        "dtype": cfg.data.store_dtype,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def cache_path(cfg: Config) -> Path:
    """Path to the cached activation tensor for this config."""
    cache_dir = Path(cfg.data.cache_dir)
    return cache_dir / f"acts_{_cache_key(cfg)}.pt"


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def load_corpus_tokens(cfg: Config, model) -> torch.Tensor:
    """Load + tokenise the corpus into a (n_sequences, ctx_len) int tensor.

    Uses the model's own tokenizer (via ``model.to_tokens``) so token ids line up
    with the model. Sequences shorter than ctx_len are skipped; longer ones are
    truncated. We prepend BOS (TransformerLens default) for consistency with how the
    pretrained SAE was trained.
    """
    from datasets import load_dataset  # imported lazily so `import activations` is cheap

    ds = load_dataset(cfg.data.dataset, split="train", streaming=True)

    seqs: list[torch.Tensor] = []
    ctx = cfg.data.ctx_len
    pbar = tqdm(total=cfg.data.n_sequences, desc="tokenising corpus")
    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        # to_tokens returns (1, seq) with BOS prepended by default.
        toks = model.to_tokens(text)
        if toks.shape[1] < ctx:
            continue
        seqs.append(toks[0, :ctx])
        pbar.update(1)
        if len(seqs) >= cfg.data.n_sequences:
            break
    pbar.close()

    if not seqs:
        raise RuntimeError(
            f"No sequences of length >= {ctx} found in {cfg.data.dataset}. "
            "Lower data.ctx_len or pick a different dataset."
        )
    return torch.stack(seqs, dim=0)  # (n_seq, ctx_len)


@torch.no_grad()
def harvest_activations(
    cfg: Config,
    model,
    device: torch.device,
    batch_size: int = 32,
    use_cache: bool = True,
) -> torch.Tensor:
    """Return a (n_tokens, d_model) tensor of activations at ``cfg.model.hook_name``.

    Caches to disk; subsequent calls with the same config load instantly. Pass
    ``use_cache=False`` to force re-harvest.
    """
    out_path = cache_path(cfg)
    if use_cache and out_path.exists():
        print(f"[activations] loading cached activations from {out_path}")
        return torch.load(out_path, map_location="cpu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    store_dtype = _torch_dtype(cfg.data.store_dtype)
    hook = cfg.model.hook_name

    tokens = load_corpus_tokens(cfg, model)  # (n_seq, ctx_len), on CPU
    n_seq = tokens.shape[0]

    chunks: list[torch.Tensor] = []
    model.eval()
    for start in tqdm(range(0, n_seq, batch_size), desc=f"harvesting {hook}"):
        batch = tokens[start : start + batch_size].to(device)
        # names_filter restricts the cache to just our hook -> far less memory.
        _, cache = model.run_with_cache(batch, names_filter=hook, return_type=None)
        acts = cache[hook]                       # (b, ctx_len, d_model)
        acts = acts.reshape(-1, acts.shape[-1])  # (b*ctx_len, d_model)
        chunks.append(acts.to("cpu", dtype=store_dtype))

    activations = torch.cat(chunks, dim=0)  # (n_tokens, d_model)
    if use_cache:
        torch.save(activations, out_path)
        print(f"[activations] cached {activations.shape[0]:,} activations to {out_path}")
    return activations


def iterate_activation_batches(
    activations: torch.Tensor,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    seed: int = 0,
    infinite: bool = True,
) -> Iterator[torch.Tensor]:
    """Yield (batch_size, d_model) float32 batches from a pool of activations.

    Used by the training loop. Re-shuffles each epoch. When ``infinite`` is True it
    loops forever (the trainer decides when to stop via step count); otherwise it
    yields one epoch.
    """
    n = activations.shape[0]
    g = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            yield activations[idx].to(device, dtype=torch.float32)
        if not infinite:
            return


def estimate_mean_activation(activations: torch.Tensor, n_sample: int = 100_000) -> torch.Tensor:
    """Mean activation vector over a sample — used to init the SAE's ``b_dec``."""
    n = activations.shape[0]
    if n > n_sample:
        idx = torch.randperm(n)[:n_sample]
        sample = activations[idx]
    else:
        sample = activations
    return sample.float().mean(dim=0)
