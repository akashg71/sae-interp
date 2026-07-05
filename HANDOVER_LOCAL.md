# Handover — continuing on the personal laptop

> Regenerated 2026-07-05. Supersedes the original scaffold-era handover.
> You are an agent picking this project up on Akash's **personal laptop**, where
> Hugging Face is reachable. The other machine (work laptop) blocks `huggingface.co`
> (Netskope, HTTP 403), which shapes the priority tasks below.

## Who you're working with

Akash is a strong Ruby engineer (Simply Business day job — thinks in `binding.pry`,
overmind, bundler) who is **newer to Python and ML tooling**. He learns best by
dropping into a live console and inspecting values, not by reading walls of theory.

- Map concepts to Ruby: `breakpoint()`/ipdb ≈ `binding.pry`, venv ≈ per-project bundler.
- Always give copy-paste commands, including `source .venv/bin/activate` — plain
  `python` is not on his PATH.
- He's using this project to learn mech-interp AND modern Python. Explain as you go.

## Where things stand (see git log)

- ✅ Phase 0–2 done: smoke test, feature exploration (`results/features.md`, labelled),
  causal interventions (commit `62babfe`).
- ✅ Phase 3–4 progressed on Colab: SAE expansion 4×→8×, training dynamics documented,
  outputs in `colab-output/op1.md`, `op2.md` (commits `40e4426`, `74b53f9`).
- ✅ `scripts/99_pdb_practice.py` — network-free debugger tutorial that mimics the
  corpus loop in `src/sae_interp/activations.py::load_corpus_tokens`, using **real**
  GPT-2 tokenisation via `tiktoken` (same BPE vocab; the tiktoken vocab downloads from
  an Azure blob, so it worked even on the blocked network).
- Remaining: Phase 4 frontier eval polish / Phase 5 writeup, per `RUNBOOK.md`.
- The work laptop has uncommitted scratch edits (a `breakpoint()` in `activations.py`,
  `activations copy.py`, README tweaks). They intentionally stay there — don't look
  for them here.

## Setup on this laptop

```bash
git clone https://github.com/akashg71/sae-interp.git && cd sae-interp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install ipdb tiktoken        # debugger console + tokenizer for the practice script

# Hugging Face should be reachable here (expect 200):
curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models/gpt2

# First run downloads GPT-2 (~500 MB) + one pretrained SAE:
python scripts/00_smoke_test.py
```

## Priority task 1 — commit a real activation sample for offline use

The work laptop can never download GPT-2, so save a small **real** sample there to
explore offline. Run this here, then commit + push:

```python
import torch
from transformer_lens import HookedTransformer
from sae_interp.config import load_config
from sae_interp.activations import load_corpus_tokens

cfg = load_config("configs/sae_gpt2small.yaml")
model = HookedTransformer.from_pretrained("gpt2")
cfg.data.n_sequences = 8                                  # tiny sample, git-friendly
tokens = load_corpus_tokens(cfg, model)                   # (8, ctx_len) real pile-10k tokens
_, cache = model.run_with_cache(tokens, names_filter=cfg.model.hook_name, return_type=None)
torch.save(
    {"tokens": tokens, "acts": cache[cfg.model.hook_name].half().cpu()},  # ~1.6 MB
    "colab-output/sample_batch.pt",
)
```

Commit as e.g. `chore(data): add real activation sample for offline exploration`.

## Priority task 2 — teach the codebase through the debugger

Akash wants to *understand* this code, pry-style. Since HF works here, run the real
pipeline under the debugger:

1. Add `breakpoint()` inside `load_corpus_tokens` in `src/sae_interp/activations.py`
   (after `toks = model.to_tokens(text)`).
2. `PYTHONBREAKPOINT=ipdb.set_trace python scripts/03_train_sae.py --steps 50 --sanity`
3. At the prompt, show him: `toks.shape`, `model.to_str_tokens(text)[:10]`, walking the
   call stack with `u`/`d`, `interact` for a full REPL. Then a second breakpoint in
   `harvest_activations` to inspect `cache[hook].shape` — (batch, ctx, d_model).
4. **Remove all `breakpoint()` lines before committing** (grep for them).

`scripts/99_pdb_practice.py` is the offline warm-up version of the same lesson —
its comments list the console commands to try.

## Conventions (non-negotiable)

- **Conventional Commits** (`feat:`, `fix:`, `chore:`, `docs:`, …), imperative mood.
- **No `Co-Authored-By` trailer**, no watermarks — Akash is the sole author.
- Read `RUNBOOK.md` (what to type), `NOTES_OFFLINE.md` (guided tour + Python refresher),
  and `CONCEPTS.md` (mech-interp glossary) before changing code.
- Checkpoints land in `results/checkpoints/` (gitignored). Phase 3 full training wants
  a GPU — Colab notebooks in `notebooks/` are ready for that.
