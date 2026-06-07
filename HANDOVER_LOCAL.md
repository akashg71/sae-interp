# Handover — continuing on the personal laptop

This repo was scaffolded and verified on a different machine where **Hugging Face was
network-blocked**, so the Phase 0 smoke test could not be run there. Everything that
doesn't need the network is done and checked. You're picking it up on a personal laptop
(or Colab) where Hugging Face IS reachable.

## Where things stand

- ✅ Full repo structure, `pyproject.toml`, `requirements.txt` (+ exact `requirements.lock.txt`).
- ✅ All core modules implemented and **unit-checked offline** (custom SAE trains, decoder
  stays unit-norm, metrics/save-load work; feature-analysis + clamp/ablate + CE-recovered
  logic exercised against a mock model).
- ✅ Phase scripts `00`–`04`, two Colab notebooks, optional Streamlit app, README + RUNBOOK.
- ⛔️ **Not yet run:** anything that downloads from Hugging Face (smoke test + Phases 1–4),
  because the build machine blocked `huggingface.co`. This is the very next thing to do.

## Verified library facts (already baked into the code)

- `transformer-lens 3.3.0`, `sae-lens 6.44.2`, `torch 2.12.0`, Python 3.11.
- `SAE.from_pretrained(release, sae_id, device, dtype)` returns **just the SAE** (scripts
  also handle the old tuple form).
- Pretrained SAE `gpt2-small-res-jb` / `blocks.8.hook_resid_pre` confirmed to exist.
- MPS has no float64 (handled in metrics).

## First steps here (do these in order)

```bash
# 1. Environment
python3.11 -m venv .venv && source .venv/bin/activate    # or use Colab's python
pip install -r requirements.txt
pip install -e .

# 2. Confirm Hugging Face is reachable (should be 200, not 403)
curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models/gpt2

# 3. THE GATE — run the smoke test (downloads GPT-2 ~500MB + one SAE)
python scripts/00_smoke_test.py
```

If the smoke test prints top-activating tokens, continue with Phases 1→4 per **RUNBOOK.md**.
Phase 3 (training) wants a GPU — run it on Colab/Kaggle (notebooks are ready).

## Notes for the next agent

- Read `README.md` (esp. "Network / corporate proxy") and `RUNBOOK.md` first — they have
  the full plan and every command.
- `truststore` is a no-op off a corporate network; harmless.
- The project is now entirely personal — no corporate data or accounts are involved.
