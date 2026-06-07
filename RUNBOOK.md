# RUNBOOK — how to operate this project

A short, practical "what do I type" guide. Assumes you're a strong engineer but new to
the ML tooling. For the *why*, see README.md and the docstrings in `src/sae_interp/`.

---

## 0. One-time setup

```bash
cd "<this folder>"
source .venv/bin/activate          # the venv is already created (Python 3.11)
pip install -r requirements.txt    # if starting fresh
pip install -e .                   # makes `import sae_interp` work
cp .env.example .env               # optional: add your HF_TOKEN
```

Quick health check (no network needed):

```bash
python -c "from sae_interp.config import load_config; from sae_interp import device; \
print('config OK', load_config('configs/sae_gpt2small.yaml').d_sae, device.get_device())"
```

---

## ⚠️ Before anything that touches Hugging Face

Phases 0–4 download GPT-2 / the SAE / the corpus from `huggingface.co`. On the
corporate network that host is **blocked** (Netskope "Generative AI – Block", HTTP 403).
You must first do **one** of:

- **Get the exception:** request `huggingface.co` + CDN (`*.hf.co`,
  `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`) via the **Fresh Service
  Catalogue** (InfoSec + DNA review), **or**
- **Use a different network:** personal machine, or **Colab/Kaggle** (recommended for
  Phase 3 training anyway — the notebooks are Colab-ready).

Test reachability quickly:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models/gpt2
# 200 = reachable.  403 = still blocked (X-Direct-Response block page).
```

The TLS/cert side is already handled by `truststore` (installed) — you should *not* see
`CERTIFICATE_VERIFY_FAILED`. If you do, ensure `truststore` is installed and you didn't
set `SAE_INTERP_NO_TRUSTSTORE=1`.

---

## 1. Phase 0 — smoke test (the gate)

```bash
python scripts/00_smoke_test.py
```
First run downloads GPT-2 (~500 MB) + one SAE (tens of MB), then prints top-activating
tokens for a few features. If you see that without errors, you're good. Cached
afterwards (fast + offline).

## 2. Phase 1 — explore features

```bash
python scripts/01_explore_pretrained_sae.py --auto 20        # auto-pick 20 active features
python scripts/01_explore_pretrained_sae.py --features 12 100 --top-k 12   # specific ones
```
Writes `results/features.md` — a table to annotate. Read the examples, label 3–5 clearly
interpretable features, note their indices for Phase 2.

## 3. Phase 2 — causal validation

```bash
python scripts/02_causal_intervention.py --features 12 100
python scripts/02_causal_intervention.py --features 12 --alpha 12 --prompt "I went to the"
```
Look for: clamping steers generations toward the feature's concept; ablating removes it.
Capture good before/after pairs for the writeup. Sweep `--alpha` if the effect is weak.

## 4. Phase 3 — train your own SAE  (do this on a GPU)

Local quick check (slow but proves the loop):
```bash
python scripts/03_train_sae.py --steps 2000 --sanity
```
`--sanity` first overfits a few batches (recon loss should plummet) to confirm wiring.

Full run on Colab/Kaggle: open `notebooks/01_feature_exploration.ipynb` style setup,
or clone the repo and run `scripts/03_train_sae.py` there with the GPU runtime. Scale up
`configs/sae_gpt2small.yaml`: `data.n_sequences` (more activation tokens),
`train.steps`, and later `sae.expansion_factor` (4 → 8/16). Add `--wandb` for logging
(set `WANDB_API_KEY`).

Checkpoints land in `results/checkpoints/` (gitignored).

## 5. Phase 4 — the frontier (headline result)

```bash
python scripts/04_eval_frontier.py --steps 3000          # L0 vs variance-explained
python scripts/04_eval_frontier.py --steps 3000 --ce     # also CE-loss-recovered (slower)
```
Writes `results/frontier_points.json` and `results/figures/frontier.png`. Tune the λ
sweep via `--lambdas 1e-4 5e-4 1e-3 ...` or `metrics.l1_sweep` in the config.

## 6. Phase 5 — writeup

Use `notebooks/02_results_figures.ipynb` to regenerate figures, fill in
`results/features.md`, and write up method + frontier + causal results + limitations.

Optional UI: `streamlit run app/feature_browser.py`.

---

## Common knobs (configs/sae_gpt2small.yaml)

| Want to... | Change |
|------------|--------|
| Run faster while iterating | lower `data.n_sequences`, `train.steps` |
| Sparser SAE (fewer active features) | raise `sae.l1_coeff` (λ) |
| Bigger dictionary | raise `sae.expansion_factor` (4 → 8/16) |
| Different layer/site | `model.layer` + `model.hook_name` (keep in sync) **and** a matching `pretrained_sae.sae_id` |
| Use a CPU even with a GPU present | `get_device(prefer="cpu")` in code |

## Troubleshooting

- **`CERTIFICATE_VERIFY_FAILED`** → `truststore` missing/disabled; reinstall requirements.
- **HTTP 403 / block page from huggingface.co** → network block; see the ⚠️ section.
- **`Cannot convert a MPS Tensor to float64`** → already handled in metrics; if it
  appears elsewhere, move the tensor to CPU before `.double()`.
- **MPS slow / op unsupported** → set `PYTORCH_ENABLE_MPS_FALLBACK=1`, or train on GPU.
- **Hook name not found** → run `print([h for h in model.hook_dict if 'resid' in h])` and
  pick the right one; update `model.hook_name`.
