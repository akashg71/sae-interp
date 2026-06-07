# SAE Interpretability — GPT-2-small

A mechanistic-interpretability project: train a **sparse autoencoder (SAE)** on the
internal activations of GPT-2-small, find **interpretable features**, and **causally
validate** them by clamping/ablating those features and watching the model's behaviour
change. The deliverable is a clean repo + a writeup with real metrics (the
sparsity↔reconstruction frontier and a causal intervention result).

> **Status:** scaffold complete; core code implemented and unit-checked offline.
> The Phase 0 smoke test (which downloads GPT-2 + a pretrained SAE) has **not** been
> run yet because Hugging Face is blocked on the build network — see
> [Network / corporate proxy](#network--corporate-proxy-important) below. Everything
> that doesn't need the network has been verified to work.

---

## What's here

```
.
├── configs/sae_gpt2small.yaml   # all knobs: model, hook, d_sae, lambda, lr, steps...
├── src/sae_interp/              # the library (importable as `sae_interp`)
│   ├── config.py                #   typed YAML config loader
│   ├── device.py                #   get_device(): cuda -> mps -> cpu
│   ├── net.py                   #   .env loading + OS-trust-store TLS fix (corporate proxies)
│   ├── activations.py           #   harvest + cache model activations at a hook point
│   ├── sae.py                   #   the SAE (encoder/decoder, ReLU, L1, unit-norm decoder)
│   ├── train.py                 #   custom training loop + dead-feature resampling
│   ├── metrics.py               #   L0, variance-explained, CE-loss-recovered, frontier
│   ├── features.py              #   max-activating examples, density, logit-lens naming
│   └── interventions.py         #   clamp / ablate features via TransformerLens hooks
├── scripts/                     # one runnable script per phase (00-04)
├── notebooks/                   # Colab/Kaggle-runnable exploration + figures
├── app/feature_browser.py       # optional Streamlit feature browser
└── results/                     # figures + annotated feature table (filled in later)
```

The five phases (see [the plan](#the-plan)) build on each other; each script prints
what to run next.

---

## Setup

Verified working: **Python 3.11.15**, installed via Homebrew (`brew install python@3.11`).
(The system Python 3.9 is too old; 3.13 risks ML-lib lag. 3.10–3.11 is the sweet spot.)

```bash
# 1. Create + activate the virtualenv (already created at .venv)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt        # resolved versions
# exact pins are recorded in requirements.lock.txt (pip install -r that for reproducibility)

# 3. Editable-install the package so `import sae_interp` works everywhere
pip install -e .

# 4. Secrets (optional but recommended)
cp .env.example .env                   # then put your HF_TOKEN in it
```

`net.bootstrap()` (called at the top of every script) loads `.env` and points Python's
SSL at the OS trust store, so it works behind a corporate TLS-inspection proxy without
disabling certificate verification.

---

## Network / corporate proxy (IMPORTANT)

This repo downloads the model, the pretrained SAE, and the corpus from the
**Hugging Face Hub** (`huggingface.co`). On the machine this was scaffolded on, two
network issues showed up — you need both resolved before Phase 0 can run:

1. **TLS interception (solved in code).** The corporate proxy (Netskope) presents its
   own root CA, which Python's bundled `certifi` doesn't trust →
   `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`.
   Fixed by `truststore` (in `requirements.txt`), wired in via `sae_interp.net`, which
   makes Python use the macOS keychain where the corporate CA is already trusted. No
   action needed beyond installing requirements.

2. **Hugging Face is category-blocked (needs an exception — cannot fix in code).**
   The proxy returns **HTTP 403** with a "Generative AI – Block" page for
   `huggingface.co` (root, API, and file downloads), and `cdn-lfs.huggingface.co`
   doesn't resolve. This is a Simply Business InfoSec policy, not a code problem.
   **To unblock, either:**
   - request an exception for `huggingface.co` (and its CDN, e.g. `*.hf.co`,
     `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`) via the **Fresh Service
     Catalogue** (reviewed by InfoSec + DNA), **or**
   - run the HF-dependent phases off the corporate network — e.g. on a personal
     machine, or on **Google Colab / Kaggle** (which is also the recommended place to
     train in Phase 3 anyway). The notebooks in `notebooks/` are Colab-ready.

   > Note: this is a *personal portfolio* project. Running it on a corporate device/
   > network may itself fall under the company AI-use policy — worth checking before
   > requesting the exception.

Once Hugging Face is reachable, `python scripts/00_smoke_test.py` should pass and the
rest follows. Downloads are cached under `~/.cache/huggingface`, so it's a one-time hit.

---

## The plan

| Phase | Script | What it does | Compute |
|------:|--------|--------------|---------|
| 0 | `00_smoke_test.py` | Load GPT-2 + 1 pretrained SAE, print top-activating tokens | Mac |
| 1 | `01_explore_pretrained_sae.py` | Harvest activations, collect max-activating examples, pick interpretable features | Mac |
| 2 | `02_causal_intervention.py` | Clamp/ablate features, measure logit + generation effects | Mac |
| 3 | `03_train_sae.py` | Train our own L1 ReLU SAE on residual-stream activations | **GPU (Colab/Kaggle)** |
| 4 | `04_eval_frontier.py` | Sweep λ, plot L0 vs variance-explained / CE-recovered | GPU/Mac |
| 5 | writeup | README/blog with frontier plot, annotated features, causal results | — |

```bash
python scripts/00_smoke_test.py                         # the gate
python scripts/01_explore_pretrained_sae.py --auto 20   # explore, writes results/features.md
python scripts/02_causal_intervention.py --features 12 100
python scripts/03_train_sae.py --steps 2000 --sanity    # small local run; full run on GPU
python scripts/04_eval_frontier.py --steps 3000 --ce    # headline frontier plot
streamlit run app/feature_browser.py                    # optional UI
```

---

## The SAE (what we implement)

For an activation `x ∈ R^d` at the hook point (`d = d_model = 768`):

```
f      = ReLU( (x - b_dec) @ W_enc.T + b_enc )      # sparse codes, f ∈ R^{d_sae}
x_hat  = f @ W_dec.T + b_dec                         # reconstruction
loss   = ||x - x_hat||²  +  λ · ||f||₁
```

Key choices (and why) are documented in `src/sae_interp/sae.py`:
centering by `b_dec` before the encoder, **unit-norm decoder columns** (enforced via
gradient projection + renormalisation), and λ as the knob that traces the
sparsity↔reconstruction frontier. Dead features are periodically resampled during
training (`src/sae_interp/train.py`).

We **use the pretrained SAE only for Phases 1–2** (via SAELens); the SAE we *train* in
Phase 3 is our own implementation.

---

## Verified library APIs (as installed)

The original brief was written from early-2026 knowledge; the actually-installed
versions are newer, so these were checked against the live packages:

| Package | Installed | Notes vs. brief |
|---------|-----------|-----------------|
| `transformer-lens` | 3.3.0 | `HookedTransformer.from_pretrained("gpt2")`, `run_with_cache`, `run_with_hooks`, `generate` all as expected. Hook `blocks.8.hook_resid_pre` confirmed via `get_act_name`. |
| `sae-lens` | 6.44.2 | `SAE.from_pretrained(release, sae_id, device, dtype)` now returns **just the SAE** (older versions returned a `(sae, cfg, sparsity)` tuple). Scripts handle both. |
| pretrained SAE | `gpt2-small-res-jb` / `blocks.8.hook_resid_pre` | **Confirmed present** in the SAELens directory (repo `jbloom/GPT2-Small-SAEs-Reformatted`, 13 SAEs). |
| `torch` | 2.12.0 | MPS works; note MPS has no float64 — metrics move to CPU before `.double()`. |
| Python | 3.11.15 | via Homebrew. |

Exact pins: `requirements.lock.txt`.

---

## Reproducibility

Seeds are set in `train.set_seed`; configs are the single source of truth. Activation
caches and SAE checkpoints are written under `activation_cache/` and
`results/checkpoints/` and are **gitignored** (regenerate from code). Figures in
`results/figures/` are small and committed so the writeup renders on GitHub.

See **RUNBOOK.md** for the day-to-day operating guide.
