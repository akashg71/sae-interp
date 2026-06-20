# Offline Reference Notes
> Written for you to read while away. Come back to finish Phase 1 annotation.

---

## 1. File Structure

```
sae-interp/
│
├── configs/
│   └── sae_gpt2small.yaml      ← THE master config. All numbers (batch size,
│                                  steps, layer, SAE size) live here. Scripts
│                                  read this; you rarely touch script code itself.
│
├── scripts/                    ← Run these in order, one per phase.
│   ├── 00_smoke_test.py        ✅ DONE — proved stack works
│   ├── 01_explore_pretrained_sae.py  ✅ DONE — produced results/features.md
│   ├── 02_causal_intervention.py     ← NEXT (Phase 2)
│   ├── 03_train_sae.py               ← Phase 3 (needs GPU, use Colab)
│   └── 04_eval_frontier.py           ← Phase 4
│
├── src/sae_interp/             ← The actual Python library. Scripts import from here.
│   ├── __init__.py             ← Just re-exports: device, config loader
│   ├── config.py               ← Reads sae_gpt2small.yaml into a typed dataclass
│   ├── device.py               ← Picks CPU/MPS/CUDA automatically
│   ├── sae.py                  ← OUR OWN sparse autoencoder implementation (the point of phase 3)
│   ├── features.py             ← Bug was here today (W_dec shape fix). Feature analysis tools.
│   ├── activations.py          ← Runs model, collects residual stream activations
│   ├── interventions.py        ← Clamp/ablate features to causally steer the model
│   ├── metrics.py              ← L0, variance explained, CE-loss-recovered
│   ├── net.py                  ← Small utility: tokenise a corpus efficiently
│   └── train.py                ← Training loop for Phase 3
│
├── results/
│   └── features.md             ← YOUR TODO: annotate the Label column (see section 5)
│
├── notebooks/                  ← Colab-ready. Use for Phase 3 (GPU training).
├── app/feature_browser.py      ← Optional Streamlit UI (stretch goal)
├── requirements.txt            ← Python packages needed
├── pyproject.toml              ← Makes `import sae_interp` work as a package
├── RUNBOOK.md                  ← Short "what to type" guide
└── README.md                   ← Full explanation of every design choice
```

**The mental model:** configs → scripts → src library → results.
You edit config or results; you rarely need to touch src unless debugging.

---

## 2. Python Syntax You'll See Here — Quick Refresher

### Type hints (everywhere in this codebase)
```python
# Old Python style (what you might remember):
def greet(name):
    return "hi " + name

# Modern Python 3.10+ style used here:
def greet(name: str) -> str:
    return "hi " + name

# Common patterns you'll see:
def fn(x: int, y: float = 1.0) -> list[str]:  ...
def fn(model, sae, indices: list[int]) -> torch.Tensor:  ...
```
Type hints after `:` and after `->` are just documentation hints — Python doesn't enforce them at runtime. They help your IDE catch errors.

### Dataclasses
```python
from dataclasses import dataclass

@dataclass                    # @dataclass is a "decorator" — a function that wraps the class
class SAEOutput:
    x_hat: torch.Tensor       # these are fields, NOT class variables
    loss: torch.Tensor

out = SAEOutput(x_hat=tensor1, loss=tensor2)
print(out.loss)               # access like normal attributes
```
Think of `@dataclass` as auto-generating `__init__`, `__repr__` etc. for you. You'll see `SAEOutput` returned by the SAE's forward pass.

### f-strings
```python
feature_idx = 7137
density = 3.24
print(f"feature #{feature_idx} | density {density:.2f}%")
# → "feature #7137 | density 3.24%"
# The :.2f means "2 decimal places, float format"
```

### List comprehensions
```python
# Instead of:
result = []
for i in range(10):
    result.append(i * 2)

# Python idiom:
result = [i * 2 for i in range(10)]

# With a condition:
result = [i for i in range(10) if i % 2 == 0]  # only even numbers
```
You'll see these constantly in the scripts.

### `@torch.no_grad()` decorator
```python
@torch.no_grad()
def collect_activations(model, tokens):
    # Inside here, PyTorch doesn't track gradients.
    # This saves memory and speeds up inference (we're not training here).
    ...
```

### Walrus operator (`:=`) — Python 3.8+
```python
# Assign AND test in one line:
if (n := len(data)) > 10:
    print(f"too many items: {n}")
```
You might see this occasionally; it's just assignment inside an expression.

### `__pycache__` folders
Those folders under `src/sae_interp/` with `.pyc` files — Python auto-creates them. They're compiled bytecode cache. Safe to ignore entirely; git ignores them too.

### Virtual environment (`.venv`)
```bash
source .venv/bin/activate    # "enter" the env — always do this first
# your terminal prompt changes to show (.venv)
# now `python` and all imports use the packages installed here

deactivate                   # "leave" the env
```
The `.venv` folder holds all installed packages locally — keeps this project isolated from your system Python.

### Running scripts
```bash
# Always activate the venv first, then:
python scripts/02_causal_intervention.py --features 7137 13481

# flags after -- are command-line arguments, parsed inside the script
# --features takes a list of feature indices
```

---

## 3. What Is Hugging Face?

Hugging Face (huggingface.co) is effectively **GitHub for AI models**. It's a hosting platform where researchers upload:
- **Pre-trained model weights** (the billions of numbers that make a neural net work)
- **Datasets** (text corpora used to train/test)
- **Code** (model architectures, training scripts)

Everything is free to download. You reference a model by `"owner/name"`, e.g. `"openai-community/gpt2"`.

In this project, HF is used to download two things automatically:
1. GPT-2 weights (via `transformer_lens`) — cached to `~/.cache/huggingface/`
2. The dataset `NeelNanda/pile-10k` (via `datasets`) — small text corpus
3. The pretrained SAE weights (via `sae_lens`) — also cached

After first run, everything is cached locally. **You won't need internet again** until you change models.

---

## 4. The Models Pulled From Hugging Face — And Why

### GPT-2-small (`"gpt2"`)
- **What:** OpenAI's 2019 language model. 124M parameters, 12 transformer layers, hidden size (d_model) = 768.
- **Why this one:** Small enough to run on a laptop (fits in ~500MB RAM), well-studied, lots of existing interpretability work to compare against. The SAE community standardised on it.
- **What it does:** Given text, predicts the next token. Internally, each layer transforms a "residual stream" — a 768-dimensional vector per token that carries all information through the network.

### Pretrained SAE (`"gpt2-small-res-jb"`, hook `blocks.8.hook_resid_pre`)
- **What:** A Sparse Autoencoder trained on GPT-2's layer 8 residual stream by Joseph Bloom (a researcher at Anthropic). 24,576 features (32× expansion of 768).
- **Why:** We use his *already-trained* SAE in Phases 1 and 2 to skip straight to the interesting part — interpreting features. Phase 3 is where we train our *own* SAE from scratch.
- **What "hook_resid_pre" means:** TransformerLens lets you intercept activations at named points inside the model. `blocks.8.hook_resid_pre` = the residual stream vector entering block 8 (the 9th of 12 layers). Layer 8 is a sweet spot — shallow enough that features are relatively clean, deep enough that they're semantically meaningful.

### Dataset (`"NeelNanda/pile-10k"`)
- **What:** 10,000 text documents sampled from "The Pile" (a large mixed-domain text dataset). ~tens of MB.
- **Why:** We need real text to run through GPT-2 and collect activations. This specific corpus is a community standard for SAE interpretability work.

### The data flow
```
Text corpus  →  GPT-2  →  layer 8 residual stream  →  SAE  →  sparse features
(pile-10k)      (HF)       768-dim vector/token          (HF)    24576 numbers/token
                                                                  most are 0
```

---

## 5. What Is `features.md`? What Are You Labelling?

### What a "feature" is

The SAE takes each 768-dim residual stream vector and maps it to a **24,576-dim sparse vector** where almost all values are 0. Each of the 24,576 dimensions is a "feature". 

The key discovery from SAE research: **each of these features tends to activate on a coherent, human-interpretable concept** — not a random mixture. E.g., "Python code", "medical terminology", "pivot/contrast words like therefore/so/thus".

### What Phase 1 actually did

1. Took 2,000 text documents, ran them through GPT-2
2. At layer 8, fed the residual stream into the pretrained SAE
3. For each of the 20 most-active features, found the tokens where that feature fires hardest
4. Showed you the surrounding context (the `【highlighted】` token is where the feature peaked)

### What `features.md` contains

The table has 20 rows — one per feature. Each row has:

| Column | Meaning |
|--------|---------|
| **Feature** | The feature index (0–24575). Just an ID number. |
| **Density** | % of all tokens where this feature is non-zero. ~1% is typical and healthy. >10% usually means "fires on everything" (not useful). |
| **Logit-lens hint** | Top tokens the feature's decoder direction pushes the output toward. A shortcut hint — not always reliable but useful. |
| **Top example** | The highest-activation token in context. `【token】` = the peak. Read the surrounding words for meaning. |
| **Label** | **YOU fill this in.** Your interpretation of what concept this feature detects. |

### How to label

Read the top example and 3–4 more from Phase 1 output (printed to terminal or re-run script). Ask: *what do these tokens have in common?*

Examples from today's run:
```
#7137  → fires on: <p1></p1>, .port=, .version=, ACCOUNT_KEY, </footer>
         → Label: "code tokens / property-file syntax / dot notation"

#13481 → fires on: "it may be necessary", "must always", "it is necessary to"
         → Label: "modal necessity / obligation language"

#16836 → fires on: "so I was", "therefore", "thus we", "instead they"
         → Label: "contrast/consequence pivot words"

#488   → fires on: import statements, ### markdown headers, int declarations
         → Label: "code structure / indentation boundaries"
```

A label can be vague — "fires around technical/scientific text" is fine. The point is human pattern recognition. You're doing the thing a computer can't: noticing the *concept*.

### What happens with your labels (Phase 2)

You'll pick 3–5 features with clear labels and run:
```bash
python scripts/02_causal_intervention.py --features 7137 13481 16836
```

This **clamps** those features to an artificial high value during generation to see if the model starts producing text matching your label. E.g., clamping feature #13481 should make GPT-2 produce more "must/necessary/required" language. That's the **causal validation** — proving the feature isn't just correlated with a concept but actually *causes* it.

---

## When You're Back: Exact Next Steps

```bash
cd /Users/akashgupta/Projects/sae-interp
source $HOME/.local/bin/env   # make uv/python available
source .venv/bin/activate     # activate project env

# 1. Open results/features.md and fill in the Label column for at least 3-5 features.
#    Best candidates: #7137, #13481, #16836, #488, #9577

# 2. Run Phase 2 causal intervention on your chosen features:
TRANSFORMERLENS_ALLOW_MPS=1 python scripts/02_causal_intervention.py --features 7137 13481 16836

# 3. (Optional) Try a specific prompt + feature clamp to see steering in action:
TRANSFORMERLENS_ALLOW_MPS=1 python scripts/02_causal_intervention.py \
  --features 13481 --alpha 15 --prompt "The doctor told the patient that"
```

The `TRANSFORMERLENS_ALLOW_MPS=1` prefix just suppresses a warning about the Apple Silicon GPU backend — the code still works.
