# Concepts Reference

Running log of every idea, term, and intuition from this project.
Updated as new things come up — if something is confusing, it gets added here.

---

## Norms and Distances

### What is a "norm"?
A norm is a way to measure the size or length of a vector. Given a vector
`x = [x₁, x₂, ..., xₙ]`, different norms measure "size" differently.

---

### L2 Norm (Euclidean length)
```
||x||₂ = sqrt(x₁² + x₂² + ... + xₙ²)
```
The straight-line distance from the origin to point x in n-dimensional space.
This is exactly the Pythagorean theorem extended to n dimensions.

**Squared L2** (what we use for reconstruction loss):
```
||x||₂² = x₁² + x₂² + ... + xₙ²   (no square root)
```
Squaring removes the sqrt (faster, smoother gradient) and amplifies large errors more.

---

### L1 Norm (Manhattan distance)
```
||x||₁ = |x₁| + |x₂| + ... + |xₙ|
```
Sum of absolute values. Named "Manhattan" because it's like walking on a city grid —
you can only go horizontal or vertical, not diagonal. Distance from (0,0) to (3,4)
is 3+4=7 in Manhattan, sqrt(3²+4²)=5 in Euclidean.

**Key difference vs L2:**
- L2 gradient: `2x` — shrinks as x→0, so values become *small but nonzero*
- L1 gradient: `sign(x)` = ±1 — *constant*, so even tiny values feel full pressure to zero
- Result: L1 pushes values to *exactly zero* (sparse). L2 pushes toward *small* (smooth).

**Why reconstruction loss uses L2, not L1:**
Squared L2 gives larger gradients for larger errors (proportional to error size), which
means the optimizer prioritizes fixing big mistakes. L1 would treat a reconstruction
error of 10 the same as 10 errors of 1 — no extra urgency for large misses.
Also: L2 is smooth (differentiable everywhere). L1 has a kink at 0 (non-differentiable),
which complicates optimization.

---

### L0 "Norm" (not really a norm)
```
||x||₀ = count of non-zero elements in x
```
Not a true mathematical norm — it's discontinuous (jumps, can't be differentiated).
Can't be directly minimized by gradient descent. L1 is used as a differentiable *proxy*:
minimizing L1 tends to drive L0 down too.

**In this project:** L0 = average number of SAE features active per token.
- L0 = 1524 → 1524 of 3072 features fire per token (bad, not sparse)
- L0 = 40 → 40 of 3072 features fire per token (good, interpretable)

---

## Regularization

Adding a penalty term to the loss to enforce a desired property on the model.
Without regularization, models overfit or use resources unconstrained.

```
Total Loss = Task Loss + λ × Penalty
                         ↑
                    regularization strength
```

| Type | Penalty | Effect |
|------|---------|--------|
| L1 (Lasso) | λ Σ\|wᵢ\| | Drives weights to exactly zero → sparse |
| L2 (Ridge/weight decay) | λ Σwᵢ² | Drives weights toward small but nonzero → smooth |
| Dropout | Randomly zero neurons | Forces redundancy, prevents memorisation |

**In this project:** L1 regularization on the *feature activations* f (not the weights).
```
L1_loss = λ × Σ|fᵢ|   i = 0..d_sae-1
```
Each feature that activates costs λ. The optimizer only keeps a feature on if its
contribution to reconstruction is worth the cost.

**λ too small → L0 stays high** (features are cheap, use all of them)
**λ too large → L0 near zero** (features are too expensive, SAE barely uses any, VE drops)
**λ sweet spot → L0 = 20–100** (sparse but still reconstructing well)

---

## SAE Loss Function (putting it all together)

```
x     = original 768-dim residual stream vector (input)
x_hat = SAE reconstruction of x (output)
f     = d_sae-dim sparse feature activations (latent)

recon_loss = Σᵢ (xᵢ - x_hatᵢ)²     ← squared L2, summed over 768 dims
l1_loss    = λ × Σⱼ |fⱼ|            ← L1 on feature activations

Total Loss = recon_loss + l1_loss
```

The two terms fight each other:
- `recon_loss` wants to use ALL features (more features = better reconstruction)
- `l1_loss` wants to use NO features (each feature costs λ)
- Equilibrium: use the *minimum* set of features that justify their cost

---

## Variance Explained (VE)

How much of the variation in the original data the model captures.

```
VE = 1 - Var(x - x_hat) / Var(x)
```

- VE = 1.000 → perfect reconstruction (SAE explains everything)
- VE = 0.000 → SAE explains nothing (same as predicting the mean)
- VE = -5.79 → SAE is *worse* than predicting the mean (random init at step 1)

**Why VE=1.000 with L0=1524 is still a bad result:**
Achieving VE=1.000 by using 1524 features is trivial — the SAE has enough capacity to
memorise everything. A *meaningful* result is VE=0.87 with L0=40 — good reconstruction
under the hard constraint of sparsity. The constraint is what makes individual features
have to mean something distinct.

---

## Transformer Architecture (GPT-2-small Specific Dimensions)

```
Vocab embedding:     50,257 tokens  →  768-dim vector per token
Position embedding:  1,024 positions →  768-dim vector per position
                     (added together)

× 12 Transformer blocks:
  ┌─ LayerNorm (768-dim)
  ├─ Multi-Head Attention
  │   ├── 12 heads
  │   ├── each head: 64-dim  (768 / 12 = 64 per head)
  │   ├── W_Q per head: (768 → 64)
  │   ├── W_K per head: (768 → 64)
  │   ├── W_V per head: (768 → 64)
  │   └── W_O (concat all heads → 768): (768 → 768)
  ├─ Residual add
  ├─ LayerNorm (768-dim)
  ├─ MLP (feed-forward)
  │   ├── W_in:  (768 → 3072)   [4× expansion]
  │   ├── GELU activation
  │   └── W_out: (3072 → 768)
  └─ Residual add

Final LayerNorm (768)
Unembed W_U: (768 → 50,257)   ← maps residual stream to next-token probabilities
```

Total parameters: ~124 million.

---

## The Residual Stream

The central data structure. Each token has a 768-dim vector that is *read from and
written to* by every layer. Attention heads and MLP layers don't replace it — they
add to it (residual connections). By the final layer, the vector contains all the
information accumulated across all 12 layers.

```
token → embed (768) → block 0 → block 1 → ... → block 11 → unembed → logits
                      ↑add        ↑add                ↑add
                   attn+MLP    attn+MLP             attn+MLP
```

The SAE hooks into this stream at `blocks.8.hook_resid_pre` = the 768-dim vector
*entering* block 8 (before block 8's attention or MLP have run).

---

## Multi-Head Attention

Each attention head independently asks: "for this token, which other tokens are
relevant, and what information should I pull from them?"

```
Q = x @ W_Q   (768 → 64): "what am I looking for?"
K = x @ W_K   (768 → 64): "what do I offer to others?"
V = x @ W_V   (768 → 64): "what information do I carry?"

attention_score = softmax( Q · Kᵀ / √64 )   ← how much to attend to each position
head_output     = attention_score · V         ← weighted sum of values

All 12 heads concatenated → 768-dim → W_O → added to residual stream
```

The `/ √64` prevents the dot products from getting so large that softmax saturates
(all attention going to one token). This is the "scaled dot-product attention."

---

## Superposition (Why SAEs Are Needed)

GPT-2 has a 768-dim residual stream but must represent thousands of features
(syntax, semantics, factual knowledge, positional info, etc.).

**Solution:** Store multiple features in the same dimensions simultaneously,
overlapping them. This works because features are *sparse* — "is_French",
"is_programming_code", "refers_to_a_person" aren't all active at once for most tokens.
The interference between overlapping features is tolerable.

**Consequence:** Individual neurons (dimensions) are *polysemantic* — they respond to
many unrelated concepts. You can't interpret a single neuron directly.

**What SAEs do:** Find the underlying sparse features by mapping 768-dim → 3072-dim
(or more), where each dimension is forced to be active rarely. The hope is that the
larger space has enough room for each concept to get its own dimension.

---

## Sparse Autoencoder (SAE) Architecture

```
INPUT:  x ∈ ℝ⁷⁶⁸  (residual stream vector)

ENCODE:
  f = ReLU( (x - b_dec) @ W_enc + b_enc )
       ↑ subtract data center first (b_dec acts as learned mean)
       ↑ W_enc: (768, d_sae) — linear projection to feature space
       ↑ ReLU: sets negative values to zero (key sparsity mechanism)
  f ∈ ℝ^d_sae   (e.g. 3072 or 6144), mostly zeros

DECODE:
  x_hat = f @ W_dec + b_dec
          ↑ W_dec: (d_sae, 768) for SAELens, (768, d_sae) for our custom SAE
          ↑ each row/column is a "dictionary direction" in 768-dim space

OUTPUT: x_hat ∈ ℝ⁷⁶⁸  (reconstruction of original residual stream)
```

**Decoder weight layout (important — source of a bug we fixed):**
- Our custom SAE: `W_dec` shape = (768, d_sae) — each *column* is a feature direction
- SAELens pretrained SAE: `W_dec` shape = (d_sae, 768) — each *row* is a feature direction
- Bug: code was always using `W_dec[:, feature_idx]` (column), which crashed on SAELens
- Fix: detect layout from shape — if `shape[0] < shape[1]` use column; else use row

---

## Expansion Factor

```
d_sae = expansion_factor × d_model
```

How many times bigger the SAE's hidden layer is vs the residual stream.

| expansion_factor | d_sae | d_model | Ratio |
|-----------------|-------|---------|-------|
| 4 (tried, failed) | 3,072 | 768 | 4× |
| **8 (current)** | **6,144** | **768** | **8×** |
| 32 (pretrained) | 24,576 | 768 | 32× |

**Why it matters for sparsity:**
With 4× expansion, each of the 3072 features has to carry a lot of weight. The optimizer
finds it better to use many features simultaneously (L0=1500) than to be sparse.
With 8× (or 32×), there are enough features that each can specialise — a feature that
fires only on "medical patient context" doesn't need to also cover "legal proceedings"
because there are enough features for both to exist separately.

**The U-shaped L0 curve (training dynamics):**
```
step   1:  L0 = 1536  (all features random, most accidentally fire)
step 500:  L0 = 1187  ← lowest (L1 killing features that aren't useful yet)
step 20k:  L0 = 1524  ← rose back up (features became useful, optimizer re-uses them)
```
Why does L0 first drop then rise? Early in training, most features point in random
directions and genuinely don't help. L1 kills them efficiently. But as gradient
descent runs, features learn to point in *useful* directions. Now the optimizer
can see: "feature #847 costs λ=0.004 to activate, but reduces recon loss by 15 —
worth it." Features get invited back in as they become meaningful. With too small an
expansion factor, there aren't enough features to be selective, so all of them end up
"worth it."

---

## Dead Features and Resampling

A "dead" feature is one that never fires (activation = 0 on every token in a window).
```
dead_mask = (step - last_fired[feature]) > dead_feature_window
```
Default window: 2000 steps. If a feature hasn't fired in 2000 steps, it's dead.

**Why features die:** If a feature's encoder direction points somewhere unhelpful, it
never activates, never gets gradient signal, and stays dead permanently.

**Resampling (Anthropic's recipe):**
1. Find the tokens the SAE currently reconstructs *worst* (highest squared error)
2. Point the dead feature's encoder/decoder toward one of those poorly-reconstructed vectors
3. Reset Adam's momentum for that feature (stale momentum from being-dead would fight learning)
4. Give it a fresh window to prove itself

**What unstable resampling looks like (8× run, 20k steps):**
```
step 10000: resampled   18 features → normal
step 12000: resampled  104 features → dead_fraction jumps to 15.6%
step 14000: resampled  960 features → L0 crashes to 895 briefly, recovers to ~1400
step 16000: resampled  533 features → L0 crashes to 919 briefly, recovers to ~1375
step 18000: resampled  831 features → L0 crashes to 692 briefly, recovers to ~1210
```
Root cause: `dead_feature_window=2000` is too short for 6144 features. Features are given
only 2000 steps to prove themselves, many fail, and they die in waves. This creates
massive "shock" events that temporarily crash L0 but don't produce a stable sparse solution.

**The staircase pattern:** Each shock crashes L0 to a temporarily low value, which then
climbs back to a new, lower equilibrium. The equilibria over time: 1515 → 1450 → 1400 → 1375 → 1210.
The run was still descending at step 20000.

**Fix:** `dead_feature_window: 5000` — gives features 2.5× more time, smaller resampling
batches, less chaos per event.

---

## Feature Interpretability Methods

### 1. Max-Activating Examples
Run many real documents through GPT-2. Record feature activation at every token
position. Find the top-k positions where the feature fired hardest. Show context.

The `【highlighted】` token is where the feature peaked. Read surrounding words to
identify the concept. Most reliable method — grounded in real data.

### 2. Logit-Lens
Mathematical shortcut. Project the feature's decoder direction through GPT-2's
unembedding matrix `W_U`:
```
logit_contribution = W_dec[feature_idx] @ W_U
```
Top tokens in this vector are what this direction "votes for" at the output.
Fast but unreliable — ignores LayerNorm and the fact we're at layer 8, not the
final layer. Use as hypothesis, verify with examples.

**When they disagree:** Trust max-activating examples. We found feature #7137 had
misleading logit-lens tokens ("SourceFile") that didn't match the actual examples
(XML tags, config files). The examples were right.

### 3. Feature Density
What fraction of tokens activate this feature.
```
density = count(f_j > 0) / total_tokens
```
- 0%: dead feature (never fires)
- 0.5–5%: healthy, specific concept
- >10%: fires on almost everything, too generic to be useful

---

## Causal Interventions (Phase 2)

Finding that a feature *correlates* with a concept isn't enough. Causal validation
proves the feature is *upstream* of the concept, not just associated with it.

### Clamping (Steering)
Add `α × decoder_direction` to the residual stream at every token:
```
x_edited = x + α × W_dec[feature_idx]
```
This forces the feature artificially ON. If the generation then shifts toward the
feature's labelled concept, the feature is causally upstream.

`α` (alpha) is the steering strength. Start at 8–20. Higher = stronger effect but
can break coherence (model loses track of context). Sweet spot varies per feature.

### Ablation
Remove the feature's contribution from the residual stream:
```
f = sae.encode(x)
x_edited = x - f[feature_idx] × W_dec[feature_idx]
```
Removes only this feature's contribution, leaving everything else intact. If output
shifts away from the concept, the feature was actively contributing to it.

**Why ablation often shows no difference:**
Ablation only works if the feature was *already firing naturally* on your prompt.
If the prompt doesn't trigger the feature (f_j ≈ 0), there's nothing to ablate.
This is why "I am a doctor to save" showed no ablation effect (feature #9577 wasn't
triggered by that sentence) but "Apollo Hospital in Delhi have saved the life of"
did (the feature was our highest Phase 1 example for exactly that text).

### What we found: Feature #9577
Label: "institutionally confined or vulnerable people — patients, prisoners,
detainees, addicts, juveniles under care/custody"

Not just "medical text." The SAE independently discovered that hospitals, prisons,
detention centres, and rehabilitation facilities share a structural similarity in
how language describes the people inside them.

Causal result (prompt: "Apollo Hospital in Delhi have saved the life of"):
```
baseline:  "...a young girl who was shot dead by a man trying to rob her."
clamped:   "...a man who was injured. He was taken to the hospital."
ablated:   "...a young woman who was shot in the head by a man trying to rob her."
```
Three distinct outputs. Feature contributed the "young/vulnerable" framing — removing
it changed "girl" → "woman" (less specifically vulnerable).

---

## Phase Status

| Phase | Status | Key result |
|-------|--------|------------|
| 0 — Smoke test | ✅ Done | Stack confirmed working, GPT-2 + SAE loaded |
| 1 — Feature exploration | ✅ Done | 5 features labelled; #9577 most interesting |
| 2 — Causal validation | ✅ Done | #9577 proven causal; #13481 mislabelled (corrected) |
| 3 — Train custom SAE | 🔄 Improving | 8× (L0=1210, VE=1.0); resampling unstable (window too small); rerunning with dead_window=5000, 40k steps |
| 4 — Frontier evaluation | ⏳ Pending | L0 vs VE sweep across λ values |

---

*This file is updated as new concepts come up. Last updated: 8× run analysis — resampling staircase dynamics, dead_feature_window fix.*
