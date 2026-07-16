# ToDo — Harden the SAE Phase 3–4 Writeup

**For:** the agent working on the sae-interp project.
**Scope:** the **Phase 3–4 writeup** ("Training an SAE from scratch — dynamics, dead features, and a flat frontier"). Live post: `akash-personal-website.vercel.app/engineering/sae-gpt2-phase3-phase4`.

**Goal:** the training-dynamics narrative is strong (the staircase and the `dead_feature_window` cascade diagnosis are the highlights, keep them). Before this gets amplified on LinkedIn to recruiters/hiring managers, fix a few framing and method issues that a sharp reader could puncture. The theme of every fix below is the same: don't let a metric read as an achievement when it's actually a symptom, and mark hypotheses as hypotheses.

**Priority:** Group A (correctness/framing — do before posting anywhere) → Group B (surface the numbers) → Group C (small polish).

**Note for the agent:** several items need facts from the code that aren't in the writeup (dataset handling, held-out vs train VE, natural activation scale). Check the code and answer them in the post rather than guessing.

---

## Group A — Framing & correctness (blocking for the LinkedIn post)

- [x] **A1 — Reframe "VE=1.000 / perfect reconstruction" as a symptom, not a win.**
  At L0≈1161 active features in a 768-dim space, near-perfect reconstruction is trivial and *expected* — it signals the SAE is in the low-sparsity regime, not that it's good. State plainly:
  - This is a **mechanics / training-dynamics demonstration, not a feature-discovery result.**
  - Features at L0≈1161 are almost certainly **not interpretable** (unlike Bloom's L0≈40 SAE) — so don't imply this SAE is at feature-quality.
  - VE=1.000 is the **consequence of insufficient sparsity**, not an achievement.
  Current phrasing invites the opposite read; make the caveat explicit wherever VE=1.000 appears (body + status table).

- [x] **A2 — Resolve the dataset-size / generalization question (this may be the real story).**
  256k unique vectors × 40k steps × 4096/batch ≈ ~164M samples ≈ **~640 passes over the same data**. At that many epochs the SAE likely **memorizes** the set, so VE=1.000 may be a **train-set** number with no generalization meaning — and this, more than step count, could explain both the perfect reconstruction and the flat frontier. Answer in the post, from the code:
  - Are activations **streamed fresh** (e.g. SAELens `ActivationsStore`) or is a **fixed 256k set reused** every epoch?
  - Is VE measured on **held-out** activations or on the training set?
  - If it's a fixed set with train-set VE: say so, add held-out VE if feasible, and revise the Phase 4 explanation to name dataset size as a likely confound. Ideally re-run with a larger / streamed set and report held-out VE.

- [x] **A3 — Soften the flat-frontier explanation to hypothesis + confound.**
  The "L1 too small to bite at 3k steps" story fits the low-λ end but **not** λ=5×10⁻³, where the penalty is ~15% of the loss (≈29 vs recon ≈182) and L0 *still* didn't move. Rewrite as: "flat frontier — working hypothesis is the early-training regime, with dataset size (A2) as a likely confound; not a settled explanation." Acknowledge the high-λ point the current explanation doesn't cover.

- [x] **A4 — Downgrade "the gap to L0=40 is just compute."**
  That's a hypothesis, not something the data shows — the flat sweep itself hints sparsity pressure (λ and/or data) may be limiting, not only steps. Change "just compute" to something like "primarily capacity + training, though my own flat sweep suggests sparsity pressure may also be a factor."

- [x] **A5 — Fix the status-table consistency bug (contradicts the Phase 1–2 post).**
  The Phase 3–4 status table still says **"cross-domain institutional confinement cluster"** and **"#9577 proven causal."** Both contradict the reframed Phase 1–2 post ("marginalised/othered," "hypothesis, not confirmed") and "proven causal" reintroduces the exact overclaim that was removed. Update the table to match:
  - Phase 1 → "marginalised/othered institutional-object cluster (hypothesis)"
  - Phase 2 → "#9577 shows a consistent causal effect (clamp strong, ablation weaker); one mislabel corrected via evidence"

---

## Group B — Surface the evidence you already have
- [x] **B1 — Show the numbers behind claims.** The post asserts training dynamics but the reader should see the metric that backs each: keep the staircase table (good), and make sure the dead-fraction climb (2.6% → 9.9%) and the resample-event sizes are clearly tied to the L0 plateaus. If any of these live only in logs, pull them into the post or link them in the repo.

---

## Group C — Small polish
- [x] **C1 — Hedge the 4× "too small for specialisation" diagnosis.** Doubling to 8× still left ~19% active, so the non-sparsity is more about the **training regime** than dictionary size. Soften the 4× claim accordingly (or note both factors).
- [x] **C2 — Keep the highlights and the honesty.** Do **not** trim: the staircase, the `dead_feature_window` cascade diagnosis, the "what I'd do next" section. These are the load-bearing differentiators.

---

## Definition of done
- VE=1.000 reframed as symptom (A1); dataset/held-out question answered from the code, ideally with a larger/streamed re-run (A2).
- Flat-frontier explanation and "just compute" both softened to hypotheses with the confound named (A3, A4).
- Status table reconciled with the Phase 1–2 post (A5).
- Supporting numbers surfaced (B1); 4× diagnosis hedged (C1); highlights retained (C2).
- Post re-published.

*After this is done, send the updated Phase 3–4 writeup back for a final pass, then we move to the LinkedIn post (lead with the reframed #9577 finding; use the from-scratch training + dynamics as proof of depth; link the blog).*
