# ToDo — Harden the SAE Phase 1–2 Writeup

**Scope:** Strengthen the published Phase 1–2 writeup ("SAE interpretability on GPT-2: first results").
**Live post:** `akash-personal-website.vercel.app/engineering/sae-gpt2-phase1-phase2`

Goal: make the lead result (#9577) defensible to an interpretability hiring audience, replace the
too-clean label with an honest hypothesis that fits the evidence better, and restructure so the hook
comes first. The honesty and "what didn't work" content are the differentiators — keep them.

Priority order: **Group A** (experiments) → **Group B** (reframe) → **Group C** (precision) → **Group D** (writeup polish).
A unblocks B.

---

## Group A — Validation experiments for #9577

The current cross-domain claim rests on a single prompt at a single alpha. That's the same
one-example trap correctly caught for #13481. Fix it with more evidence before rewriting the claim.

- [x] **A1 — Multi-prompt steering.** Clamp #9577 across ~6 diverse prompts, record top-10 boosted
  tokens for each. Suggested set:
  - neutral: `"The weather today is"`
  - everyday: `"I went to the shop to buy"`
  - crime: `"The police arrested the"`
  - medical: `"After surgery, patients with severe"`
  - travel/migration: `"At the border, the officials saw"`
  - civic: `"The new policy will affect"`

  *Acceptance:* a table of prompt → top-10 boosted tokens. Does the same concept appear across
  domains, or only in some?

- [x] **A2 — Alpha sweep.** For 2–3 prompts above, sweep alpha (e.g. 5, 10, 20, 40). Note where the
  concept first appears, where it stabilises, and where it degrades into noise. Does the effect hold
  at low alpha (in-distribution), or only at alpha=40 (possibly off-distribution)?

- [x] **A3 — Max-activating breadth check.** Expand max-activating examples for #9577 from top-10
  to top 30–50 and categorise contexts. Count: how many are medical vs prison/detention vs
  refugee/migration vs other. If firing examples are mostly medical while steering is broad, that
  mismatch must be stated, not papered over.

- [x] **A4 — Boosted-token audit.** Categorise the clamp-boosted token list into:
  - (a) confined/institutional — patients, inmates, detainees, prisoners, juveniles, offenders, addicts
  - (b) socially othered but not confined — homosexuals, Iraqis, migrants
  - (c) neither

  *Acceptance:* categorised list. This is the direct evidence for the reframed label in B1.

---

## Group B — Reframe #9577 as an honest hypothesis

- [x] **B1 — Replace the label.** Current label ("people under institutional control / confinement
  with limited agency") is a clean human story over messier data — several boosted tokens
  (homosexuals, Iraqis, migrants) aren't confined. Reframe to fit the A4 audit, e.g. "marginalised /
  othered / low-agency groups of people." Frame as a bias / representation finding — safety-relevant
  and more interesting to the target audience.

- [x] **B2 — State the fire-vs-steer tension openly.** Feature fires on medical/hospital text but
  steers toward a much broader group. Name this as the central puzzle, present the broader-label
  hypothesis as the best current explanation. Don't paper over it.

- [x] **B3 — Downgrade conclusion to hypothesis.** Present #9577 as a hypothesis with visible
  tensions, supported by A1–A4 evidence. State plainly what it does and doesn't establish. Surfacing
  the uncertainty reads as sophisticated, not weak.

---

## Group C — Causal-metric precision and methodology transparency

- [x] **C1 — Define the boost metric.** State exactly what "+7.31 juveniles" means — e.g. "next-token
  logit difference, clamped minus baseline, at the clamp position." Apply consistently wherever
  boosted tokens appear.

- [x] **C2 — Document feature selection.** Say which 20 of the 24,576 features were explored and how
  they were chosen (random / by density / hand-picked). One or two sentences. Removes cherry-picking
  doubt.

- [x] **C3 — Strengthen or caveat ablation.** The framework claims "clamp toward AND ablate away," but
  the only ablation evidence is girl→woman. Either add 1–2 more ablation diffs, or explicitly
  acknowledge the ablation signal was weaker than the clamp signal. Both are fine.

---

## Group D — Writeup polish

- [x] **D1 — Add a TL;DR lede.** ~3 lines at the top of the post, leading with the #9577 result
  including its caveat. Skimming reader gets the hook immediately; chronological journey stays below.

- [x] **D2 — Hedge the density heuristic.** "1–4% healthy, >10% too vague" is stated like a law.
  Mark it as a rule of thumb. Half a sentence.

- [x] **D3 — Note the better logit-lens tool.** One line: applying the final LayerNorm or a tuned
  lens would refine the logit-lens projection. Shows awareness that a better tool exists.

- [x] **D4 — DO NOT TRIM the honesty.** Explicitly keep:
  - the #13481 mislabel correction
  - the "What didn't work" section (#7137)
  - the interpretable-≠-causally-upstream insight
  - the bug-found section

  These are the strongest differentiators. Do not clean them up to look more polished.

---

## Definition of done

- A1–A4 produce concrete artefacts (tables / counts), saved under `results/`.
- The #9577 section reads as a hypothesis-with-evidence using the reframed label, with the
  fire-vs-steer tension stated.
- Boost metric defined; feature selection documented; ablation strengthened or caveated.
- TL;DR lede added; density and logit-lens hedges in; all "what didn't work" content retained.
- Post re-published.

---

## Optional / stretch

- [ ] Run a lighter A1-style multi-prompt check on one other feature (#16836) to show the
  methodology generalises beyond #9577.
