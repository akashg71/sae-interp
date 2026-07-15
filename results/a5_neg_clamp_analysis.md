# A5 — Negative Clamping at α=-40: Feature #9577

**Run:** `python scripts/02_causal_intervention.py --features 9577 --alpha -40`
**Raw output:** `results/a5_negative_clamp_raw.txt`
**Question being tested:** What does the *opposite* of feature #9577 look like?
If positive clamping steers toward vulnerable/institutional people, negative clamping
suppresses that direction and should reveal what fills the space instead.

---

## Boosted tokens under α=-40 (what becomes MORE likely when feature is suppressed)

| Prompt | Top tokens |
|---|---|
| Weather (neutral) | eele, aligned, resistor, external, proxies, chrom, capacitor |
| Shop (everyday) | isoft (Microsoft), vertisement (advertisement), hower, control chars |
| Police arrested the | dayName, inventoryQuantity, xon (Exxon), Inspection, WARRANT |
| After surgery patients | dealership, primer, dentist, sunscreen, Insurance, warranty |
| Border officials saw | municip, mpire, Solar, orb, フォ (Japanese fragment) |
| New policy will affect | Offline, Network, Market, EntityItem, PsyNet, Transaction |

**Clear pattern:** Non-human, technical, commercial content. Code identifiers
(dayName, inventoryQuantity, EntityItem, PsyNet), commercial language (dealership,
Insurance, warranty, Market, Transaction), electronics (resistor, capacitor),
corporate fragments (isoft = Microsoft). No people at all.

---

## Generation diffs under α=-40 (what the model generates when feature suppressed)

### Most revealing pairs:

**"At the border, the officials saw"**
- baseline: `"...a man with a gun and a knife."`
- clamped:  `"...a bright light of the day, and they were ready to go."`
- **Human subject disappears entirely.** The gun-carrying man is replaced by
  an environmental/abstract scene with no person present.

**"I went to the shop to buy"**
- baseline: `"...a new pair of shoes."`
- clamped:  `"...a new stereo amplifier...power supply...power supply..."` (repetition)
- Shift from a human purchase to a technical/electronics product. Power supply
  mentioned twice — off-distribution repetition starting to appear.

**"After surgery, patients with severe"**
- baseline: `"...pain and swelling...walk...doctor..."`
- clamped:  `"...pain and swelling of the lower back...acetaminophen-containing
  acetaminophen-containing..."` (drug name repeats)
- Stays in medical register but loses the human patient — shifts toward pharmaceutical
  product language. Patient becomes less visible.

**"The police arrested the"**
- baseline: `"...the man who allegedly shot and killed a man in the parking lot..."`
- clamped:  `"...the following day, and the police have been looking for a possible
  criminal investigation..."` — procedural, institutional language; the human subject
  (man) is no longer named. Abstract crime procedure with no person.

**Less revealing pairs:**
- Weather: mild difference, both coherent — feature doesn't fire naturally here
- Border: coherent shift but abstract
- Policy: shift toward tech/internet domain

---

## What this reveals about the feature

The negative direction of feature #9577 points toward:
- **Non-human, technical, transactional content** (code identifiers, commercial terms)
- **Absence of human subjects** — generations lose the person, not just the vulnerability
- **Abstract or procedural language** where no specific person appears

This is consistent with the hypothesis: the feature encodes a dimension from
"non-human / technical / transactional" → "human subjects of institutional attention."
The midpoint (natural state, no intervention) is neutral context. Positive clamping
pulls toward one pole; negative clamping pulls toward the other.

---

## The ReLU caveat — why the boosted tokens contain garbage

SAE features use ReLU activations: during training, feature activations are always ≥ 0.
The SAE has never been trained on negative activations. So α=-40 pushes the residual
stream into territory the SAE was never trained to represent.

This is visible in the boosted-token table:
- Control characters (`\x04`, `\x1c`)
- Raw byte fragments (`🔃`, `覚醒`, `フォ`)
- Partial corporate tokens (`isoft`, `Downloadha`, `raltar`)

These are off-distribution artifacts, not meaningful predictions. The logit-level
signal is contaminated. However, the generations (which go through GPT-2's full
forward pass) are more robust — GPT-2 itself wasn't constrained to ReLU-positive
territory, so it partially recovers. This is why the generations are semi-coherent
even though the token-level boosted list is noisy.

**Practical implication:** The generation diffs are trustworthy evidence; the
boosted-token table under negative clamping should be read with caution.

---

## Should this go in the post?

Yes, as a short extension — for two reasons:
1. The generation diffs showing the human subject disappearing (border example,
   surgery example) are genuine new evidence for the feature interpretation.
2. Demonstrating the ReLU caveat with real output is more instructive than just
   explaining it abstractly.

Suggested placement: brief subsection after the generation diffs, titled something
like "The negative direction."
