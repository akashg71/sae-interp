# A1 — Multi-prompt Steering Analysis: Feature #9577

**Run:** `python scripts/02_causal_intervention.py --features 9577 --alpha 20`
**Raw output:** `results/a1_9577_steering.txt`

---

## Boosted-token table (α=20)

| Prompt | Max boost | Top tokens | Signal quality |
|---|---|---|---|
| "The weather today is" | +2.03 | Haitian, Armenian, Sioux, inmates, Indones., Guam, Jill, Becky | Weak, noisy — names + ethnic minorities + 1 confinement token |
| "I went to the shop to buy" | +2.51 | patients, grandchildren, daughters, students, youngsters, elderly, children | Clear care-receiving cluster |
| "The police arrested the" | +1.92 | unconscious, starving, frail, terrified, numbers (760, 680…) | **Weakest signal; NO confinement tokens** |
| "After surgery, patients with severe" | +3.31 | juveniles, inmates, prisoners, detainees, addicts, youths, Iraqis, offenders | **Strongest; classic confinement cluster** |
| "At the border, the officials saw" | +2.71 | 166, 27, 183, 34, 276, 28, 141, 145, mentally | **Near-zero conceptual signal — 9 of 10 are numbers** |
| "The new policy will affect" | +2.95 | traumatized, mentally, abused, Somali, schizophrenic, incarcerated, retarded, impoverished, homosexuals | Broader marginalisation cluster |

---

## Token category breakdown (A4 done simultaneously)

Collecting all distinct interpretable tokens across prompts:

**Confined/institutional** (7): juveniles, inmates, prisoners, detainees, addicts, offenders, incarcerated

**Socially othered / not confined** (7): Iraqis, homosexuals, Somali, Haitian, Armenian, Sioux, Indonesian

**Vulnerable but not othered or confined** (18): patients, children, elderly, grandchildren, granddaughter,
daughters, youngsters, students, frail, starving, unconscious, terrified, traumatized, abused,
mentally (ill), schizophrenic, impoverished, retarded

**Noise** (names, numbers, fragments): Haku, Jill, Becky, Danielle, Lydia, 760, 680, 780, 920, 166, 27…

---

## Key findings

### 1. The confinement contexts gave the WEAKEST signals
"The police arrested the" was the weakest prompt (+1.92 max, 0 confinement tokens).
"At the border, the officials saw" produced 9 numbers out of 10 tokens — functionally no signal.
These are the two most intuitively "confinement" prompts in the set.

**Implication:** The feature is not primarily about confinement as a physical or legal state.

### 2. The medical prompt is the feature's "home"
"After surgery, patients with severe" gives +3.31 — the highest in the set — and the cleanest,
most classifiable cluster. The original max-activating examples were all medical (hospital text).
A1 confirms: the medical register is where this feature is strongest.

### 3. The feature generalises, but unevenly
In non-medical contexts where the feature does fire, it shows different facets of the same concept:
- Shopping context → care-receiving people (children, elderly, patients)
- Civic/policy context → socially marginalised people (mentally ill, impoverished, incarcerated)

### 4. Fire-vs-steer tension
The feature FIRES on medical text but the steering effect generalises to broader "vulnerable/marginalised
people" language in civic contexts. The medical context is not uniquely about medical care — it is the
context in which GPT-2's representation of "people-as-objects-of-institutional-concern" is strongest.

---

## Revised label (feeds into B1)

**Original label:** "Institutionally confined or vulnerable people — patients, prisoners, detainees, addicts, juveniles under care/custody"

**Revised label (A1 evidence):** "Vulnerable or socially marginalised people — concentrated in medical/patient registers, extending to policy language about disadvantaged groups. The confinement cluster (inmates, prisoners, detainees) appears only in contexts already primed with medical/patient language."

**Honest frame:** This is likely a bias / representation feature. GPT-2 has learned that certain groups of people (patients, prisoners, the mentally ill, the impoverished, ethnic minorities) share a common representation in text — they are described as objects of societal concern, care, or control rather than as agents. The feature captures that shared representation across domains.

---

## What this means for next steps

- **B1:** Replace label with revised version above; frame as a bias/representation finding.
- **B2:** State fire-vs-steer tension: fires on medical text, steers toward broader vulnerability/marginalisation cluster.
- **A3:** Expand max-activating examples to top 50 to check whether firing examples are purely medical or mixed.
- **A2:** Alpha sweep recommended on 2 prompts: "After surgery" (home turf) and "The new policy" (generalisation case).
