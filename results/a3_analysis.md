# A3 — Max-Activating Breadth Check: Feature #9577 (top-50)

**Run:** `python scripts/01_explore_pretrained_sae.py --features 9577 --top-k 50`
**Raw output:** `results/a3_maxact_50.txt`

---

## Category counts (all 50 examples)

| Category | Count | % | Representative examples |
|---|---|---|---|
| Medical/clinical (human) | 34 | 68% | biliary patients, ESRD patients, cancer-patients, DBS patients, schizophrenic patients, dialysis patients, surgery, pre-eclampsia, ITP patients, tibial prosthesis |
| Biology/animal research | 10 | 20% | kidneys of mice (UUO), eyes of 8 rabbits, tumors in rabbits, soleus muscle rats, mesenteric arteries of rat, female frogs (2×) |
| Law enforcement / coercive | 2 | 4% | police officer holding AR-15; AR-15 to head of unarmed subdued man |
| Social/development | 2 | 4% | food-insecure children entering kindergarten; education for poor communities India |
| Persecution/atrocity | 1 | 2% | "camps used to isolate and murder Jews, Serbs, Roma" |
| Sports/other | 1 | 2% | NY Giants NFL interest in D.J. Fluker |
| **Prison / detention / migration** | **0** | **0%** | — |

---

## Key findings

### 1. Zero prison/detention/migration examples
The tokens the feature steers toward in A1/A2 (inmates, prisoners, detainees, Iraqis as migrants)
are entirely absent from the contexts it fires on. 0 of 50 firing examples are prison, detention
centre, or border/migration contexts. This is the clearest evidence for the fire-vs-steer tension.

### 2. The Apollo Hospital document accounts for 5 of the top 50 hits
Examples ranked #1, #16, #22, #25, #39 are all the same sentence at different token positions:
"Apollo Hospital in Delhi have saved the life of a 14-year-old Iraqi student."
This is the highest-activating example (act=9.85) — likely because it combines the medical register
with an explicitly Iraqi, young, foreign patient. Both signals reinforcing each other.
Effective distinct documents in the top 50: ~46.

### 3. The fired token is usually a preposition ("of" / "in"), not a vulnerable-group noun
Across the top-50 examples, the 【highlighted token】 is:
- "of" in ~17 cases: "records【of】patients", "fluid【of】patients", "life【of】14-year-old", "kidneys【of】mice"
- "in" in ~15 cases: "brachytherapy【in】comparison", "carbamazepine【in】children", "FDG【in】schizophrenic patients"
- Content words in remaining: "murder", "holding", "saved", "followed", "life", "status"

The feature fires on the preposition in the construction "[clinical procedure/measurement] of/in [subject]"
— anticipating the vulnerable-group noun rather than detecting it. This suggests the feature encodes
a SYNTACTIC EXPECTATION: "an institutional subject follows" — not the subject noun itself.

### 4. Animal research examples share the same register as human clinical examples
"Kidneys of mice subjected to UUO", "16 eyes of 8 rabbits", "muscles of rabbits" — these use
identical depersonalised, institutional language to human clinical trials. The feature appears to
capture the linguistic register of "living subjects of institutional research/procedures" regardless
of whether those subjects are human.

---

## Revised understanding of the feature (feeds B1/B2)

**What the feature fires on:** The preposition "of" or "in" in academic/clinical language describing
a procedure or measurement done TO a living subject. Most commonly human patients in medical research
text; also lab animals.

**What the feature steers toward (from A1/A2):** A broader cluster of institutionalised, marginalised,
or vulnerable people — including groups not represented in the firing examples at all (prisoners,
detainees, the mentally ill, the impoverished).

**Fire-vs-steer tension (for B2):**
The feature fires in the medical/clinical register but steers toward the semantic concept of
"person under institutional control or social marginalisation." This is consistent with the feature
encoding the EXPECTATION of a vulnerable subject following a clinical preposition — and that
expectation generalising, under steering, to the broader class of vulnerable/marginalised people
the model has learned to associate with institutional language.

---

## What this means for the label (B1)

Original label: "Institutionally confined or vulnerable people"
A1 finding: Confinement contexts (police, border) gave weakest steering signals
A3 finding: Zero prison/detention/migration firing examples; fires on medical prepositions

**Best current label:** "Living subjects of institutional care or research — fires on medical/clinical
register language; steers toward a broader class of vulnerable or marginalised people under
institutional description. Likely encodes the linguistic expectation of a vulnerable subject
following a clinical preposition (of/in) rather than detecting the subjects themselves."

**For the public writeup (B1):** Frame as a bias/representation finding.
GPT-2 has learned that certain groups of people are described in institutional language as objects
of care, measurement, or control — patients, lab animals, prisoners, the mentally ill — and this
feature captures that shared representation. The unexpected finding is that the model generalises
this from the medical register to a much broader class of marginalised people under steering.
