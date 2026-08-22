# Datathon Project Workflow: Can Your Brain Learn a Liar?

## Project Title

**Can Your Brain Learn a Liar?**
**Testing Whether Neural Signatures of Deception Become Personalized Through Repeated Social Interaction**

Alternative title:

**Between Two Minds: Learning Relationship-Specific Neural Signatures of Deception**

---

## 1. Core Project Question

**As two people repeatedly interact, does the brain learn the deception patterns of that specific opponent?**

Instead of building a standard EEG lie detector, the project investigates whether deception-related neural patterns are:

1. universal across people,
2. specific to an individual, or
3. specific to the relationship between two interacting people.

Deception may not live only inside the deceiver's brain. Neural responses associated with deception may also emerge in the observer, and in the interaction between both brains.

---

## 2. Dataset

**Source:** [An EEG Dataset of Neural Signatures in a Competitive Two-Player Game Encouraging Deceptive Behavior](https://doi.org/10.6084/m9.figshare.24760827) (Chen, Wallraven, Fazli — figshare, CC BY 4.0).

Contents, verified against the dataset listing:

| File | Size |
|---|---:|
| `Raw.zip` — raw EEG | 5 GB |
| `Preprocessed.zip` — preprocessed EEG | 1.35 GB |
| `OneDCNN.zip` — data pre-formatted for 1D-CNN classification | 1.36 GB |
| `behavioral log and trigger timestamp.zip` | 1.46 MB |
| `readme.txt` | 2.12 kB |
| `Participant information and risk taking tendency.xlsx` | 9.04 kB |
| **Total** | **7.71 GB** |

Design facts:

- 12 participant pairs / dyads, 24 total participants
- Two-player deception game with role switching
- Simultaneous EEG recordings, 30 electrodes per player
- Behavioral logs and EEG trigger timestamps
- A separate participant-info spreadsheet includes a **risk-taking tendency** score per participant — not used anywhere in this workflow by default, but worth keeping in mind as an optional covariate for the personalization analyses (e.g., does personalization gain correlate with risk tendency?) if the trial-count gate in §7 leaves room for it.

Because both participants are recorded simultaneously, the dataset supports individual EEG analysis *and* dyadic/inter-brain analysis — this is what makes the relationship-specific hypothesis testable at all.

**Archives to pull:** `Preprocessed.zip` and `behavioral log and trigger timestamp.zip` are the primary working data, plus `Participant information and risk taking tendency.xlsx`. `OneDCNN.zip` is pulled only for the Experiment 1 sanity check in §11 — it is a separate data path and does not feed any other experiment. Pull `Raw.zip` only if a specific analysis requires re-deriving preprocessing steps.

**Compute:** a GPU is available, so the Experiment 1 CNN check carries no practical cost concern. Nothing else in the pipeline needs one.

---

## 3. Main Scientific Hypotheses

These are frozen before modeling begins (see §8). They are not revised after looking at results.

**H1 — Universal Deception Signature.** Neural patterns associated with deception generalize across participants. If true, a model trained on most participants should predict deception in completely unseen participants.

**H2 — Person-Specific Deception Signature.** Different people have different deception-related neural patterns. If true, models trained on prior data from the same person should outperform universal population models.

**H3 — Relationship-Specific Deception Signature** *(the project's main hypothesis)*. Repeated interaction causes neural responses to become specific to a particular pair. If true, models using prior interactions from the same dyad should outperform models trained only on population-level data.

---

## 4. Main Project Story

**Can EEG detect deception?**
→ **Are those patterns universal across people?**
→ **Do personalized models work better?**
→ **Does knowing the specific relationship improve prediction even further?**
→ **Does the observer's brain become more informative as interaction continues?**
→ **Do neural relationships between the two brains change over time?**

The goal is not to report a single classification accuracy. The project answers:

**Does repeated social interaction make deception-related neural patterns increasingly relationship-specific?**

---

## 5. Decisions Locked Before Execution

These were settled deliberately during planning. They are constraints on implementation, not open questions to revisit mid-build.

| Decision | Choice | Consequence |
|---|---|---|
| Time / team | Not a limiting constraint | Scope is bounded by what the data supports (§7), not by a deadline. §51 is a risk ordering, not a triage list. |
| Submission format | GitHub repo **plus** a hosted live web app | §28–§38 are in scope and are the primary judge-facing artifact. Static hosting (Vercel / Netlify / GitHub Pages) is available. |
| Statistical framing | Trial-count gate + frozen hypotheses, and a **paired per-dyad test** as the headline | Rewrites §14, §20, §23, §24. The unit of analysis is the dyad, n = 12. |
| Inference at judging time | **Fully precomputed.** No live model execution, no client-side inference | The web app only reads and displays. Every configuration a judge can reach must be enumerated in advance and baked into the results file. |
| Pipeline ↔ web interface | A single frozen, versioned `results/results.v1.json` | The frontend builds against a hand-written fixture with fake numbers in parallel with real modeling. Swapping in real results is a file replacement. |
| Frontend stack | React + TypeScript + Tailwind, shadcn for primitives | Per this repo's standing conventions. |
| Visual direction | **Editorial long-form science feature** (§29) | Not a dashboard layout. Figures stay editorial rather than instrument-panel dense. |
| Deep learning | 1D-CNN as an **Experiment 1 sanity check only** | Does not enter Experiments 2–8 and does not appear in the web app. |

Rubric details for the datathon were not available at planning time. Nothing in this workflow depends on rubric weighting.

---

# ANALYSIS WORKFLOW

## 6. Phase 1: Reconstruct the Interaction Timeline

Preserve the chronological structure of each dyad. Do not shuffle trials — the project depends on how interactions change with experience.

For every trial, build a table with:

- Pair ID, Participant ID, Partner ID, Role
- Round number, Truth/deception condition, Behavioral outcome
- Trigger timestamp, EEG window
- Previous interaction history

Conceptually, each pair is a chronological sequence: Round 1 → Round 2 → … → Final Round.

This table is the foundation for everything downstream, and it is also the input to the gate in §7. Build it first and validate it before any modeling.

---

## 7. Phase 1b: Trial-Count Gate — Go / No-Go

**This step exists because the plan's headline results are differences between model scores, and differences of noisy estimates are noisier than either estimate alone.** With 12 dyads and a limited number of rounds per participant, several experiments may not have enough data to produce a stable number. That has to be discovered before modeling, not after.

Before training anything, count and report:

- Trials per participant
- Trials per dyad
- Trials per role (deceiver / observer) per participant
- Trials per truth/deception condition, per dyad — including class balance
- Trials in each early / middle / late chronological split
- The size of the smallest test fold each experiment would produce

Write this to a table in the repo and to the results file. Then apply a **written go/no-go**, with the threshold chosen and recorded *before* the counts are looked at:

- Any experiment whose smallest test fold clears the threshold is reported as a **result**.
- Any experiment whose smallest test fold falls below it is demoted to **exploratory** — reported with its confidence interval and explicitly labeled underpowered, never presented as a finding.

Experiments 3, 4, and 5 (§13, §14, §15) are the ones most likely to be affected, because they operate on within-dyad chronological splits. Experiment 8's sliding-window analysis (§18) is next most at risk, because it multiplies the number of tests.

Demotion is not failure. A project that says "this design could not answer this question with this sample" is more credible than one that reports a difference whose confidence interval crosses zero and calls it an effect.

---

## 8. Phase 1c: Freeze Hypotheses and Primary Metric

Before any model is fit, commit to the repo, in writing:

- The three hypotheses as stated in §3
- The **primary metric**: the paired per-dyad comparison defined in §20
- The go/no-go threshold from §7
- Which experiments are confirmatory and which are exploratory
- The plan for reporting a null result (§47, §48)

Anything decided after seeing results is labeled post-hoc when reported. This costs nothing and is the difference between a result and a story fitted to noise.

---

## 9. Phase 2: EEG Preprocessing

Use the provided preprocessed EEG where appropriate, while documenting the existing preprocessing steps.

Construct EEG windows around relevant events (pre-event, onset, post-event), e.g.:

- −2000 ms to 0 ms
- −1000 ms to +1000 ms
- shorter sliding windows for temporal analysis

The exact window depends on trigger definitions and task structure.

---

## 10. Phase 3: Feature Engineering

Start with interpretable EEG features, computed per electrode, frequency band, and temporal window. Interpretability is not a nice-to-have here — §34 and the scalp maps in §25 depend on the model exposing coefficients or SHAP values over named features.

**Frequency-domain:** delta, theta, alpha, beta power; gamma power if signal quality supports it.

**Time-domain:** mean amplitude, variance, peak amplitudes, ERP-related features, signal statistics.

**Behavioral (if available):** reaction time, previous truth/deception outcome, recent deception frequency, success/failure of previous deception, cumulative interaction history, round number. *(Optionally: risk-taking tendency from the participant-info spreadsheet.)*

**Dyadic EEG (inter-brain):** correlation, cross-correlation, coherence, phase-locking value, wavelet coherence, mutual information, lagged relationships, network-based coupling features. Interpret these as statistical inter-brain relationships — not proof of brain-to-brain communication.

---

# MODELING WORKFLOW

## 11. Experiment 1: Basic Deception Classification

**Research question:** Is there enough EEG information to distinguish deceptive and truthful trials?

Target: Truth = 0, Deception = 1.

Use interpretable models — logistic regression, SVM, random forest, gradient boosting/XGBoost. These are the models that carry through the rest of the project.

**1D-CNN sanity check, scoped to this experiment only.** Train a 1D-CNN on `OneDCNN.zip` against the same folds, as a *feature-adequacy check*: if the CNN substantially beats logistic regression on identical folds, the handcrafted feature set in §10 is leaving signal on the table and §10 should be revisited. That is the entire purpose of running it.

The CNN does **not** carry into Experiments 2–8, does not feed the interpretability work in §25, and does not appear anywhere in the web app. It is a diagnostic, reported as one line in the results and one sentence in the write-up. Running it on the small within-dyad folds of Experiments 3–5 would produce the least stable numbers in the project, which is precisely why it is confined here, where the pooled data is largest.

This experiment is a baseline and sanity check, not the project's main innovation.

---

## 12. Experiment 2: Universal Model

**Research question:** Can a deception model generalize to completely unseen participant pairs?

Use **Leave-One-Dyad-Out Cross-Validation**: train on pairs 2–12, test on pair 1; train on pairs 1, 3–12, test on pair 2; continue until every pair has been the held-out test pair once.

This tests whether the model learns a generalizable deception signature rather than recognizing participants.

Metrics: AUROC, balanced accuracy, F1, precision, recall (AUROC and balanced accuracy are especially useful if classes are imbalanced).

Because each fold yields one held-out dyad, this experiment naturally produces **12 per-dyad scores** — which is exactly the structure the paired test in §20 consumes. Keep the per-fold scores; do not discard them after averaging.

---

## 13. Experiment 3: Personalized Model

**Research question:** Does knowing the individual improve prediction?

Compare a **population model** (train on other participants → predict target participant) against a **personalized model** (train on earlier trials from the target participant → predict later trials from the same participant), always preserving chronology.

Example: train on rounds 1–30, test on rounds 31–40 — never mix past and future trials randomly.

Both models are evaluated **on the same held-out trials** for each participant, so the two scores are paired rather than independent. Record the score for each participant individually; the aggregate is computed by the procedure in §20, not by pooling trials across participants.

Subject to the §7 gate: if the later-rounds test set is too small per participant, this experiment is reported as exploratory.

---

## 14. Experiment 4: Dyad-Specific Model *(central experiment)*

**Research question:** Does previous experience with the exact same opponent add information beyond person-specific learning?

Compare three levels, all evaluated on the same held-out trials within each dyad:

- **Universal** — model knows neither the target participant nor the relationship.
- **Person-Specific** — model has prior information about the relevant participant.
- **Dyad-Specific** — model has prior information about interactions between the exact two participants.

For **each dyad independently**, compute:

- `Person Gain_d = Person-Specific_d − Universal_d`
- `Dyad Gain_d = Dyad-Specific_d − Person-Specific_d`

This yields 12 values of each — one per dyad. **These 12 paired differences are the result.** They are tested by the procedure in §20 and displayed by the per-dyad panel in §35.

Do **not** report this as a single difference between two pooled AUROCs. A pooled difference discards the pairing, treats trials as the unit of analysis when the hypothesis is about dyads, and produces a number whose uncertainty is much harder to characterize honestly.

---

## 15. Experiment 5: Does Prediction Improve With Interaction History?

**Research question:** Does deception become more distinguishable after participants have interacted for longer?

Split each dyad chronologically into early / middle / late interaction (or use round number continuously), and compare prediction performance across time.

Stronger method: train with increasing amounts of prior interaction history, always testing on future trials — e.g., train rounds 1–5, 1–10, 1–20, 1–30, each time testing on later rounds. This produces a relationship-learning curve, **one curve per dyad**, and the curves are shown individually as well as aggregated.

**Important control:** more training data alone improves models. Compare the *same amount* of same-dyad history vs. other-dyad history to isolate whether improvement comes specifically from familiarity with the relationship.

This experiment is the most demanding of the trial-count gate in §7 — it subdivides an already-small per-dyad sample several times over. Expect it to be reported as exploratory unless the counts are unexpectedly generous, and design the figure so it reads honestly either way.

---

## 16. Experiment 6: Observer-Only Prediction

One of the most interesting experiments. Predict deception using only the **observer's** EEG (input: observer EEG; target: partner's truth/deception).

**Research question:** Does the receiver's brain contain information about whether their partner is deceiving them?

Compare early / middle / late interaction. If observer-only performance increases with interaction history, the receiving participant's neural response may become increasingly informative about the partner's behavior. Report per dyad, and test the early-to-late change with the paired procedure in §20.

Avoid claiming subconscious lie detection unless behavioral and statistical evidence directly supports it.

---

## 17. Experiment 7: One Brain vs Two Brains

Compare prediction using: (1) deceiver EEG only, (2) observer EEG only, (3) both participants' EEG, (4) inter-brain features, (5) EEG plus behavioral history. This localizes where the predictive information actually lives.

Evaluate all five input sets on identical folds so the comparisons are paired. This experiment supplies the numbers behind the judge-facing control in §33.

---

## 18. Experiment 8: Who Gives Away the Lie First?

**Research question:** When does deception-related information first become detectable in each participant's EEG?

Use sliding windows before the decision, e.g.: −1500 to −1250 ms, −1250 to −1000 ms, −1000 to −750 ms, −750 to −500 ms, −500 to −250 ms, −250 to 0 ms.

Train separate models for deceiver EEG and observer EEG. Find the earliest window where prediction reliably exceeds a permutation-based null distribution — call this **Deception Information Onset**. Compare `T_deceiver` vs `T_observer`.

This experiment runs many tests across windows, so correct for multiple comparisons and say which correction was used. Subject to the §7 gate like everything else.

Avoid interpreting a temporal lag as proof that information transferred directly from one brain to another.

---

# STATISTICAL VALIDATION

## 19. Prevent Data Leakage

Do not use ordinary random train/test splitting across all trials — it can let the same participant or pair appear in both training and testing, causing the model to recognize participants instead of deception.

Use:

- Leave-One-Dyad-Out CV for population generalization
- Chronological train/test splits for within-dyad analyses
- Proper grouped cross-validation for tuning

---

## 20. The Paired Per-Dyad Test *(primary inferential procedure)*

**The dyad is the unit of analysis. n = 12.**

Every headline comparison in this project — universal vs. person-specific, person-specific vs. dyad-specific, early vs. late, deceiver vs. observer — is computed **within each dyad first**, producing 12 paired differences. Those 12 values are then tested directly:

1. Compute the metric under both conditions for dyad *d*, on the same held-out trials.
2. Take the difference `Δ_d`.
3. Repeat for all 12 dyads.
4. Test the resulting 12 differences with a **sign test** and a **paired permutation test** (randomly flipping the sign of each `Δ_d` across many iterations to build the null).
5. Report the median `Δ`, the full distribution of the 12 values, the number of dyads with positive `Δ`, and the p-value.

Why this and not a difference of two pooled AUROCs: the hypotheses in §3 are claims about dyads, so dyads are what should be counted. Pairing removes the between-dyad variance that would otherwise swamp the effect. It produces a legitimate p-value from a small sample without pretending trials are independent observations. And the result is directly visualizable — 12 dots, a zero line, and a judge can see the evidence rather than take a number on faith (§35).

Report the pooled aggregate metric too, as a descriptive summary. It is not the test.

---

## 21. Permutation Testing

Classification above 50% does not automatically indicate a meaningful signal. Build a null distribution by shuffling labels while preserving relevant grouping constraints, then compare true performance against it. Report the observed score, the null distribution, and the permutation p-value.

This applies to absolute performance claims (Experiments 1, 2, 6, 8). The paired procedure in §20 applies to comparison claims. Both appear in the results file.

---

## 22. Confidence Intervals

Report uncertainty via bootstrap confidence intervals, variability across dyads, or cross-validation distributions. With only 12 dyads, avoid overinterpreting small differences — and where §7 demoted an experiment to exploratory, the confidence interval is the headline for that experiment, not an afterthought.

---

## 23. Per-Dyad Results Are the Primary Result

Not a supplementary check — the primary presentation. Every comparison ships with its 12 per-dyad values visible, showing whether the effect is consistent across pairs or driven by two or three of them. An effect present in 10 of 12 dyads and an effect present in 6 of 12 with two large outliers can produce the same mean; only the per-dyad view distinguishes them.

This is what §35 renders for judges, and it is a strong candidate for the money figure (§45).

---

# SIGNATURE PROJECT METRIC

## 24. Neural Familiarity Index

Defined per dyad, consistent with §20:

`NFI_d = Dyad-Specific Performance_d − Population Performance_d`

The project reports the **distribution of NFI across the 12 dyads** — median, spread, count above zero, and the paired-test p-value — not a single pooled scalar.

- **Median NFI > 0, consistent across dyads** — relationship-specific history improves prediction.
- **NFI centered near 0** — dyads behave like the population model.
- **Median NFI < 0** — relationship-specific information does not help.
- **NFI positive on average but wildly inconsistent across dyads** — a real and reportable finding in its own right, and one the pooled formulation would have hidden.

Use this as a project-defined descriptive metric, not an established neuroscience measure.

---

# INTERPRETABILITY

## 25. What Is the Model Using?

Investigate feature importance across Channel × Frequency × Time using logistic regression coefficients, permutation importance, SHAP (for tabular models), or ablation studies. Create scalp maps of electrode contribution, and compare early vs. late, truth vs. deception, deceiver vs. observer.

CNN saliency is out of scope — the CNN is confined to §11 and never becomes an object of interpretation.

All of this is precomputed and written to the results file; §34 renders it.

---

## 26. Inter-Brain Network Analysis

Represent both participants as a network: nodes are EEG electrodes/regions, edges are inter-brain coupling metrics. Compare truth vs. deception and early vs. late interaction, and show whether the structure or strength of inter-brain relationships differs between conditions.

**Inter-brain synchrony is a statistical relationship between two recorded signals. It is not evidence of communication between brains.** This caveat travels with every presentation of these results, including §36.

---

# RESULTS ARTIFACT

## 27. The Frozen Results Schema

Everything the web app shows is precomputed. The Python pipeline's product is a single versioned file, `results/results.v1.json`, and that file is the **only** interface between the analysis and the frontend.

Consequences, all deliberate:

- **The frontend can start immediately.** Hand-write `results/fixtures/results.v1.fixture.json` with the correct shape and invented numbers on day one. The web app is built and designed against it while modeling proceeds in parallel. Shipping real results is a file swap.
- **Every judge-reachable configuration must be enumerated in advance.** The cross-product the app exposes — dyad × round × role × model level × input source — is small and fully enumerable, but it must be decided before the schema is frozen. That is why the dashboard scope (§31–§36) was settled before this section.
- **Nothing can break at judging time.** No Python runs, no model loads, no backend to be asleep or rate-limited. The hosted site is static.
- **Schema changes are deliberate.** A change to the shape bumps the version and updates the fixture in the same change. It is not something to drift into.

Top-level shape:

```text
results.v1.json
├── meta                  schema version, generated-at, git-describe of pipeline, dataset checksum
├── gate                  §7 trial counts, the pre-registered threshold, per-experiment go/no-go verdict
├── frozen                §8 hypotheses, primary metric, confirmatory/exploratory designation
├── dyads[12]             per-dyad identity, participant ids, round count, class balance
├── experiments
│   ├── exp1              pooled baseline scores per model family, CNN sanity-check line
│   ├── exp2              per-dyad leave-one-dyad-out scores + aggregate
│   ├── exp3              per-participant population vs personalized, paired
│   ├── exp4              per-dyad universal / person / dyad triplets, PersonGain_d, DyadGain_d
│   ├── exp5              per-dyad learning curves + the same-vs-other-dyad control
│   ├── exp6              per-dyad observer-only, early/middle/late
│   ├── exp7              per-fold scores for all five input sets
│   └── exp8              per-window scores, onset estimates, multiple-comparison correction used
├── tests                 §20 paired tests and §21 permutation nulls, one entry per claim
├── interpretability      per-condition feature importances, scalp-map values
├── interbrain            §26 network nodes/edges per condition (truth/deception × early/late)
├── trials                the enumerated trials the Deception Lab (§31) can display
└── failures              §41 selected false positives and false negatives with attributions
```

`run_experiments.py` writes this file end to end. Notebooks read it for presentation; they are not the pipeline.

---

# INTERACTIVE SUBMISSION EXPERIENCE

## 28. Submission Philosophy

The submission is a GitHub repo plus a hosted live URL. Because the datathon is submission-based rather than presented live, the project must fully explain itself without the team present. Structure the judge experience in three layers:

1. **Immediate hook** — opening, headline framing
2. **Interactive discovery** — the Deception Lab, the model-level comparison, the per-dyad evidence
3. **Technical verification** — validation methodology, stats, failure cases

## 29. Visual Direction: Editorial Long-Form

**The site is a long-form science feature, not a dashboard.** A judge clicking a link without the team present is doing a reading task, not a tool-use task. A dashboard assumes someone who already knows what they are looking for; an editorial piece teaches them as they scroll and still contains every interactive component.

What this means concretely:

- Generous typographic hierarchy; the written argument is the spine, and the interactive components are **figures embedded inside it** rather than panels arranged on a grid.
- Figures are styled editorially — legible, spacious, captioned, one idea per figure. Not dense instrument-panel readouts. A figure should be readable at a glance and rewarding on inspection.
- The §37 walkthrough is already a linear narrative; the layout takes that seriously rather than fighting it.
- Scroll order *is* the argument order. A judge who reads top to bottom and never touches a control still gets the complete case.

Explicitly rejected, per this repo's standing design rules: boilerplate gradients, default component spacing and shadows, centered-hero-with-emoji layouts, generic rounded cards on a grid, stock "clean SaaS" styling, purple/blue gradient backgrounds. An obvious or default choice is a signal to reconsider it. The result must not look templated.

Stack is React + TypeScript + Tailwind with shadcn for primitives, restyled rather than shipped at defaults. The scalp maps (§34) and the two-brain network (§36) need custom SVG regardless of direction — no off-the-shelf chart library draws either.

## 30. Opening

Open with the title, then:

**We tested whether deception has a universal EEG signature — or whether repeated interaction causes neural responses to become specific to an opponent.**

Immediately show the headline facts: 12 pairs, 24 participants, simultaneous EEG, repeated deception, role switching. Do not open with technical model descriptions.

## 31. The Deception Lab

*(Merges what were previously three separate components — the trial browser, the timeline slider, and the early/late toggle. All three moved a judge along the same round axis; one interface does the work of three.)*

A single panel with:

- **Pair selector** — which dyad
- **Role selector** — deceiver / observer
- **Round slider** — moves through the relationship from first round to last, with **EARLY** and **LATE** preset buttons for the direct comparison

As the slider moves, the panel updates: round number, truth/deception condition, model prediction and confidence, selected EEG features, important electrodes, inter-brain features, and the network state. The presets make the early-vs-late comparison a single click while the slider preserves the continuous exploration.

All states are precomputed and enumerated in `trials` and `experiments` in the results file (§27).

## 32. Stranger vs Familiar *(centerpiece)*

Three selectable conditions — **STRANGER / PERSON-SPECIFIC / DYAD-SPECIFIC** — updating model performance for each.

This is the most direct interactive demonstration of the project's main question, and it is the component everything else supports. It should be the most refined thing on the page.

## 33. Who Gives Away the Lie?

Let the judge pick the data source — deceiver's brain / observer's brain / both brains / inter-brain features — and show the corresponding model performance. Backed by Experiment 7 (§17).

## 34. Explain the Prediction

A **"What did the model see?"** section showing top predictive features and electrode importance on a scalp map, precomputed per condition from §25. This is where the decision to keep the modeling interpretable pays off.

## 35. The Per-Dyad Panel *(primary result view)*

**Promoted from a browsable extra to a primary result**, because the paired per-dyad test in §20 is now the project's inferential procedure — which makes this panel a direct picture of the evidence rather than a supplementary gallery.

Show the 12 dyads as 12 paired differences against a zero line: `DyadGain_d`, `PersonGain_d`, `NFI_d`, observer-only early-to-late change. A judge looking at 12 dots, most above or below zero, is looking at exactly what the sign test consumes. Annotate with the median, the count above zero, and the p-value.

Each dyad also carries its compact fingerprint — frontal coupling, temporal lag, alpha synchrony, observer response, personalization gain — browsable across all 12.

This is the strongest money-figure candidate (§45).

## 36. Two-Brain Network Visualization

A two-brain network — deceiver on the left, observer on the right — with edges representing measured inter-brain relationships, and controls for Truth ↔ Deception and Early ↔ Late. This lets a judge compare four states: Early Truth, Early Deception, Late Truth, Late Deception.

**Kept in the hero flow as a deliberate choice.** The accompanying caveat is not optional and travels with it wherever it appears, in the same visual weight as the figure itself:

> These edges are statistical relationships between two simultaneously recorded EEG signals. They are not evidence of communication between brains. Inter-brain analyses in this project are descriptive.

The §7 gate applies here too — if the inter-brain comparisons are underpowered, the figure says so on its face rather than in a footnote.

## 37. The Judge's Walkthrough

Design the site so a judge naturally moves through this sequence in under two minutes:

| Time | Judge sees / does | Takeaway |
|---|---|---|
| 0:00–0:10 | Title + headline facts (12 pairs, 24 brains, repeated deception, simultaneous EEG) | Immediate curiosity |
| 0:10–0:30 | Framing line: this isn't lie detection, it's testing whether deception signatures personalize | Understands the real question |
| 0:30–1:00 | Toggles Stranger → Person-Specific → Dyad-Specific, watches prediction change (§32) | Sees the core hypothesis tested live |
| 1:00–1:20 | Sees the 12 per-dyad differences against zero (§35) | Sees the actual evidence, not just an average |
| 1:20–1:50 | Selects Observer EEG Only (§33), explores the two-brain network (§36) | Grasps the observer-side and inter-brain findings |
| 1:50+ | Scrolls to the trial-count gate, Leave-One-Dyad-Out CV, chronological validation, paired permutation testing, confidence intervals, failure cases | Technical trust established |

## 38. Demo Video / GIF (15–30 seconds)

Do not assume judges will explore the app. Embed a GIF or short video in the README following the same sequence as §37: select Pair 04 → move the round slider early → late → watch the network visualization change → switch Stranger to Dyad-Specific → watch performance change → show the 12 per-dyad differences → switch to Observer EEG → reveal the main result.

---

# STORY MODE VS TECHNICAL MODE

## 39. Default Judge View

Use plain language by default, e.g.: **"The observer's EEG became more informative later in the interaction."** Provide a **"See technical details"** expansion revealing AUROC, confidence intervals, the paired-test p-value and how many of the 12 dyads showed the effect, the cross-validation procedure, and feature definitions.

Where §7 demoted an experiment to exploratory, the plain-language version says so plainly — "we could not test this reliably with 12 dyads" is a sentence a judge should be able to read without opening the technical panel.

---

# MAKE METHODOLOGY PART OF THE STORY

## 40. Show Why Random Splitting Is Wrong

**Bad:** the same participant appears in both training and test data — risk: the model memorizes the participant.

**Good:** train on 11 dyads, test on the 12th completely unseen dyad. Explain it directly: **"Every pair takes one turn as the completely unseen test relationship."**

Extend the same treatment to the unit-of-analysis choice: show why comparing two pooled scores across all trials is misleading when the hypothesis is about dyads, and why the project counts 12 paired differences instead (§20). Methodology done visibly is a differentiator, not a digression.

---

# FAILURE ANALYSIS

## 41. Show When the Model Is Wrong

Include several false positives and false negatives, each with actual label, predicted label, confidence, and influential features. Make clear the model detects statistical patterns, not thoughts.

---

# ETHICS AND LIMITATIONS

## 42. What This Project Does Not Claim

- The model does not read thoughts.
- It is not a real-world lie detector.
- It does not determine whether arbitrary people are lying.
- EEG classification in this controlled experiment should not be interpreted as proof of deception outside the experimental setting.
- Inter-brain synchrony is not proof of direct brain-to-brain communication.
- The sample contains only 12 dyads, so results require cautious interpretation — and where the trial-count gate (§7) found an analysis underpowered, that is stated rather than papered over.

Frame the project as: **using deception as a controlled social interaction to study how neural responses change through repeated interaction.**

---

# README AND REPOSITORY

## 43. Repository Structure

The pipeline is Python modules driven by a single entry point that emits the results file. Notebooks present; they do not compute.

```text
README.md
requirements.txt
run_experiments.py          writes results/results.v1.json end to end

data/                       downloaded archives, gitignored

src/
  timeline.py               §6 trial table reconstruction
  gate.py                   §7 trial counts + go/no-go
  preprocessing.py          §9 windowing
  features.py               §10 EEG, behavioral, inter-brain features
  models.py                 §11 model families
  cnn_check.py              §11 Experiment 1 sanity check only
  experiments/              exp1 … exp8, one module each
  stats.py                  §20 paired tests, §21 permutation nulls, §22 CIs
  interpret.py              §25 importances, scalp-map values
  interbrain.py             §26 network construction
  emit.py                   assembles and validates results.v1.json

results/
  results.v1.json           the frozen artifact (§27)
  schema/results.v1.schema.json
  fixtures/results.v1.fixture.json    hand-written, fake numbers, for frontend dev

notebooks/                  presentation only, read results.v1.json
  01_data_reconstruction.ipynb
  02_gate_and_freeze.ipynb
  03_preprocessing.ipynb
  04_feature_engineering.ipynb
  05_baseline_models.ipynb
  06_personalization.ipynb
  07_dyad_specific_analysis.ipynb
  08_observer_analysis.ipynb
  09_interbrain_analysis.ipynb
  10_statistical_testing.ipynb
  11_visualizations.ipynb

app/                        React + TypeScript + Tailwind, static build
figures/
```

## 44. README Opening

Do not begin with installation instructions. Begin with:

> # Can Your Brain Learn a Liar?
>
> When two people repeatedly deceive each other, does the brain learn the patterns of its specific opponent?
>
> We analyzed simultaneous EEG from 12 interacting pairs to test whether deception signatures are universal, person-specific, or relationship-specific.

Then the live URL, a short demo GIF, the headline finding, and the main visualization. Installation instructions come afterward.

---

# THE "MONEY FIGURE"

## 45. Main Summary Visualization

Two candidates; pick whichever the actual results make more legible, and consider showing both.

**Candidate A — the per-dyad evidence (§35).** Twelve paired differences plotted against a zero line, annotated with the median, the count above zero, and the p-value. This *is* the test, drawn. It answers the main question and shows its own uncertainty in the same image, which is rare and persuasive.

**Candidate B — the input × model-level grid.** One figure summarizing all results, answering: **Does familiarity with the person or relationship improve deception prediction?**

| Input / Model | Universal | Person-Specific | Dyad-Specific |
|---|---:|---:|---:|
| Deceiver EEG | Actual result | Actual result | Actual result |
| Observer EEG | Actual result | Actual result | Actual result |
| Both Brains | Actual result | Actual result | Actual result |
| Dyadic Features | Actual result | Actual result | Actual result |

Every cell carries its uncertainty, not just a point estimate. Cells belonging to experiments the §7 gate marked exploratory are visually distinguished from confirmatory ones.

---

# FINAL PROJECT QUESTIONS

## 46. Five Main Questions

1. Can EEG distinguish truthful and deceptive interactions?
2. Are deception-related EEG patterns universal or person-specific?
3. Does previous experience with a specific opponent improve prediction?
4. Does the observer's brain become increasingly informative about their partner's deception?
5. Do relationships between the two participants' neural signals change over repeated interaction?

---

# POSSIBLE FINAL CONCLUSIONS

## 47. If Relationship-Specific Learning Is Supported

**Deception-related neural patterns generalized poorly across strangers but became more predictive when models incorporated information specific to the individual and dyad. The improvement was consistent across dyads, with [N] of 12 pairs showing positive dyad-specific gain. Neural responses in observers also became more informative across repeated interaction, suggesting that deception-related neural dynamics may partly depend on relationship history rather than representing a fully universal neural signature.**

## 48. If Relationship-Specific Learning Is Not Supported

**Despite repeated interaction, dyad-specific history did not significantly improve prediction beyond participant-specific or population-level models. Across 12 dyads the paired differences were centered near zero, and the effect was not consistent in direction. This suggests that deception-related EEG patterns in this dataset are more strongly driven by individual or general neural characteristics than by relationship-specific adaptation.**

Both conclusions are written before results are seen (§8), and both are equally publishable outcomes of this design. A clean, well-powered null is a better submission than an overclaimed effect.

---

# FINAL PITCH

## 49. One-Sentence Pitch

**Can your brain learn a liar? We use simultaneous EEG from two interacting players to test whether neural signatures of deception are universal, person-specific, or learned through repeated interaction with a particular opponent.**

## 50. 30-Second Pitch

Traditional EEG deception projects ask whether a person's brain can reveal when they are lying. We ask something different: what happens after two people repeatedly interact? Using simultaneous EEG from 12 pairs, we compare universal, personalized, and relationship-specific machine-learning models to test whether deception becomes easier to distinguish as a person gains experience with a particular opponent. We treat each pair as one observation and test the 12 paired differences directly, rather than reporting a single pooled accuracy. We also investigate whether the observer's brain and the neural relationship between both participants become increasingly informative over time.

---

# EXECUTION ORDER

## 51. Build Order

Time is not the binding constraint — the data is. This is a dependency and risk ordering, not a triage list.

**Revised 2026-08-22** to reflect actual execution order and status. The original ordering put the fixture and frontend scaffold at steps 3 and 5, in parallel with the experiment work — that did not happen; experiments 1/3/4 were built and run first, and the fixture/scaffold were never started. Reordered below to (a) record what actually happened, (b) front-load the fixture + frontend scaffold now, since they have no dependency on the remaining experiments and no external compute (Colab/SSH) to wait on, and (c) keep going on experiments that don't block on either.

1. ✅ **Reconstruct reliable trial chronology and roles** (§6). Done.
2. ✅ **Run the trial-count gate and freeze hypotheses** (§7, §8). Done — includes the exp4 Amendment 1 (n=11→10, dyad-grain).
3. ✅ **Build a leakage-free deception baseline** (§11), including the CNN feature-adequacy check. Done.
4. 🔶 **Compare universal vs. personalized prediction** (§12, §13). Experiment 3 (personalized) done. Experiment 2 (universal/LODO) running now on Colab, unattended — external, not blocking anything below.
5. ✅ **Test dyad-specific history** (§14) and the paired per-dyad tests (§20). Done — Experiment 4, primary claim not supported, see `results/exp4_dyadic.json`.
6. ⏭️ **Freeze `results.v1.json` and hand-write the fixture** (§27) — **start now.** Every experiment the schema references except exp5–exp8 already has a real results file; the rest of the schema's shape can be hand-written with invented numbers per §27's own instruction, exactly as originally planned. No dependency on exp2 finishing.
7. ⏭️ **Scaffold the web app against the fixture** (§29–§36) — **start now, right after step 6**, in parallel with step 8 below. This is the item most overdue relative to the original plan (it was meant to run alongside steps 4–5, not after).
8. **Analyze early vs. late interaction** (§15), **test observer-only EEG** (§16), **one brain vs. two brains** (§17), **information onset** (§18) — next experiment work, each independent of the frontend track.
9. **Add inter-brain features and the network analysis** (§10, §26) — highest methodological risk, most cautious claims; §10's dyadic/PLV features are already computed, §26's network analysis/modeling is not.
10. **Swap real results into the app, then the demo GIF and README** (§38, §44) — once exp2/5/6/7/8 land, swap the fixture for `results/results.v1.json` (a file swap per §27, not a rebuild).

The central project should remain strong even if the inter-brain hyperscanning analysis is inconclusive — and under §7 and §20, "inconclusive" is a reportable outcome rather than a gap.

---

# FINAL DESIGN PRINCIPLE

Do not build the submission around **"We trained an AI lie detector."**

Build it around **"We used deception as a controlled social setting to investigate whether one brain learns the patterns of another person through repeated interaction."**

The machine-learning models are the tools used to answer that scientific question — not the project itself.
