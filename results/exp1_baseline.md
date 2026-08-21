# Experiment 1 -- Basic Deception Classification (S11)

**What this experiment asks:** is there enough information in single-trial EEG to distinguish deceptive from truthful trials at all, pooled across all available dyads? This is a baseline and sanity check, not the project's main innovation -- the generalization claim (does it transfer to an unseen dyad) is Experiment 2's leave-one-dyad-out design.

## Design decisions

- **Rows: deceiver rows primary** (`role == 'deceiver'`), because S11 asks about the person doing the deceiving, and pooling both roles into one training set would put a trial's deceiver row and observer row (same label) into different folds, a leakage channel unrelated to the science. Observer rows are reported alongside as a secondary, clearly-labelled run (S16's construct).
- **Feature set:** the unreliable tier is excluded from the modeling matrix. For `single_brain` alone this is unreliable (180 delta cols, all stimulus-windows; 30 theta cols, Feedback-pre window only -- see onednn_notes.md/models.py note; NOT delta-only, diverges from the plan's naive expectation). The headline set is reliable+marginal (1770 columns); reliable-only (1560 columns) is reported as a robustness check, decided in advance.
- **Behavioral columns excluded** from the EEG feature sets (`outcome` is post-hoc w.r.t. the label; the rest are behavioral, not neural) and instead fit as a separate labelled reference (behavioral-only logistic regression).

## Cross-validation scheme

`StratifiedKFold(k=5, shuffle=True, random_state=0)`, grouping: none, deliberately -- see caveat

k=5 (not 10): with ~10.6k rows and ~1.7-1.8k features, 5 folds leave ~8.5k training / ~2.1k test rows per fold, enough for a stable AUROC estimate, while halving the fit count for four model families plus a CNN.

**Why no grouping here but leave-one-dyad-out in Experiment 2:** exp1 is the pooled baseline asking whether the signal exists at all; exp2 asks whether it transfers to an unseen dyad. Allowing the same dyad in both train and test here means the model can partly recognize participants, so **pooled across dyads; optimistically biased relative to exp2's leave-one-dyad-out (S12). exp1 is an upper bound and sanity check, not the generalization claim.**

## Model results (primary: deceiver rows, reliable+marginal)

| model | AUROC | Bal. Acc. | F1 | Precision | Recall |
|---|---|---|---|---|---|
| logistic_regression (PRIMARY) | 0.534 [0.526, 0.541] | 0.524 [0.521, 0.528] | 0.482 [0.468, 0.497] | 0.503 [0.499, 0.508] | 0.464 [0.436, 0.491] |
| logistic_regression_l1 | 0.531 [0.525, 0.537] | 0.522 [0.513, 0.532] | 0.484 [0.464, 0.503] | 0.500 [0.490, 0.510] | 0.468 [0.440, 0.496] |
| svm | 0.534 [0.523, 0.545] | 0.525 [0.510, 0.541] | 0.494 [0.477, 0.510] | 0.503 [0.486, 0.520] | 0.485 [0.468, 0.502] |
| random_forest | 0.547 [0.533, 0.561] | 0.528 [0.522, 0.534] | 0.386 [0.369, 0.403] | 0.526 [0.516, 0.535] | 0.305 [0.286, 0.325] |
| gradient_boosting | 0.531 [0.519, 0.544] | 0.521 [0.504, 0.537] | 0.464 [0.444, 0.485] | 0.500 [0.481, 0.520] | 0.433 [0.410, 0.457] |
| majority_class (reference) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | - | - | - |
| behavioral_only_lr (reference) | 0.527 [0.518, 0.537] | 0.507 [0.501, 0.513] | - | - | - |

### Per-fold AUROC (all 5 folds, primary LR)

fold0=0.5326, fold1=0.5398, fold2=0.5239, fold3=0.5365, fold4=0.5361

## Robustness check: deceiver rows, reliable-only feature set

| model | AUROC | Bal. Acc. |
|---|---|---|
| logistic_regression | 0.529 [0.518, 0.541] | 0.519 [0.513, 0.526] |
| logistic_regression_l1 | 0.527 [0.517, 0.536] | 0.519 [0.513, 0.525] |

## Secondary: observer rows, reliable+marginal feature set

| model | AUROC | Bal. Acc. |
|---|---|---|
| logistic_regression | 0.537 [0.529, 0.545] | 0.525 [0.520, 0.530] |
| logistic_regression_l1 | 0.537 [0.533, 0.542] | 0.529 [0.521, 0.536] |
| svm | 0.530 [0.522, 0.538] | 0.519 [0.510, 0.528] |
| random_forest | 0.548 [0.534, 0.562] | 0.526 [0.512, 0.540] |
| gradient_boosting | 0.543 [0.531, 0.556] | 0.530 [0.523, 0.537] |

## Permutation null (S21)

Primary model (LR), primary feature set, 200 permutations. Observed AUROC = 0.5338; null mean = 0.4998 (sd=0.0076, 5th/50th/95th percentile = 0.4863/0.5001/0.5112). **p = 0.0050**.

## CNN diagnostic (feature-adequacy sanity check only)

CNN mean AUROC = 0.5318 on exact-trial folds (overlap fraction 0.9562). Delta vs primary LR = -0.0020. **Verdict: features adequate; no S10 revisit indicated.**

## Environment

- GPU: **not present** despite S5's assumption (`torch=2.8.0+cpu`, cuda_available=False, no `nvidia-smi`). CNN trained on CPU.
- Gradient boosting implementation actually used: **xgboost.XGBClassifier**.

## Frozen-file confirmation

`results/gate.json`, `results/trial_count_gate.md`, `results/frozen_hypotheses.md` were read-only inputs and were not written to or contradicted by this experiment (exp1 is `N/A (not gated)` in the gate, consistent with running it). Frozen-file mtimes were checked and predate this task; see `results/exp1_baseline.json`'s `validations.10_frozen_files_untouched` block.
