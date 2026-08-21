"""
src/cnn_check.py -- Section 11's 1D-CNN feature-adequacy sanity check ONLY.

Purpose and only purpose: if a shallow CNN trained on raw epoch signal substantially
beats logistic regression on IDENTICAL folds, S10's handcrafted features are leaving
signal on the table and S10 should be revisited. It does not enter Experiments 2-8,
does not feed S25, does not appear in the web app. It is one line in the results and
one sentence in the write-up -- deliberately no architecture search, no ensembling,
no augmentation, no saliency, no separate hyperparameter study.

--------------------------------------------------------------------------------
DATA SOURCE AND LABEL SEMANTICS (see data/processed/onednn_notes.md for the full
investigation -- this module implements what that document concluded)
--------------------------------------------------------------------------------
Input: data/raw/OneDCNN/DecisionMaking.mat's data_X, axis 0 of the trailing size-2
dimension (Player/deceiver -- verified by correlation against the archive's own
player.x, mean r=0.936 vs 0.011 for the observer slice, see onednn_notes.md S3).

Label: data_y's 4-class label, mapped through src/preprocessing.py's own
DM_CLASS_MAP -- NOT a naive literal reading of the archive's className strings.
The archive's instructed-trial labels use a swapped L/T convention already
identified and corrected by this project's own preprocessing code; using the
naive mapping gives 50% positional disagreement against trial_table.csv, the
corrected mapping gives exact 0/484 disagreement (onednn_notes.md S4). Getting
this wrong would silently train the CNN on badly-mislabeled data and make the
S11 comparison meaningless, so DM_CLASS_MAP is imported directly from
src.preprocessing rather than re-derived.

Feedback.mat is never used here (different construct: observer correctness, not
deception condition -- onednn_notes.md S2).

--------------------------------------------------------------------------------
FOLD SHARING -- the whole point of this module
--------------------------------------------------------------------------------
Exact per-trial identity join (not the dyad-level fallback) is used, because
onednn_notes.md S4-S6 established: (a) within a pair_num bucket, data_X's epoch
order matches trial_table.csv's chronological (round, trial) order exactly once
the corrected label mapping is used, and (b) src/preprocessing.py's own
compute_alignment already handles the two sessions (pair_num 20, 22) where the
archive's own epoch count is short of the trial table's, and pair_num == 21
(Player_sub22_Observer_sub19) is dropped entirely -- it is the one session
src/preprocessing.py's own alignment step found unrecoverable (ratio 0.353) and
single_brain.parquet already excludes it.

Each joined trial inherits the fold its single_brain.parquet deceiver row
received in the interpretable models' StratifiedKFold split (src/models.py's
default_splitter). Trials with no match are dropped with a printed count, never
reassigned or imputed.

--------------------------------------------------------------------------------
ARCHITECTURE -- deliberately plain (S11's instruction: this is a sanity check,
not a project in itself)
--------------------------------------------------------------------------------
3 Conv1d blocks (30->64->128->128 channels, kernel 7, BatchNorm, ReLU, MaxPool),
global average pool, dropout, single linear output -> BCEWithLogitsLoss. Adam,
early stopping on a validation slice carved out of the TRAINING fold only (never
the test fold), fixed seed, deterministic algorithms where they do not error.

--------------------------------------------------------------------------------
ENVIRONMENT -- no GPU (S5 assumed one; this environment has none, re-verified in
this run's Step 1 and reported, not silently worked around). Trained on CPU with
the already-installed torch 2.8.0+cpu build. At this data size (11,129 epochs,
30 channels, 350 samples) CPU training a shallow CNN over 5 folds is feasible in
a bounded time; if a fold's epoch count is cut short for time, that is reported
explicitly (see run() and the results file), never silently.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import preprocessing as PP  # noqa: E402  -- reuse DM_CLASS_MAP, compute_alignment
import models as M  # noqa: E402

ONEDCNN_DIR = REPO_ROOT / "data" / "raw" / "OneDCNN"
DECISIONMAKING_MAT = ONEDCNN_DIR / "DecisionMaking.mat"

SEED = M.SEED
UNRECOVERABLE_PAIR_NUM = 21  # Player_sub22_Observer_sub19, see onednn_notes.md S5-S6

# THE OneDCNN .mat file's own `className` cell array is
# ['sponL','sponT','instT','instL'] -- but every one of the 22 per-session raw
# archive files under data/raw/Preprocessed/DecisionMaking/*.mat consistently uses
# ['sponL','sponT','instL','instT'] (positions 3/4 swapped) for the SAME data_y
# integer codes (verified: data_y's numeric class index agrees with the raw
# archive's per-epoch argmax(y) index at 100% for every pair_num checked, so the
# same index means the same underlying trial in both files -- only the ATTACHED
# NAME differs). Using OneDCNN's own className array to interpret data_y is
# therefore wrong 2 times out of 4 classes; using the raw archive's consistent
# ordering (below) and then DM_CLASS_MAP is what reconciles to trial_table.csv at
# 0 mismatches (verified on pair_num 1 and 5, both 484/484 exact). This is
# independent of and additional to the DM_CLASS_MAP L/T swap documented in
# onednn_notes.md S4 -- two separate corrections are needed, not one.
_RAW_ARCHIVE_CLASS_NAMES = ["sponL", "sponT", "instL", "instT"]

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def check_environment() -> dict:
    info = {"torch_available": _HAS_TORCH}
    if _HAS_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        info["cuda_available"] = False
        info["device"] = None
    return info


# ---------------------------------------------------------------------------
# Data loading and labeling
# ---------------------------------------------------------------------------

def load_onedcnn_arrays() -> dict:
    m = sio.loadmat(
        DECISIONMAKING_MAT,
        variable_names=["data_X", "data_y", "pair_num", "className", "participant_pairs"],
    )
    return m


def build_labels(data_y: np.ndarray, class_names: list[str] = None) -> np.ndarray:
    """Corrected mapping: (1) use the raw archive's consistent className ordering
    (_RAW_ARCHIVE_CLASS_NAMES), NOT OneDCNN.mat's own mislabeled className array
    (see module-level note); (2) apply DM_CLASS_MAP's L/T swap correction
    (imported from src.preprocessing). Both corrections are required."""
    dy = data_y.ravel()
    labels = np.empty(len(dy), dtype=int)
    for i, v in enumerate(dy):
        name = _RAW_ARCHIVE_CLASS_NAMES[v - 1]
        _, condition = PP.DM_CLASS_MAP[name]
        labels[i] = 1 if condition == "lie" else 0
    return labels


def build_dm_class_key_sequence(data_y_bucket: np.ndarray, class_names: list[str] = None) -> list[tuple]:
    """Per-epoch (card_type_raw, condition) key sequence for one pair_num bucket,
    in the same format compute_alignment() expects -- built from OneDCNN's own
    data_y through the raw-archive class-name ordering + DM_CLASS_MAP, so
    compute_alignment can be reused directly without re-loading the raw
    Preprocessed .mat file. `class_names` param kept for call-site compatibility
    but intentionally unused -- see module-level note on why OneDCNN's own
    className array is not trustworthy for this purpose."""
    return [PP.DM_CLASS_MAP[_RAW_ARCHIVE_CLASS_NAMES[v - 1]] for v in data_y_bucket]


# ---------------------------------------------------------------------------
# Trial-identity join: pair_num bucket -> trial_table deceiver rows -> fold
# ---------------------------------------------------------------------------

def build_trial_fold_map(fold_assignment: pd.DataFrame) -> dict:
    """fold_assignment: DataFrame with columns [pair_id, session_id, round, trial,
    dyad_trial_seq, participant_id, fold] -- the deceiver rows' fold membership
    from the interpretable models' outer StratifiedKFold split (row order matches
    src/models.py's prepare_modeling_frame output for role='deceiver',
    feature_set='reliable_plus_marginal', pre-NaN-drop join key).

    Returns: dict pair_num -> {"session_id": str, "row_to_epoch": dict or None,
    "table_rows": DataFrame sorted by (round, trial), "ratio": float}
    """
    m = sio.loadmat(DECISIONMAKING_MAT, variable_names=["pair_num", "data_y", "className", "participant_pairs"])
    pn = m["pair_num"].ravel()
    dy = m["data_y"].ravel()
    class_names = [str(x[0]) for x in m["className"][0]]
    pp = m["participant_pairs"]
    pair_names = {i + 1: (str(pp[i][0][0]), str(pp[i][1][0])) for i in range(pp.shape[0])}

    trial_table = pd.read_csv(REPO_ROOT / "data" / "processed" / "trial_table.csv")

    out = {}
    for pnum, (player_name, observer_name) in pair_names.items():
        if pnum == UNRECOVERABLE_PAIR_NUM:
            out[pnum] = {"excluded": True, "reason": "alignment_not_recoverable"}
            continue

        player_sub = player_name.replace("Player_", "")
        observer_sub = observer_name.replace("Observer_", "")

        # Find this session's deceiver rows in trial_table: role==deceiver,
        # participant_id == player_sub, partner_id == observer_sub.
        sess_rows = trial_table[
            (trial_table["participant_id"] == player_sub)
            & (trial_table["partner_id"] == observer_sub)
            & (trial_table["role"] == "deceiver")
        ].sort_values(["round", "trial"]).reset_index(drop=True)

        idx = np.where(pn == pnum)[0]
        epoch_keys = build_dm_class_key_sequence(dy[idx], class_names)

        row_to_epoch, ratio, exclusions = PP.compute_alignment(sess_rows, epoch_keys, "DecisionMaking")

        out[pnum] = {
            "excluded": row_to_epoch is None,
            "session_participant": player_sub,
            "session_partner": observer_sub,
            "table_rows": sess_rows,
            "archive_indices": idx,
            "row_to_epoch": row_to_epoch,
            "ratio": ratio,
            "n_exclusions": len(exclusions),
        }
    return out


# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class ShallowCNN(nn.Module):
        def __init__(self, n_channels: int = 30):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            h = self.net(x).squeeze(-1)
            h = self.dropout(h)
            return self.fc(h).squeeze(-1)


def _set_determinism(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    device: str, seed: int = SEED, max_epochs: int = 40, patience: int = 6,
) -> dict:
    """Per-channel normalization uses TRAINING-fold statistics only. Validation
    slice for early stopping is carved from the training fold, never the test
    fold."""
    _set_determinism(seed)

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    rng = np.random.default_rng(seed)
    n = len(X_train_n)
    val_size = max(1, int(0.15 * n))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32)

    train_ds = TensorDataset(to_tensor(X_train_n[tr_idx]), to_tensor(y_train[tr_idx]))
    val_ds = TensorDataset(to_tensor(X_train_n[val_idx]), to_tensor(y_train[val_idx]))
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = ShallowCNN(n_channels=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(loss_fn(logits, yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(to_tensor(X_test_n).to(device)).cpu().numpy()
    y_score = 1 / (1 + np.exp(-test_logits))
    y_pred = (y_score >= 0.5).astype(int)
    metrics = M.compute_metrics(y_test, y_score, y_pred)
    metrics["n_epochs_trained"] = epoch + 1
    return metrics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    fold_assignment: pd.DataFrame,
    verbose: bool = True,
    ckpt_load_fold=None,
    ckpt_save_fold=None,
) -> dict:
    """ckpt_load_fold(fold_i) -> metrics dict or None, ckpt_save_fold(fold_i,
    metrics) -> None: optional per-fold checkpoint hooks (see
    src/experiments/exp1_baseline.py's CHECKPOINT_DIR note -- this run's
    execution environment was found to kill long-background processes after
    roughly 20-25 minutes; without per-fold checkpointing, a kill partway
    through the CNN's 5 CPU folds would lose all of them, not just the one in
    progress)."""
    env = check_environment()
    if not env["torch_available"]:
        return {"skipped": True, "reason": "torch not available", "environment": env}

    device = env["device"]
    if verbose:
        print(f"[cnn_check] device={device} torch={env['torch_version']} "
              f"cuda_available={env['cuda_available']}")

    fold_map = build_trial_fold_map(fold_assignment)

    m = sio.loadmat(DECISIONMAKING_MAT, variable_names=["data_X", "data_y", "className", "pair_num"])
    data_X = m["data_X"]  # (11129, 30, 350, 2)
    data_y = m["data_y"].ravel()
    class_names = [str(x[0]) for x in m["className"][0]]
    pn = m["pair_num"].ravel()

    labels_full = build_labels(data_y, class_names)

    # Assemble (epoch_row_index -> fold) using each pair_num bucket's row_to_epoch map.
    epoch_to_fold = {}
    n_dropped_no_match = 0
    n_excluded_session = 0
    overlap_checks = []

    for pnum, info in fold_map.items():
        if info.get("excluded"):
            idx = np.where(pn == pnum)[0]
            n_excluded_session += len(idx)
            continue
        row_to_epoch = info["row_to_epoch"]
        table_rows = info["table_rows"]
        archive_indices = info["archive_indices"]
        if row_to_epoch is None:
            n_dropped_no_match += len(archive_indices)
            continue
        for table_pos, epoch_pos in row_to_epoch.items():
            table_row = table_rows.iloc[table_pos]
            fold_row = fold_assignment[
                (fold_assignment["pair_id"] == table_row["pair_id"])
                & (fold_assignment["round"] == table_row["round"])
                & (fold_assignment["trial"] == table_row["trial"])
                & (fold_assignment["participant_id"] == table_row["participant_id"])
            ]
            if len(fold_row) != 1:
                n_dropped_no_match += 1
                continue
            global_epoch_idx = int(archive_indices[epoch_pos])
            epoch_to_fold[global_epoch_idx] = int(fold_row["fold"].iloc[0])

    n_matched = len(epoch_to_fold)
    n_total_archive = data_X.shape[0]
    if verbose:
        print(f"[cnn_check] matched {n_matched}/{n_total_archive} archive trials to a fold "
              f"(excluded_session={n_excluded_session}, no_match={n_dropped_no_match})")

    idx_arr = np.array(sorted(epoch_to_fold.keys()))
    fold_arr = np.array([epoch_to_fold[i] for i in idx_arr])

    X_player = data_X[idx_arr, :, :, 0]  # (n, 30, 350) player/deceiver slice
    y_labels = labels_full[idx_arr]

    per_fold = []
    for fold_i in sorted(set(fold_arr.tolist())):
        cached = ckpt_load_fold(fold_i) if ckpt_load_fold else None
        if cached is not None:
            per_fold.append(cached)
            if verbose:
                print(f"[cnn_check] [checkpoint] fold {fold_i}: loaded cached "
                      f"auroc={cached['auroc']:.4f}")
            continue
        test_mask = fold_arr == fold_i
        train_mask = ~test_mask
        t0 = time.time()
        metrics = train_one_fold(
            X_player[train_mask], y_labels[train_mask],
            X_player[test_mask], y_labels[test_mask],
            device=device, seed=SEED,
        )
        metrics["fold"] = int(fold_i)
        metrics["train_seconds"] = time.time() - t0
        per_fold.append(metrics)
        if ckpt_save_fold:
            ckpt_save_fold(fold_i, metrics)
        if verbose:
            print(f"[cnn_check] fold {fold_i}: auroc={metrics['auroc']:.4f} "
                  f"n_test={metrics['n_test']} epochs={metrics['n_epochs_trained']} "
                  f"time={metrics['train_seconds']:.1f}s")

    aggregate = {}
    for key in ["auroc", "balanced_accuracy", "f1", "precision", "recall"]:
        aggregate[key] = M.ci95_from_folds([f[key] for f in per_fold])

    return {
        "environment": env,
        "fold_sharing": "exact-trial",
        "n_matched_trials": n_matched,
        "n_excluded_session_trials": n_excluded_session,
        "n_no_match_trials": n_dropped_no_match,
        "n_total_archive_trials": n_total_archive,
        "overlap_fraction": n_matched / n_total_archive,
        "per_fold": per_fold,
        "mean": {k: aggregate[k]["mean"] for k in aggregate},
        "ci95": aggregate,
        "label_balance": {
            "lie": int((y_labels == 1).sum()),
            "truth": int((y_labels == 0).sum()),
        },
    }


if __name__ == "__main__":
    print("src/cnn_check.py is invoked by src/experiments/exp1_baseline.py; "
          "it needs a fold_assignment DataFrame from the interpretable models' "
          "CV split and is not meant to run standalone.")
