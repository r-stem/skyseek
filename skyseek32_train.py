#!/usr/bin/env python
# coding: utf-8
"""
SkySeek-3 — Classifier Training Script

Trains the SkySeek-3 classifier on top of a **frozen** SkySeek-2 encoder.

Assumptions
-----------
- A preprocessed SkySeek table exists (FITS/FITS.GZ) containing:
    * FLUX, IVAR, EXPTIME_CHAN                (shape: (N, L))
    * Z, COEFF_i                              (scalar metadata columns)
    * SPECTYPE_ID, ZWARN_ID                   (categorical metadata)
    * SWRONG, ZDIFF                           (labels / label precursor)

- A trained SkySeek-2 autoencoder checkpoint exists, saved by skyseek2_train.py
  with keys: "cfg", "autoencoder_state", etc.

This script:
  - loads the SkySeek-2 AE checkpoint,
  - freezes encoder/decoder,
  - builds SkySeek-3 classifier on top of the latent + metadata,
  - applies safe-log to flux using S2 cfg params,
  - sigma-scales metadata numerics according to COLUMN_SCHEMA,
  - trains per-head focal-loss classifier (ZWRONG, SWRONG).
"""

from __future__ import annotations

import os
import math
import logging
from pathlib import Path
from typing import Dict, Any, Mapping, Tuple
import warnings
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from astropy.table import Table, vstack

from skyseek32_classifier import (
    COLUMN_SCHEMA,
    safe_log_flux_numpy,
    SkySeek3Model,
    SkySeek3TensorDataset,
    build_skyseek3_from_s2_ckpt,
)

# ==================================================
# 1. Configuration dictionary
# ==================================================

CFG: Dict[str, Any] = {
    # ---- paths ----
    "table_path": r"/data/vi_allsurveys_input_10.fits.gz",    # VI data
    "s2_checkpoint": r"/model/skyseek22_epoch43.pt",          # autoencoder base 
    "output_root": ".",
    "iteration": 2,

    # ---- data settings ----
    "test_fraction": 0.2,       # split fraction
    "z_thresh": 0.001,          # threshold on |ZDIFF| for ZWRONG label
    "random_seed": 42,          # random seed used for data splitting

    #Training
    "batch_size": 8,
    "epochs": 100,
    "learning_rate": 2e-4,
    "weight_decay": 5e-5,
    "use_amp": False,            # mixed precision
    "num_workers": 1,
    "pin_memory": True,
    "scheduler_T_max": 100,      # = targetted no. of epochs
    "scheduler_eta_min": 1e-6,

    # ---- model hyperparameters (classifier) ----
    "shared_width": 96,          # If num_shared_layers = 0, set = 36 + meta_dims
    "num_shared_layers": 3,
    "head_width": 48,
    "num_head_layers": 2,
    "dropout": 0.1,
    "use_rare_head": False,      # RARE head exists but lambda_R will be 0
    "main_oversampling": 2,      # 0/<=1 off, 2=double, 3=triple

    # metadata MLP
    "meta_hidden_dims": [24],    # e.g. [24] or [24,24]
    "meta_out_dim": 12,
    "meta_dropout": 0.0,

    # ---- loss hyperparameters ----
    "alpha_Z": 2/3,              # pos-class weight for Zwrong (~1/10 for 0.001)
    "alpha_S": 5/6,              # pos-class weight for Swrong (~1/20)
    "alpha_R": 0.75,
    #guidance: if you want positive samples to be upweighted by X, then alpha = X/(X+1)
    "gamma": 2.0,                # focal loss gamma; 0 => weighted BCE
    "lambda_Z": 1.0,
    "lambda_S": 2.0,
    "lambda_R": 0.0,             # no gradients for RARE for now
    "prob_threshold_Z": 0.5,     # confidence threshold for predictions, not for labels
    "prob_threshold_S": 0.5,
    "prob_threshold_R": 0.5,

    # ---- checkpointing ----
    "checkpoint_every": 87,      # save every N epochs
    "log_file": True,            # False for stdout only

}


# Suppress specific PyTorch warnings that clutter the log

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="enable_nested_tensor is True, but self.use_nested_tensor is False*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.cuda.amp.GradScaler*",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.cuda.amp.autocast*",
)


# ==================================================
# 2. Repro + logging utils
# ==================================================

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def setup_logging(log_dir: Path | None, enable_file: bool, log_name: str) -> None:
    """
    Configure logging.

    - If enable_file is True and log_dir is not None:
        Writes logs to log_dir / log_name
    - Otherwise:
        Logs only to stdout

    In all cases, stdout is enabled.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove ANY existing handlers (important when re-running in a notebook/kernel)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # --- Always attach stdout handler ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # --- Optional file handler ---
    if enable_file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_name)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ==================================================
# 3. Table loading / train-test split
# ==================================================

def load_skyseek3_table(path: str) -> Table:
    """
    Load the preprocessed SkySeek-3 training table from a FITS or FITS.GZ file.
    """
    return Table.read(path)


def train_test_split(
    table: Table,
    test_fraction: float,
    seed: int,
) -> Tuple[Table, Table]:
    """
    Random train/test split over rows.
    """
    N = len(table)
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test = int(math.floor(test_fraction * N))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    train_tab = table[train_idx]
    test_tab = table[test_idx]

    return train_tab, test_tab


# ==================================================
# 4. Metadata sigma stats + scaling
# ==================================================

def compute_metadata_sigma_stats(
    table: Table,
    schema: Mapping[str, Any] = COLUMN_SCHEMA,
) -> Dict[str, float]:
    """
    Compute σ per numeric metadata column using **training** table.

    Uses COLUMN_SCHEMA["numerics"]. Keys missing in the table are skipped.
    """
    numerics = schema.get("numerics", {})
    stats: Dict[str, float] = {}
    for col, use_sigma in numerics.items():
        # Only compute σ for columns that will actually be scaled
        if not use_sigma:
            continue
        if col not in table.colnames:
            continue
        vals = np.asarray(table[col], dtype=np.float64)
        sigma = float(vals.std(ddof=0) + 1e-8)
        stats[col] = sigma
    return stats


def build_metadata_matrix(
    table: Table,
    sigma_stats: Mapping[str, float],
    schema: Mapping[str, Any] = COLUMN_SCHEMA,
) -> np.ndarray:
    """
    Build metadata matrix (N, metadata_dim) from table according to COLUMN_SCHEMA.

    - Numeric features: one scalar per numerics key.
        * If schema["numerics"][col] is True, divide by sigma_stats[col].
        * If False, leave raw value.
    - Categorical features: one-hot encoding per classes entry.

    Order = [ all numerics (in schema["numerics"].keys() order),
              all class one-hots (for each classes key, in its value list order) ]
    """
    numerics = schema.get("numerics", {})
    classes = schema.get("classes", {})

    N = len(table)

    # ----- numeric part -----
    numeric_cols = list(numerics.keys())
    numeric_feats = []
    for col in numeric_cols:
        if col not in table.colnames:
            raise KeyError(f"Numeric metadata column {col!r} not found in table.")
        vals = np.asarray(table[col], dtype=np.float32)
        use_sigma = bool(numerics[col])
        if use_sigma:
            sigma = float(sigma_stats.get(col, 0.0))
            if sigma > 0:
                vals = np.sign(vals) * np.log(np.abs(vals) / sigma + 1).astype(np.float32)
        numeric_feats.append(vals.reshape(N, 1))

    if numeric_feats:
        numeric_mat = np.concatenate(numeric_feats, axis=1)  # (N, num_numeric)
    else:
        numeric_mat = np.zeros((N, 0), dtype=np.float32)

    # ----- categorical part (one-hot) -----
    cat_mats = []
    for col, allowed_vals in classes.items():
        if col not in table.colnames:
            raise KeyError(f"Class metadata column {col!r} not found in table.")
        raw = np.asarray(table[col], dtype=np.int64).reshape(N, 1)
        allowed = np.asarray(allowed_vals, dtype=np.int64).reshape(1, -1)
        one_hot = (raw == allowed).astype(np.float32)  # (N, n_vals)
        cat_mats.append(one_hot)

    if cat_mats:
        cat_mat = np.concatenate(cat_mats, axis=1)    # (N, class_dim)
    else:
        cat_mat = np.zeros((N, 0), dtype=np.float32)

    # ----- concat numeric + cat -----
    metadata = np.concatenate([numeric_mat, cat_mat], axis=1).astype(np.float32)
    return metadata


# ==================================================
# 5. Spectra tensor building
# ==================================================

def build_spectra_tensor(
    table: Table,
    s2_cfg: Mapping[str, Any],
) -> np.ndarray:
    """
    Build spectra tensor (N, 3, L_spec) = [safe_log_flux, IVAR_raw, EXPTIME_CHAN_raw]
    using the SAME safe-log as SkySeek-2.

    safe-log parameters (eps, scale) are taken from the SkySeek-2 cfg in the
    autoencoder checkpoint.
    """
    flux_col = "FLUX"
    ivar_col = "IVAR"
    exptime_col = "EXPTIME_CHAN"

    if flux_col not in table.colnames:
        raise KeyError(f"Flux column {flux_col!r} not found in table.")
    if ivar_col not in table.colnames:
        raise KeyError(f"IVAR column {ivar_col!r} not found in table.")
    if exptime_col not in table.colnames:
        raise KeyError(f"Exptime column {exptime_col!r} not found in table.")

    # Astropy Table has array-valued columns; stack row-wise
    flux_all = np.vstack(table[flux_col].tolist()).astype(np.float32)      # (N, L)
    ivar_all = np.vstack(table[ivar_col].tolist()).astype(np.float32)      # (N, L)
    exptime_all = np.vstack(table[exptime_col].tolist()).astype(np.float32)  # (N, L)

    eps = float(s2_cfg["safe_log_eps"])
    scale = float(s2_cfg["safe_log_scale"])

    flux_safe = safe_log_flux_numpy(flux_all, eps=eps, scale=scale)        # (N, L)

    # Stack into (N, 3, L)
    spectra = np.stack([flux_safe, ivar_all, exptime_all], axis=1).astype(np.float32)
    return spectra


# ==================================================
# 6. Labels from table
# ==================================================

def build_labels(
    table: Table,
    z_thresh: float,
) -> Dict[str, np.ndarray]:
    """
    Build binary labels for ZWRONG and SWRONG.

    - SWRONG: from column "SWRONG" (0/1).
    - ZWRONG: from |ZDIFF| >= z_thresh.
    """
    labels: Dict[str, np.ndarray] = {}

    if "SWRONG" not in table.colnames:
        raise KeyError("Label column 'SWRONG' not found in table.")
    if "ZDIFF" not in table.colnames:
        raise KeyError("Column 'ZDIFF' not found in table (needed for ZWRONG).")

    swrong = np.asarray(table["SWRONG"], dtype=np.float32)
    zdiff = np.asarray(table["ZDIFF"], dtype=np.float32)

    zwrong = (np.abs(zdiff) >= float(z_thresh)).astype(np.float32)

    labels["SWRONG"] = swrong
    labels["ZWRONG"] = zwrong

    # Placeholder for RARE if/when labels are obtained:
    # labels["RARE"] = np.zeros_like(swrong, dtype=np.float32)

    return labels


# ==================================================
# 7. Focal loss + metrics (from 1.1)
# ==================================================

def focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """
    Binary focal loss for logits + {0,1} labels.

    alpha: weight for positive class (neg gets 1-alpha)
    gamma: focusing parameter; gamma=0 -> (class-weighted) BCE.
    """
    logits = logits.float()
    labels = labels.float()

    # BCE per example
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    # Probabilities
    probs = torch.sigmoid(logits)

    # p_t and alpha_t (standard focal loss form)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)

    # Focal modulation
    focal_factor = (1.0 - p_t).pow(gamma)

    loss = alpha_t * focal_factor * bce
    return loss.mean()


def compute_head_loss_and_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    gamma: float,
    threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute focal loss + confusion counts + F1 for a single head.

    Returns:
      loss: scalar tensor
      metrics: dict with tp, tn, fp, fn, n, pos_recall, neg_recall, f1, acc
    """
    loss = focal_loss_with_logits(logits, labels, alpha=alpha, gamma=gamma)

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        labels_bin = labels.float()

        tp = ((preds == 1.0) & (labels_bin == 1.0)).sum().item()
        tn = ((preds == 0.0) & (labels_bin == 0.0)).sum().item()
        fp = ((preds == 1.0) & (labels_bin == 0.0)).sum().item()
        fn = ((preds == 0.0) & (labels_bin == 1.0)).sum().item()
        n = labels_bin.numel()

        pos_recall = tp / max(tp + fn, 1)
        neg_recall = tn / max(tn + fp, 1)
        acc = (tp + tn) / max(n, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * pos_recall / max(precision + pos_recall, 1e-8)

    metrics = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": n,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "acc": acc,
        "f1": f1,
    }
    return loss, metrics

def _compute_rates_and_f1(counts: Mapping[str, int]) -> Dict[str, float]:
    """Given counts dict with tp, tn, fp, fn, return tpr, tnr, fpr, fnr, f1."""
    tp = counts.get("tp", 0)
    tn = counts.get("tn", 0)
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)

    pos = tp + fn
    neg = tn + fp

    tpr = tp / pos if pos > 0 else float("nan")   # recall / sensitivity
    tnr = tn / neg if neg > 0 else float("nan")   # specificity
    fpr = fp / neg if neg > 0 else float("nan")
    fnr = fn / pos if pos > 0 else float("nan")

    denom_f1 = 2 * tp + fp + fn
    f1 = (2 * tp / denom_f1) if denom_f1 > 0 else float("nan")

    return {"tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr, "f1": f1}

def find_best_f1_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_grid: int = 401,
):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = labels.cpu().numpy()

    best_f1 = -1.0
    best_t = 0.5

    for t in np.linspace(0.0, 1.0, n_grid):
        preds = (probs >= t).astype(np.float32)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def add_epoch_record(
    history: list,
    epoch: int,
    train_loss_total: float,
    test_loss_total: float,
    train_counts: Mapping[str, Mapping[str, int]],
    test_counts: Mapping[str, Mapping[str, int]],
) -> None:
    """Append one epoch's metrics (train + test) into history list."""
    # Map heads to short keys to match SkySeek-1.1 style names
    tr_z_stats = _compute_rates_and_f1(train_counts["ZWRONG"])
    tr_s_stats = _compute_rates_and_f1(train_counts["SWRONG"])
    te_z_stats = _compute_rates_and_f1(test_counts["ZWRONG"])
    te_s_stats = _compute_rates_and_f1(test_counts["SWRONG"])

    record = {
        "epoch": epoch,
        "train_loss_total": train_loss_total,
        "test_loss_total": test_loss_total,

        # Confusion-rate style metrics: ZWRONG
        "train_z_tpr": tr_z_stats["tpr"],
        "test_z_tpr":  te_z_stats["tpr"],
        "train_z_tnr": tr_z_stats["tnr"],
        "test_z_tnr":  te_z_stats["tnr"],
        "train_z_fpr": tr_z_stats["fpr"],
        "test_z_fpr":  te_z_stats["fpr"],
        "train_z_fnr": tr_z_stats["fnr"],
        "test_z_fnr":  te_z_stats["fnr"],

        # Confusion-rate style metrics: SWRONG
        "train_s_tpr": tr_s_stats["tpr"],
        "test_s_tpr":  te_s_stats["tpr"],
        "train_s_tnr": tr_s_stats["tnr"],
        "test_s_tnr":  te_s_stats["tnr"],
        "train_s_fpr": tr_s_stats["fpr"],
        "test_s_fpr":  te_s_stats["fpr"],
        "train_s_fnr": tr_s_stats["fnr"],
        "test_s_fnr":  te_s_stats["fnr"],

        # F1
        "train_z_f1": tr_z_stats["f1"],
        "test_z_f1":  te_z_stats["f1"],
        "train_s_f1": tr_s_stats["f1"],
        "test_s_f1":  te_s_stats["f1"],
    }

    history.append(record)


# ==================================================
# 8. Train / Eval functions
# ==================================================

def train_one_epoch(
    model: SkySeek3Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    cfg: Mapping[str, Any],
) -> Tuple[float, Dict[str, Dict[str, int]]]:
    model.train()
    total_loss = 0.0
    n_batches = 0

    use_amp = bool(cfg.get("use_amp", False))

    alpha_Z = float(cfg["alpha_Z"])
    alpha_S = float(cfg["alpha_S"])
    alpha_R = float(cfg["alpha_R"])
    gamma = float(cfg["gamma"])
    lambda_Z = float(cfg["lambda_Z"])
    lambda_S = float(cfg["lambda_S"])
    lambda_R = float(cfg["lambda_R"])
    th_Z = float(cfg["prob_threshold_Z"])
    th_S = float(cfg["prob_threshold_S"])
    th_R = float(cfg["prob_threshold_R"])

    # aggregate confusion counts per head on TRAIN
    train_counts: Dict[str, Dict[str, int]] = {
        "ZWRONG": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
        "SWRONG": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
        "RARE":   {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
    }

    for batch_idx, batch in enumerate(loader):
        spectra = batch["spectra"].to(device, non_blocking=True)    # (B, 3, L)
        metadata = batch["metadata"].to(device, non_blocking=True)  # (B, metadata_dim)
        labels_Z = batch["ZWRONG"].to(device, non_blocking=True)    # (B,)
        labels_S = batch["SWRONG"].to(device, non_blocking=True)    # (B,)
        labels_R = batch.get("RARE")
        if labels_R is not None:
            labels_R = labels_R.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            # Encode **without** building a backward graph through the autoencoder
            with torch.no_grad():
                z = model.encode(spectra)              # (B, latent_dim), detached
            logits = model(z, metadata)                # dict of heads (classifier has grads)

            loss_Z, mZ = compute_head_loss_and_metrics(
                logits["ZWRONG"], labels_Z,
                alpha=alpha_Z, gamma=gamma,
                threshold=th_Z,
            )
            loss_S, mS = compute_head_loss_and_metrics(
                logits["SWRONG"], labels_S,
                alpha=alpha_S, gamma=gamma,
                threshold=th_S,
            )

            if "RARE" in logits and lambda_R != 0.0 and labels_R is not None:
                loss_R, mR = compute_head_loss_and_metrics(
                    logits["RARE"], labels_R,
                    alpha=alpha_R, gamma=gamma,
                    threshold=th_R,
                )
            else:
                loss_R = torch.zeros((), device=device)
                mR = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}

            loss = lambda_Z * loss_Z + lambda_S * loss_S + lambda_R * loss_R

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        # aggregate counts
        for name, m in zip(["ZWRONG", "SWRONG", "RARE"], [mZ, mS, mR]):
            for key in ["tp", "tn", "fp", "fn", "n"]:
                train_counts[name][key] += int(m[key])

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, train_counts


@torch.no_grad()
def evaluate(
    model: SkySeek3Model,
    loader: DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
) -> Tuple[
    float,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, Dict[str, int]]],  # per-survey counts
]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    alpha_Z = float(cfg["alpha_Z"])
    alpha_S = float(cfg["alpha_S"])
    alpha_R = float(cfg["alpha_R"])
    gamma = float(cfg["gamma"])
    lambda_Z = float(cfg["lambda_Z"])
    lambda_S = float(cfg["lambda_S"])
    lambda_R = float(cfg["lambda_R"])
    th_Z = float(cfg["prob_threshold_Z"])
    th_S = float(cfg["prob_threshold_S"])
    th_R = float(cfg["prob_threshold_R"])

    # --------------------------
    # Global confusion counters
    # --------------------------
    agg = {
        "ZWRONG": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
        "SWRONG": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
        "RARE":   {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
    }

    # -----------------------------
    # Per-survey confusion counters
    # -----------------------------
    survey_counts = {
        "sv1": {
            "ZWRONG": {
                "th05": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
                "best": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
                },
            "SWRONG": {
                "th05": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
                "best": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
                },
        },
        "main": {
            "ZWRONG": {
                "th05": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
                "best": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
                },
            "SWRONG": {
                "th05": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
                "best": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
                },
        },
        "sv3": {
            "ZWRONG": {
                "th05": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
                "best": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
                },
            "SWRONG": {
                "th05": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0},
                "best": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}
                },
        },
    }

    # --------------------------------
    # Accumulators for threshold sweep
    # --------------------------------
    all_logits_Z, all_labels_Z = [], []
    all_logits_S, all_labels_S = [], []
    all_surveys = []

    # ==========================
    # Evaluation loop
    # ==========================
    for batch in loader:
        spectra = batch["spectra"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        labels_Z = batch["ZWRONG"].to(device, non_blocking=True)
        labels_S = batch["SWRONG"].to(device, non_blocking=True)

        labels_R = batch.get("RARE")
        if labels_R is not None:
            labels_R = labels_R.to(device, non_blocking=True)

        surveys = batch["SURVEY"]  # numpy array of strings
        all_surveys.extend([str(s) for s in surveys])

        z = model.encode(spectra)
        logits = model(z, metadata)

        loss_Z, mZ = compute_head_loss_and_metrics(
            logits["ZWRONG"], labels_Z,
            alpha=alpha_Z, gamma=gamma,
            threshold=th_Z,
        )
        loss_S, mS = compute_head_loss_and_metrics(
            logits["SWRONG"], labels_S,
            alpha=alpha_S, gamma=gamma,
            threshold=th_S,
        )

        if "RARE" in logits and lambda_R != 0.0 and labels_R is not None:
            loss_R, mR = compute_head_loss_and_metrics(
                logits["RARE"], labels_R,
                alpha=alpha_R, gamma=gamma,
                threshold=th_R,
            )
        else:
            loss_R = torch.zeros((), device=device)
            mR = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0}

        loss = lambda_Z * loss_Z + lambda_S * loss_S + lambda_R * loss_R

        total_loss += float(loss.item())
        n_batches += 1

        # --------------------------
        # Aggregate global counts
        # --------------------------
        for name, m in zip(["ZWRONG", "SWRONG", "RARE"], [mZ, mS, mR]):
            for k in ["tp", "tn", "fp", "fn", "n"]:
                agg[name][k] += int(m[k])

        # ---------------------------
        # Accumulate logits for sweep
        # ---------------------------
        all_logits_Z.append(logits["ZWRONG"].detach())
        all_labels_Z.append(labels_Z.detach())
        all_logits_S.append(logits["SWRONG"].detach())
        all_labels_S.append(labels_S.detach())

        # --------------------------
        # Per-survey confusion
        # --------------------------
        preds_Z = (torch.sigmoid(logits["ZWRONG"]) >= th_Z).cpu().numpy()
        preds_S = (torch.sigmoid(logits["SWRONG"]) >= th_S).cpu().numpy()
        labels_Z_np = labels_Z.cpu().numpy()
        labels_S_np = labels_S.cpu().numpy()

        for i, survey in enumerate(surveys):
            if survey not in survey_counts:
                continue

            # ZWRONG
            if preds_Z[i] == 1 and labels_Z_np[i] == 1:
                survey_counts[survey]["ZWRONG"]["th05"]["tp"] += 1
            elif preds_Z[i] == 1 and labels_Z_np[i] == 0:
                survey_counts[survey]["ZWRONG"]["th05"]["fp"] += 1
            elif preds_Z[i] == 0 and labels_Z_np[i] == 0:
                survey_counts[survey]["ZWRONG"]["th05"]["tn"] += 1
            elif preds_Z[i] == 0 and labels_Z_np[i] == 1:
                survey_counts[survey]["ZWRONG"]["th05"]["fn"] += 1
            survey_counts[survey]["ZWRONG"]["th05"]["n"] += 1

            # SWRONG
            if preds_S[i] == 1 and labels_S_np[i] == 1:
                survey_counts[survey]["SWRONG"]["th05"]["tp"] += 1
            elif preds_S[i] == 1 and labels_S_np[i] == 0:
                survey_counts[survey]["SWRONG"]["th05"]["fp"] += 1
            elif preds_S[i] == 0 and labels_S_np[i] == 0:
                survey_counts[survey]["SWRONG"]["th05"]["tn"] += 1
            elif preds_S[i] == 0 and labels_S_np[i] == 1:
                survey_counts[survey]["SWRONG"]["th05"]["fn"] += 1
            survey_counts[survey]["SWRONG"]["th05"]["n"] += 1

    avg_loss = total_loss / max(n_batches, 1)

    # ==========================
    # Threshold sweep (once)
    # ==========================
    all_logits_Z = torch.cat(all_logits_Z)
    all_labels_Z = torch.cat(all_labels_Z)
    all_logits_S = torch.cat(all_logits_S)
    all_labels_S = torch.cat(all_labels_S)

    best_t_Z, best_f1_Z = find_best_f1_threshold(all_logits_Z, all_labels_Z)
    best_t_S, best_f1_S = find_best_f1_threshold(all_logits_S, all_labels_S)

    probs_Z = torch.sigmoid(all_logits_Z).cpu().numpy()
    probs_S = torch.sigmoid(all_logits_S).cpu().numpy()
    labels_Z_np = all_labels_Z.cpu().numpy()
    labels_S_np = all_labels_S.cpu().numpy()

    preds_Z_best = (probs_Z >= best_t_Z).astype(np.int32)
    preds_S_best = (probs_S >= best_t_S).astype(np.int32)

    surveys_np = np.asarray(all_surveys, dtype=object)

    for i, survey in enumerate(surveys_np):
        if survey not in survey_counts:
            continue

        # ---- ZWRONG @ best threshold ----
        if preds_Z_best[i] == 1 and labels_Z_np[i] == 1:
            survey_counts[survey]["ZWRONG"]["best"]["tp"] += 1
        elif preds_Z_best[i] == 1 and labels_Z_np[i] == 0:
            survey_counts[survey]["ZWRONG"]["best"]["fp"] += 1
        elif preds_Z_best[i] == 0 and labels_Z_np[i] == 0:
            survey_counts[survey]["ZWRONG"]["best"]["tn"] += 1
        elif preds_Z_best[i] == 0 and labels_Z_np[i] == 1:
            survey_counts[survey]["ZWRONG"]["best"]["fn"] += 1
        survey_counts[survey]["ZWRONG"]["best"]["n"] += 1

        # ---- SWRONG @ best threshold ----
        if preds_S_best[i] == 1 and labels_S_np[i] == 1:
            survey_counts[survey]["SWRONG"]["best"]["tp"] += 1
        elif preds_S_best[i] == 1 and labels_S_np[i] == 0:
            survey_counts[survey]["SWRONG"]["best"]["fp"] += 1
        elif preds_S_best[i] == 0 and labels_S_np[i] == 0:
            survey_counts[survey]["SWRONG"]["best"]["tn"] += 1
        elif preds_S_best[i] == 0 and labels_S_np[i] == 1:
            survey_counts[survey]["SWRONG"]["best"]["fn"] += 1
        survey_counts[survey]["SWRONG"]["best"]["n"] += 1

    # ==========================
    # Final metrics per head
    # ==========================
    metrics_per_head: Dict[str, Dict[str, float]] = {}

    for name, counts in agg.items():
        tp, tn, fp, fn, n = (
            counts["tp"],
            counts["tn"],
            counts["fp"],
            counts["fn"],
            counts["n"],
        )

        if n == 0:
            metrics_per_head[name] = {
                "pos_recall": 0.0,
                "neg_recall": 0.0,
                "precision": 0.0,
                "acc": 0.0,
                "f1": 0.0,
            }
            continue

        pos_recall = tp / max(tp + fn, 1)
        neg_recall = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        acc = (tp + tn) / n
        f1 = 2 * precision * pos_recall / max(precision + pos_recall, 1e-8)

        metrics = {
            "pos_recall": pos_recall,
            "neg_recall": neg_recall,
            "precision": precision,
            "acc": acc,
            "f1": f1,
        }

        if name == "ZWRONG":
            metrics["best_threshold"] = best_t_Z
            metrics["best_f1"] = best_f1_Z
        elif name == "SWRONG":
            metrics["best_threshold"] = best_t_S
            metrics["best_f1"] = best_f1_S

        metrics_per_head[name] = metrics

    return avg_loss, metrics_per_head, agg, survey_counts


# ==================================================
# 9. Checkpoint util
# ==================================================

def save_checkpoint(
    out_dir: Path,
    epoch: int,
    iteration: int,
    model: SkySeek3Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler | None,
    s2_cfg: Mapping[str, Any],
    sigma_stats: Mapping[str, float],
    test_loss: float,
    cfg: Mapping[str, Any],
    is_best: bool = False,
) -> None:
    state = {
        "version": "SkySeek-3",
        "iteration": iteration,
        "epoch": epoch,
        "test_loss": float(test_loss),
        "s2_cfg": dict(s2_cfg),
        "s3_cfg": dict(cfg),
        "autoencoder_state": model.autoencoder.state_dict(),
        "classifier_state": model.classifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "sigma_stats": dict(sigma_stats),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"skyseek32{iteration}_epoch{epoch:03d}.pt"
    torch.save(state, ckpt_path)

    if is_best:
        best_path = out_dir / f"skyseek32{iteration}_best_{epoch:03d}.pt"
        torch.save(state, best_path)

# ==================================================
# 10. Main
# ==================================================

def main():
    
    seed_everything(CFG["random_seed"])

    iteration = CFG["iteration"]

    # Base directory for this run:
    # <output_root>/skyseek32{iteration}/
    base_root = Path(CFG["output_root"])        
    run_dir = base_root / f"skyseek32{iteration}"

    # Subdirectories:
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    # Logging setup
    log_to_file = bool(CFG["log_file"])
    log_name = f"skyseek32{iteration}_training.log"

    # If log_to_file is False, logs_dir is ignored by setup_logging
    setup_logging(logs_dir if log_to_file else None, log_to_file, log_name)

    logging.info("SkySeek-3 training start")
    logging.info(f"Config: {CFG}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if log_to_file:
        logging.info(f"Logging to file: {(logs_dir / log_name).resolve()}")
    else:
        logging.info("Logging to stdout only (no log file).")


    # ---- Load SkySeek-2 AE checkpoint ----
    s2_ckpt_path = Path(CFG["s2_checkpoint"])
    if not s2_ckpt_path.exists():
        raise FileNotFoundError(f"SkySeek-2 checkpoint not found: {s2_ckpt_path}")
    s2_ckpt = torch.load(s2_ckpt_path, map_location="cpu")
    s2_cfg = s2_ckpt["cfg"]
    logging.info(f"Loaded encoder from checkpoint {s2_ckpt_path}")

    # ---- Build SkySeek-3 model ----
    classifier_cfg = {
        "shared_width": CFG["shared_width"],
        "num_shared_layers": CFG["num_shared_layers"],
        "head_width": CFG["head_width"],
        "num_head_layers": CFG["num_head_layers"],
        "dropout": CFG["dropout"],
        "use_rare_head": CFG["use_rare_head"],
        "meta_hidden_dims":CFG["meta_hidden_dims"],
        "meta_out_dim":CFG["meta_out_dim"],
        "meta_dropout":CFG.get("meta_dropout", 0.0),
    }
    model = build_skyseek3_from_s2_ckpt(s2_ckpt, classifier_cfg=classifier_cfg)
    model.to(device)
    logging.info("SkySeek-3 model built and moved to device.")

    # ---- Load table and split ----
    table_path = CFG["table_path"]
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Table not found: {table_path}")
    table = load_skyseek3_table(table_path)
    logging.info(f"Loaded table with {len(table):,} rows from {table_path}")

    train_tab, test_tab = train_test_split(
        table,
        test_fraction=CFG["test_fraction"],
        seed=CFG["random_seed"],
    )
    logging.info(
        f"Train/test split: {len(train_tab):,} train rows, {len(test_tab):,} test rows "
        f"(test_fraction={CFG['test_fraction']})"
    )

    # ---- Compute metadata sigma stats on TRAIN ONLY ----
    sigma_stats = compute_metadata_sigma_stats(train_tab, schema=COLUMN_SCHEMA)
    logging.info(f"Computed sigma stats for numerics: {sigma_stats}")

    k = int(CFG.get("main_oversampling", 0))
    if k > 1:
        is_main = (train_tab["SURVEY"] == "main")
        main_rows = train_tab[is_main]
        if len(main_rows) > 0:
            train_tab = vstack([train_tab] + [main_rows] * (k - 1), metadata_conflicts="silent")
            logging.info(f"Oversampled MAIN by x{k}: new train size = {len(train_tab):,}")

    # ---- Build TRAIN tensors ----
    train_spectra = build_spectra_tensor(train_tab, s2_cfg)          # (N_tr, 3, L)
    train_meta = build_metadata_matrix(train_tab, sigma_stats, COLUMN_SCHEMA)  # (N_tr, D_meta)
    train_labels = build_labels(train_tab, z_thresh=CFG["z_thresh"])

    # ---- Build test tensors ----
    test_spectra = build_spectra_tensor(test_tab, s2_cfg)
    test_meta = build_metadata_matrix(test_tab, sigma_stats, COLUMN_SCHEMA)
    test_labels = build_labels(test_tab, z_thresh=CFG["z_thresh"])

    # ---- Wrap into datasets ----
    train_targets = {
        "ZWRONG": torch.from_numpy(train_labels["ZWRONG"]),
        "SWRONG": torch.from_numpy(train_labels["SWRONG"]),
        # "RARE": ... placeholder
    }
    test_targets = {
        "ZWRONG": torch.from_numpy(test_labels["ZWRONG"]),
        "SWRONG": torch.from_numpy(test_labels["SWRONG"]),
    }
    test_surveys = np.asarray(test_tab["SURVEY"]).astype(str)

    train_ds = SkySeek3TensorDataset(
        spectra=torch.from_numpy(train_spectra),    # (N_tr, 3, L)
        metadata=torch.from_numpy(train_meta),      # (N_tr, D_meta)
        targets=train_targets,
    )
    test_ds = SkySeek3TensorDataset(
        spectra=torch.from_numpy(test_spectra),
        metadata=torch.from_numpy(test_meta),
        targets=test_targets,
        surveys=test_surveys,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        drop_last=False,
    )

    logging.info(
        f"Train loader: {len(train_ds):,} samples, "
        f"test loader: {len(test_ds):,} samples, "
        f"Batch size: {CFG['batch_size']}"
    )

    # ---- Optimizer / scaler ----
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),  # only classifier is trainable
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )
    scaler = GradScaler("cuda", enabled=CFG.get("use_amp", False))

    #Scheduler definition
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(CFG["scheduler_T_max"]),
        eta_min=float(CFG["scheduler_eta_min"]),
    )

    # Track best test F1 per head (ZWRONG, SWRONG)
    best_test_z_f1: float = 0.0
    best_test_s_f1: float = 0.0
    history: list[Dict[str, Any]] = []
    survey_history_rows: list[dict[str, Any]] = []

    # ---- Training loop ----
    for epoch in range(1, CFG["epochs"] + 1):
        logging.info(f"===== Epoch {epoch}/{CFG['epochs']} =====")

        # ---- Train ----
        train_loss, train_counts = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            cfg=CFG,
        )

        # ---- Evaluate ----
        test_loss, test_metrics, test_counts, survey_counts = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            cfg=CFG,
        )
        scheduler.step()

        # ---- History + compact epoch summary (SkySeek-1.1 style) ----
        add_epoch_record(
            history=history,
            epoch=epoch,
            train_loss_total=train_loss,
            test_loss_total=test_loss,
            train_counts=train_counts,
            test_counts=test_counts,
        )
        rec = history[-1]
        # ---- Attach best-threshold metrics from evaluate() ----
        rec["test_z_best_f1"] = test_metrics["ZWRONG"].get("best_f1", rec["test_z_f1"])
        rec["test_s_best_f1"] = test_metrics["SWRONG"].get("best_f1", rec["test_s_f1"])
        rec["test_z_best_threshold"] = test_metrics["ZWRONG"].get("best_threshold", CFG["prob_threshold_Z"])
        rec["test_s_best_threshold"] = test_metrics["SWRONG"].get("best_threshold", CFG["prob_threshold_S"])
        
        # ---- Per-survey metrics (per epoch) ----
        row = {"epoch": epoch}
        for survey, heads in survey_counts.items():
            for head, d in heads.items():
                for mode in ["th05", "best"]:
                    counts = d[mode]
                    stats = _compute_rates_and_f1(counts)
                    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        
                    prefix = f"{survey}_{head}_{mode}"
                    row[f"{prefix}_n"] = counts["n"]
                    row[f"{prefix}_tpr"] = stats["tpr"]
                    row[f"{prefix}_tnr"] = stats["tnr"]
                    row[f"{prefix}_precision"] = precision
                    row[f"{prefix}_f1"] = stats["f1"]
        
        survey_history_rows.append(row)



        logging.info(
            f"Epoch {epoch:02d}/{CFG['epochs']} | "
            f"TrainLoss={rec['train_loss_total']:.4f}  "
            f"TestLoss={rec['test_loss_total']:.4f} | "
            f"TestZWrong:  TPR={rec['test_z_tpr']:.2%} F1={rec['test_z_best_f1']:.3f} | "
            f"TestSWrong:  TPR={rec['test_s_tpr']:.2%} F1={rec['test_s_best_f1']:.3f}"
        )

        # Checkpointing based on per-head test F1
        z_f1 = rec["test_z_best_f1"]
        s_f1 = rec["test_s_best_f1"]

        is_best = False
        if (z_f1 > best_test_z_f1) and (s_f1 > best_test_s_f1):
            best_test_z_f1 = z_f1
            best_test_s_f1 = s_f1
            is_best = True
            #saves best checkpoints along epochs where both metrics improve

        if (epoch % CFG["checkpoint_every"] == 0) or is_best:
            save_checkpoint(
                out_dir=ckpt_dir,
                epoch=epoch,
                iteration=CFG['iteration'],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                s2_cfg=s2_cfg,
                sigma_stats=sigma_stats,
                test_loss=test_loss,
                cfg=CFG,
                is_best=is_best,
            )
            logging.info(
                f"Saved checkpoint at epoch {epoch} "
                f"(peak z_f1={max(rec['test_z_best_f1'] for rec in history):.4f},"
                f" peak s_f1={max(rec['test_s_best_f1'] for rec in history):.4f})"
            )


        # ---- Save epoch history to CSV ----
    if history:
        loss_dir = Path(CFG['output_root']) / f'skyseek32{CFG["iteration"]}/logs'
        loss_dir.mkdir(parents=True, exist_ok=True)
        iteration = CFG['iteration']
        csv_path = loss_dir / f'Skyseek32{iteration}loss.csv'
        df_history = pd.DataFrame(history).set_index('epoch')
        df_history.to_csv(csv_path, index=True)
        logging.info(f'Epoch history saved to {csv_path.resolve()}')
        # ---- Save per-survey test metrics ----
    if survey_history_rows:
        survey_csv = loss_dir / f"Skyseek32{iteration}loss_surveys.csv"
        pd.DataFrame(survey_history_rows).set_index("epoch").to_csv(survey_csv)
        logging.info(f"Per-survey-by-epoch metrics saved to {survey_csv.resolve()}")


    logging.info("Training finished.")


if __name__ == "__main__":
    main()
