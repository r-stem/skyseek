#!/usr/bin/env python
# coding: utf-8
"""
SkySeek-3 model wrapper

- Wraps a frozen SkySeek-2 autoencoder (encoder + decoder).
- Adds a trainable multi-head classifier on top of the latent + metadata.
- Exposes:
    * encode(spectra) -> z
    * reconstruct(spectra, ivar=None) -> (recon_flux, mse)
    * forward(z, metadata) -> dict of logits per head

All disk I/O (FITS / NPZ / Parquet etc.) is deliberately kept out of this file.
Training / inference scripts are expected to construct tensors / arrays and then
use the helper functions and Dataset/DataLoader here.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Mapping
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from skyseek2_autoencoder import build_autoencoder

# ==================================================
# Column schema
# ==================================================

COLUMN_SCHEMA: Dict[str, Any] = {
    # Categorical metadata columns and their allowed integer IDs.
    "classes": {
        "SPECTYPE_ID": [0, 1, 2],
    },

    # Spectral channel column names and their vector length.
    "channels": {
        "FLUX": 7781,
        "IVAR": 7781,
        "EXPTIME_CHAN": 7781,
    },

    # Numeric metadata column names.
    # True/False here means: "is this scaled down by sigma?"
    "numerics": {
        "Z": False,        
        "COEFF_0": True,   
        "COEFF_1": True,
        "COEFF_2": True,
        "COEFF_3": True,
        "COEFF_4": True,
        "COEFF_5": True,
        "COEFF_6": True,
        "COEFF_7": True,
        "COEFF_8": True,
        "COEFF_9": True,
        "ZWARN_ID": False, #0/1 bit, not a vector
    },

    # Columns used in dataloading to create derived targets (e.g. ZWRONG)
    "precalc_labels": [
        "ZDIFF",
    ],

    # Label columns / generated label definitions (ZWRONG, SWRONG)
    "target_labels": {
        "SWRONG": [0, 1],
        "ZWRONG": [0, 1], # doesn't exist in starting table yet, must be calculated at load (allowing for Z_THRESH to be adjusted)
        # no RARE added yet
    },

    # Human-readable descriptions of class IDs (for interpretability only)
    "class_descriptions": {
        "SPECTYPE": ["GALAXY", "QSO", "STAR"],
        "ZWARN": [False, True],
        "SWRONG": [False, True],
        "ZWRONG": [False, True],
    },
}

def compute_metadata_dim(schema: Mapping[str, Any] = COLUMN_SCHEMA) -> int:
    """
    Compute total metadata feature dimension implied by the schema.

    - Categorical features: one-hot for each allowed ID.
    - Numeric features: each numeric feature is a single scalar (flag says whether
      it is scaled by sigma, not whether it is included).
    """
    classes = schema.get("classes", {})
    numerics = schema.get("numerics", {})

    class_dim = sum(len(v) for v in classes.values())
    numeric_dim = len(numerics)

    return class_dim + numeric_dim


# ==================================================
# Preprocessing helpers: safe-log flux
# ==================================================

def safe_log_flux_numpy(
    flux: np.ndarray,
    eps: float,
    scale: float,
) -> np.ndarray:
    """
    Safe-log transform for flux (numpy version). Matches Skyseek-2 implementation.

        y = sign(x) * log1p(|x| / scale + eps)

    where:
        - x is the original flux
        - eps and scale are taken from the SkySeek-2 cfg in the checkpoint
    """
    flux = np.asarray(flux, dtype=np.float32)
    return np.sign(flux) * np.log1p(np.abs(flux) / scale + eps).astype(np.float32)


def safe_log_flux_torch(
    flux: torch.Tensor,
    eps: float,
    scale: float,
) -> torch.Tensor:
    """
    Safe-log transform for flux (torch version).

    Same functional form as safe_log_flux_numpy; make sure eps/scale are
    taken from the SkySeek-2 cfg in the checkpoint.
    """
    return torch.sign(flux) * torch.log1p(torch.abs(flux) / scale + eps)


# ==================================================
# Preprocessing: metadata scaling by sigma
# ==================================================

def fit_metadata_sigma_df(
    df: "pd.DataFrame",
    schema: Mapping[str, Any] = COLUMN_SCHEMA,
) -> Dict[str, float]:
    """
    Compute σ (standard deviation) for each numeric metadata column.

    Returns:
        dict mapping column name -> sigma (float). Columns that are missing
        in df are silently skipped.
    """
    numerics = schema.get("numerics", {})
    stats: Dict[str, float] = {}

    for col in numerics.keys():
        if col not in df.columns:
            continue
        sigma = float(df[col].std(ddof=0)) # ddof=0 for population std
        stats[col] = sigma

    return stats


def apply_metadata_scaling_df(
    df: "pd.DataFrame",
    sigma_stats: Mapping[str, float],
    schema: Mapping[str, Any] = COLUMN_SCHEMA,
) -> "pd.DataFrame":
    """
    Apply sigma-scaling to numeric metadata columns in a pandas DataFrame.

    For each numeric column:
        - if schema["numerics"][col] is True and sigma_stats has a non-zero
          sigma for this column, divide by sigma.
        - if False, leave as raw (no scaling).

    Returns a new DataFrame.
    """
    numerics = schema.get("numerics", {})
    df_out = df.copy()

    for col, use_sigma in numerics.items():
        if not use_sigma:
            continue
        if col not in df_out.columns:
            continue
        sigma = float(sigma_stats.get(col, 0.0))
        if sigma == 0.0:
            continue
        df_out[col] = np.sign(df_out[col]) * np.log(np.abs(df_out[col]) / sigma + 1).astype(np.float32)

    return df_out


# ==================================================
# Classifier definition
# ==================================================

class SkySeek3Classifier(nn.Module):
    """
    Shared trunk + per-head branches.

    Inputs:
        z:      (B, latent_dim)
        meta:   (B, metadata_dim)
    Output:
        dict mapping head name -> logits of shape (B,)
    """

    def __init__(
        self,
        latent_dim: int,
        metadata_dim: int,
        shared_width: int = 128,
        num_shared_layers: int = 2,
        head_width: int = 64,
        num_head_layers: int = 1,
        dropout: float = 0.1,
        use_rare_head: bool = True,
        meta_hidden_dims: list[int] | None = None,
        meta_out_dim: int = 12,
        meta_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if meta_hidden_dims is None:
            meta_hidden_dims = []

        # ---- metadata perceptron ----
        mlp_layers: list[nn.Module] = []
        in_dim = metadata_dim
        for h in meta_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nn.ReLU(inplace=True))
            if meta_dropout > 0:
                mlp_layers.append(nn.Dropout(meta_dropout))
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, meta_out_dim))
        self.meta_mlp = nn.Sequential(*mlp_layers)

        classifier_input_dim = latent_dim + meta_out_dim

        # Shared trunk: Linear -> ReLU (+ Dropout) repeated
        shared_layers: List[nn.Module] = []
        d_in = classifier_input_dim
        for _ in range(num_shared_layers):
            shared_layers.append(nn.Linear(d_in, shared_width))
            shared_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            d_in = shared_width
        self.trunk = nn.Sequential(*shared_layers)

        # Per-head branches
        head_names: List[str] = ["ZWRONG", "SWRONG"]
        if use_rare_head:
            head_names.append("RARE")

        self.heads = nn.ModuleDict()
        for name in head_names:
            layers: List[nn.Module] = []
            d_h_in = shared_width
            for i in range(num_head_layers):
                # Last layer goes to 1 logit; earlier layers go to head_width.
                d_h_out = head_width if i < num_head_layers - 1 else 1
                layers.append(nn.Linear(d_h_in, d_h_out))
                if i < num_head_layers - 1:
                    layers.append(nn.ReLU(inplace=True))
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                d_h_in = d_h_out
            self.heads[name] = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        z:        (B, latent_dim)
        metadata: (B, metadata_dim)
        returns:  {"ZWRONG": logits[B], "SWRONG": logits[B], "RARE": logits[B]?}
        """
        meta_emb = self.meta_mlp(metadata)
        x = torch.cat([z, meta_emb], dim=-1)
        h = self.trunk(x)

        out: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            logit = head(h).squeeze(-1)  # (B, 1) -> (B,)
            out[name] = logit
        return out


# ==================================================
# SkySeek-3 wrapper
# ==================================================

class SkySeek3Model(nn.Module):
    """
    Wrapper around a frozen SkySeek-2 autoencoder + trainable classifier.

    API:
        encode(spectra) -> z
        reconstruct(spectra, ivar=None) -> (recon_flux, mse)
        forward(z, metadata) -> dict of logits
    """

    def __init__(
        self,
        ae_cfg: Dict[str, Any],
        ae_state_dict: Optional[Dict[str, Any]] = None,
        classifier_cfg: Optional[Dict[str, Any]] = None,
        column_schema: Mapping[str, Any] = COLUMN_SCHEMA,
    ) -> None:
        super().__init__()

        self.column_schema = dict(column_schema)

        # Build and (optionally) load the SkySeek-2 autoencoder
        self.autoencoder = build_autoencoder(ae_cfg)
        if ae_state_dict is not None:
            self.autoencoder.load_state_dict(ae_state_dict)

        # Try to infer latent_dim from cfg first, then from the autoencoder
        self.latent_dim: int = int(ae_cfg.get(
            "latent_dim",
            getattr(self.autoencoder, "latent_dim", 36),
        ))
        self.metadata_dim: int = compute_metadata_dim(self.column_schema)

        # Freeze encoder + decoder parameters
        if hasattr(self.autoencoder, "encoder"):
            for p in self.autoencoder.encoder.parameters():
                p.requires_grad = False
        if hasattr(self.autoencoder, "decoder"):
            for p in self.autoencoder.decoder.parameters():
                p.requires_grad = False

        clf_cfg = classifier_cfg or {}
        self.classifier = SkySeek3Classifier(
            latent_dim=self.latent_dim,
            metadata_dim=self.metadata_dim,
            shared_width=int(clf_cfg.get("shared_width", 128)),
            num_shared_layers=int(clf_cfg.get("num_shared_layers", 2)),
            head_width=int(clf_cfg.get("head_width", 64)),
            num_head_layers=int(clf_cfg.get("num_head_layers", 1)),
            dropout=float(clf_cfg.get("dropout", 0.1)),
            use_rare_head=bool(clf_cfg.get("use_rare_head", True)),
            meta_hidden_dims=list(clf_cfg.get("meta_hidden_dims", [])),
            meta_out_dim=int(clf_cfg.get("meta_out_dim", 12)),
            meta_dropout=float(clf_cfg.get("meta_dropout", 0.0)),
        )

    # -------------------------------------------------------------------------
    # AutoEncoder API
    # -------------------------------------------------------------------------

    def encode(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        spectra: (B, in_ch, L_spec) in the same transformed domain used by SkySeek-2.
                 (i.e., channels already safe-logged / prepared externally).

        returns: (B, latent_dim)
        """
        # Assumes the autoencoder exposes an `encode` method.
        return self.autoencoder.encode(spectra)

    def reconstruct(
        self,
        spectra: torch.Tensor,
        ivar: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        spectra: (B, in_ch, L_spec)  -- already transformed to match the AE input
        ivar:    (B, in_ch, L_spec) or (B, 1, L_spec) or None
                 This should be the same ivar channel that was fed into the AE.

        Returns:
            recon: (B, out_ch, L_spec) reconstructed spectrum (same shape as spectra)
            mse:   scalar tensor — mean squared error over valid pixels

        If ivar is provided, only pixels with ivar > 0 contribute to the MSE.
        """
        # encode + decode using the SkySeek-2 AE
        z = self.autoencoder.encode(spectra)
        recon = self.autoencoder.decode(z)

        diff_sq = (recon - spectra) ** 2

        if ivar is not None:
            # Broadcast ivar if needed and mask on ivar > 0
            mask = ivar > 0
            if mask.shape != diff_sq.shape:
                mask = mask.expand_as(diff_sq)
            valid = mask.sum()
            if valid == 0:
                mse = torch.zeros((), dtype=diff_sq.dtype, device=diff_sq.device)
            else:
                mse = diff_sq[mask].mean()
        else:
            mse = diff_sq.mean()

        return recon, mse

    # -------------------------------------------------------------------------
    # Classifier API
    # -------------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        z:        (B, latent_dim)
        metadata: (B, metadata_dim)

        Returns a dict of logits, one per head:
            {"ZWRONG": logits[B], "SWRONG": logits[B], "RARE": logits[B]?}
        """
        return self.classifier(z, metadata)


# ==================================================
# Helper: build from SkySeek-2 checkpoint
# ==================================================

def build_skyseek3_from_s2_ckpt(
    ckpt: Mapping[str, Any],
    classifier_cfg: Optional[Dict[str, Any]] = None,
    column_schema: Mapping[str, Any] = COLUMN_SCHEMA,
) -> SkySeek3Model:
    """
    Convenience helper to instantiate SkySeek-3 from a SkySeek-2 autoencoder
    checkpoint dict (the one saved by skyseek2_train.py).

    Expected keys in ckpt (based on SkySeek-2 design):
        - "cfg":              autoencoder config dict
        - "autoencoder_state": state_dict for the full AE

    Usage in training:
        s2_ckpt = torch.load(path_to_s2_ckpt, map_location="cpu")
        model = build_skyseek3_from_s2_ckpt(s2_ckpt, CLASSIFIER_CFG)
    """
    ae_cfg = dict(ckpt["cfg"])
    ae_state = ckpt["autoencoder_state"]
    return SkySeek3Model(
        ae_cfg=ae_cfg,
        ae_state_dict=ae_state,
        classifier_cfg=classifier_cfg,
        column_schema=column_schema,
    )


# ==================================================
# Tensor Dataset + DataLoader
# ==================================================

class SkySeek3TensorDataset(Dataset):
    """
    Minimal tensor-based dataset for SkySeek-3.

    IMPORTANT:
        This Dataset assumes that spectra and metadata are **already preprocessed**
        into the model's input space:

        - spectra:  (N, in_ch, L_spec)
            * flux channel already safe-logged using safe_log_flux_* with eps/scale
              from the SkySeek-2 cfg
            * ivar / exptime channels already prepared as in SkySeek-2

        - metadata: (N, metadata_dim)
            * categorical columns already one-hot encoded according to COLUMN_SCHEMA
            * numeric columns already sigma-scaled using apply_metadata_scaling_df()
              (or equivalent) with sigma_stats fitted on the **training split**

    Expects:
        spectra:  torch.Tensor of shape (N, in_ch, L_spec)
        metadata: torch.Tensor of shape (N, metadata_dim)
        targets:  optional dict of {head_name: tensor(N)} for labels
    """

    def __init__(
        self,
        spectra: torch.Tensor,
        metadata: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        surveys: Optional[object] = None,  # NEW: sequence/np array of strings
    ) -> None:
        super().__init__()
        assert spectra.shape[0] == metadata.shape[0], "spectra and metadata batch size mismatch"
        if targets is not None:
            for k, v in targets.items():
                assert v.shape[0] == spectra.shape[0], f"target {k} batch size mismatch"
        
        if surveys is not None:
            assert len(surveys) == spectra.shape[0], "surveys length mismatch"

        self.spectra = spectra
        self.metadata = metadata
        self.targets = targets or {}
        self.surveys = surveys

    def __len__(self) -> int:
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "spectra": self.spectra[idx],
            "metadata": self.metadata[idx],
        }
        for k, v in self.targets.items():
            out[k] = v[idx]
        if self.surveys is not None:
            out["SURVEY"] = str(self.surveys[idx])
        return out


def make_inference_loader(
    spectra: torch.Tensor,
    metadata: torch.Tensor,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Convenience helper to build a DataLoader for inference-time batching.

    All tensors are assumed to already be on CPU; move them to GPU in the
    training / inference loop after batching.
    """
    ds = SkySeek3TensorDataset(spectra, metadata)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )