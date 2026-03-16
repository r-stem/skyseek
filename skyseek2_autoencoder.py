#!/usr/bin/env python
# coding: utf-8
"""
SkySeek 2.0 — Autoencoder Models

Pure model definitions:
    - Encoder: Conv1d → Transformer → attention pooling → latent z
    - Decoder: Linear inflate z → Transformer → ConvTranspose1d mirror → flux reconstruction

All data handling, safe-log transforms, and training loops live in a separate
script (`skyseek2_train.py`).

This file assumes:
    - Input spectra are already in the transformed domain (e.g. safe-log flux).
    - You may still choose how many channels to feed in; by default it expects
      3 spectra channels (flux,ivar,exptime), but `in_ch` is configurable.
    - The decoder reconstructs flux only by default (`out_ch = 1`).
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# Helper functions for convolution length matching between encoder & decoder
# ==================================================

def _conv_out_length(L_in: int, kernel: int, stride: int, padding: int) -> int:
    return (L_in + 2 * padding - kernel) // stride + 1


def _deconv_out_length(L_in: int, kernel: int, stride: int, padding: int, output_padding: int) -> int:
    return (L_in - 1) * stride - 2 * padding + kernel + output_padding


# ==================================================
# Positional encoding
# ==================================================

class PositionalEncoding1D(nn.Module):
    """
    Standard sinusoidal positional encoding for sequences.

    Expects input of shape (B, L, D). Adds PE(L, D).
    """

    def __init__(self, d_model: int, max_len: int = 8000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        return x + self.pe[:, :L, :D]


# ==================================================
# CNN layers
# ==================================================

class ConvEncoder1D(nn.Module):
    """
    1D CNN stack:

      in:  (B, in_ch, L_in)
      out: (B, c_hidden2, L_out)

    Default:
      in_ch -> 16 (k=5, s=1) -> 32 (k=15, s=2)
    """

    def __init__(
        self,
        in_ch: int = 1,
        c_hidden1: int = 16,
        c_hidden2: int = 32,
        k1: int = 5,
        s1: int = 1,
        k2: int = 15,
        s2: int = 2,
    ):
        super().__init__()
        # simple "same-ish" padding: k//2
        p1 = k1 // 2
        p2 = k2 // 2
        self.conv1 = nn.Conv1d(in_ch, c_hidden1, kernel_size=k1, stride=s1, padding=p1)
        self.conv2 = nn.Conv1d(c_hidden1, c_hidden2, kernel_size=k2, stride=s2, padding=p2)
        self.act = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm1d(c_hidden1)
        self.norm2 = nn.BatchNorm1d(c_hidden2)

        self.k1, self.s1, self.p1 = k1, s1, p1
        self.k2, self.s2, self.p2 = k2, s2, p2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_ch, L_in)
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x  # (B, c_hidden2, L_out)


# ==================================================
# Encoder: spectra -> latent z
# ==================================================

class Encoder(nn.Module):
    """
    Encoder used by the autoencoder *and* later by the classifier.

    Structure:
        spectra (B, in_ch, L)
          → ConvEncoder1D
          → transpose to (B, L_enc, C_conv)
          → Linear proj to d_model
          → PositionalEncoding1D
          → TransformerEncoder
          → attention pooling over L_enc
          → latent z (B, latent_dim)
    """

    def __init__(
        self,
        in_ch: int = 1,
        conv_c1: int = 16,
        conv_k1: int = 5,
        conv_s1: int = 1,
        conv_c2: int = 32,
        conv_k2: int = 15,
        conv_s2: int = 2,
        d_model: int = 36,
        nhead: int = 3,
        num_layers: int = 3,
        dim_feedforward: int = 144,
        dropout: float = 0.5,
        max_len: int = 8000,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()

        # CNN
        self.cnn = ConvEncoder1D(
            in_ch=in_ch,
            c_hidden1=conv_c1,
            c_hidden2=conv_c2,
            k1=conv_k1,
            s1=conv_s1,
            k2=conv_k2,
            s2=conv_s2,
        )

        # Project conv channels -> length d_model for Transformer
        self.proj = nn.Linear(conv_c2, d_model)

        # Positional encoding + Transformer encoder
        self.pos_enc = PositionalEncoding1D(d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, L, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.att_query = nn.Parameter(torch.randn(d_model))

        # Final latent dimension (for AE + classifier)
        self.d_model = d_model
        self.latent_dim = latent_dim if latent_dim is not None else d_model
        if self.latent_dim == d_model:
            self.to_latent = nn.Identity()
        else:
            self.to_latent = nn.Linear(d_model, self.latent_dim)

        # Cache conv hyperparams for length calculations (decoder mirror)
        self.conv_k1, self.conv_s1, self.conv_p1 = conv_k1, conv_s1, conv_k1 // 2
        self.conv_k2, self.conv_s2, self.conv_p2 = conv_k2, conv_s2, conv_k2 // 2

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        spectra: (B, in_ch, L)

        Returns:
            z: (B, latent_dim)
        """
        if spectra.dim() != 3:
            raise ValueError(f"Expected spectra with shape (B, C, L), got {tuple(spectra.shape)}")

        # --- CNN ---
        x = self.cnn(spectra)  # (B, C_conv, L_enc)

        # Prepare for Transformer: (B, L_enc, C_conv)
        x = x.transpose(1, 2)  # (B, L, C_conv)

        # Project channels -> d_model
        x = self.proj(x)  # (B, L, d_model)

        # Add positional encoding
        x = self.pos_enc(x)  # (B, L, d_model)

        # Transformer encoder
        x = self.transformer(x)  # (B, L, d_model)

        # --- Attention pooling over L ---
        # x: (B, L, D), att_query: (D,)
        # scores: (B, L)
        scores = torch.einsum("bld,d->bl", x, self.att_query) / math.sqrt(x.shape[-1])
        att_weights = torch.softmax(scores, dim=1)  # (B, L)
        # weighted sum → (B, D)
        latent_d = torch.einsum("bl,bld->bd", att_weights, x)  # (B, d_model)

        z = self.to_latent(latent_d)  # (B, latent_dim)
        return z

    # Convenience alias
    def encode(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.forward(spectra)


# ==================================================
# Decoder: latent z -> reconstructed flux
# ==================================================

class Decoder(nn.Module):
    """
    Decoder that mirrors the encoder structure:

        z (B, latent_dim)
          → Linear inflate to (B, T_dec, d_model)
          → PositionalEncoding1D
          → TransformerEncoder (decoder side)
          → Linear to conv feature channels
          → ConvTranspose1d stack mirroring encoder convs
          → reconstructed flux: (B, out_ch, L_spec)    
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        conv_c1: int,
        conv_k1: int,
        conv_s1: int,
        conv_c2: int,
        conv_k2: int,
        conv_s2: int,
        spec_len: int,
        nhead_dec: Optional[int] = None,
        num_layers_dec: Optional[int] = None,
        dim_feedforward_dec: Optional[int] = None,
        dropout_dec: Optional[float] = None,
        max_len: int = 8000,
        out_ch: int = 1,
    ):
        super().__init__()

        # Mirror transformer hyperparameters by default
        nhead_dec = nhead_dec if nhead_dec is not None else 3
        num_layers_dec = num_layers_dec if num_layers_dec is not None else 3
        dim_feedforward_dec = dim_feedforward_dec if dim_feedforward_dec is not None else 4 * d_model
        dropout_dec = dropout_dec if dropout_dec is not None else 0.5

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.spec_len = int(spec_len)
        self.out_ch = out_ch

        # --- Conv mirror shape calculations ---
        p1 = conv_k1 // 2
        p2 = conv_k2 // 2

        # Match transpose convolution to reverse encoder convolution
        L1_enc = _conv_out_length(self.spec_len, conv_k1, conv_s1, p1)
        L2_enc = _conv_out_length(L1_enc, conv_k2, conv_s2, p2)
        self.t_dec = L2_enc  # sequence length in decoder (T_DEC)

        # Choose output_padding values that best invert convs, if possible.
        # Search over valid output_padding ranges:
        best_op1, best_op2 = 0, 0
        exact_match = False
        for op2 in range(conv_s2):  # ConvTranspose1d restriction: 0 <= op < stride
            L1_hat = _deconv_out_length(L2_enc, conv_k2, conv_s2, p2, op2)
            for op1 in range(conv_s1):
                L0_hat = _deconv_out_length(L1_hat, conv_k1, conv_s1, p1, op1)
                if L0_hat == self.spec_len:
                    best_op1, best_op2 = op1, op2
                    exact_match = True
                    break
            if exact_match:
                break

        self.deconv_op1 = best_op1
        self.deconv_op2 = best_op2
        self._exact_length_match = exact_match

        # --- Latent → sequence (decoder transformer input) ---
        self.latent_to_seq = nn.Linear(latent_dim, d_model * self.t_dec)

        # Positional encoding + Transformer
        self.pos_enc_dec = PositionalEncoding1D(d_model=d_model, max_len=max_len)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_dec,
            dim_feedforward=dim_feedforward_dec,
            dropout=dropout_dec,
            batch_first=True,
        )
        self.transformer_dec = nn.TransformerEncoder(dec_layer, num_layers=num_layers_dec)

        # Map sequence embedding -> conv feature channels
        self.seq_to_conv = nn.Linear(d_model, conv_c2)

        # ConvTranspose stack (mirror of encoder CNN)
        self.deconv2 = nn.ConvTranspose1d(
            conv_c2,
            conv_c1,
            kernel_size=conv_k2,
            stride=conv_s2,
            padding=p2,
            output_padding=self.deconv_op2,
        )
        self.bn2 = nn.BatchNorm1d(conv_c1)
        self.act = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose1d(
            conv_c1,
            out_ch,
            kernel_size=conv_k1,
            stride=conv_s1,
            padding=p1,
            output_padding=self.deconv_op1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)

        Returns:
            recon_flux: (B, out_ch, spec_len)
        """
        if z.dim() != 2:
            raise ValueError(f"Expected latent z with shape (B, latent_dim), got {tuple(z.shape)}")

        B = z.size(0)

        # --- Inflate latent to sequence ---
        seq = self.latent_to_seq(z)  # (B, d_model * T_dec)
        seq = seq.view(B, self.t_dec, self.d_model)  # (B, T_dec, d_model)

        # Positional encoding + transformer
        seq = self.pos_enc_dec(seq)
        seq = self.transformer_dec(seq)  # (B, T_dec, d_model)

        # Sequence -> conv feature channels
        seq = self.seq_to_conv(seq)      # (B, T_dec, conv_c2)
        x = seq.transpose(1, 2)          # (B, conv_c2, T_dec)

        # ConvTranspose stack
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.deconv1(x)              # (B, out_ch, L_hat)
        L_hat = x.size(-1)

        if L_hat != self.spec_len:
            # Center-crop or trim to the exact requested length.
            if L_hat < self.spec_len:
                raise ValueError(
                    f"Decoder produced length {L_hat} < spec_len {self.spec_len}; "
                    "check conv/stride settings."
                )
            # simple center-crop
            start = (L_hat - self.spec_len) // 2
            end = start + self.spec_len
            x = x[..., start:end]

        return x


# ==================================================
# Autoencoder wrapper
# ==================================================

class Autoencoder(nn.Module):
    """
    Full autoencoder:

        Encoder: spectra → latent z
        Decoder: latent z → reconstructed flux

    Inputs:
        spectra: (B, in_ch, L_spec)
            - Expected to be in whatever transformed domain you choose
              (e.g. safe-log); no normalization is applied here.

    Methods:
        encode(spectra) -> z
        decode(z) -> recon_flux
        forward(spectra, return_latent=False) -> recon or (recon, z)
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        cfg is a plain dict containing hyperparameters. Expected keys:

            # Required:
            - "spec_len": int, original spectrum length (after preprocessing)

            # Optional (with defaults mirroring SkySeek-1 where relevant):
            - "in_ch": int, input channels for encoder (default 1)
            - "conv_c1", "conv_k1", "conv_s1"
            - "conv_c2", "conv_k2", "conv_s2"
            - "d_model"
            - "nhead", "num_layers", "dim_feedforward", "dropout"
            - "latent_dim"
            - "nhead_dec", "num_layers_dec", "dim_feedforward_dec", "dropout_dec"
            - "out_ch": decoder output channels (default 1, flux only)
        """
        super().__init__()
        self.cfg = dict(cfg)  # shallow copy so we don't mutate caller's dict

        if "spec_len" not in self.cfg:
            raise ValueError(
                "cfg['spec_len'] is required to construct the autoencoder "
                "(used for decoder mirror shapes)."
            )
        self.spec_len = int(self.cfg["spec_len"])

        # Encoder hyperparameters
        in_ch = int(self.cfg.get("in_ch", 3))
        conv_c1 = int(self.cfg.get("conv_c1", 16))
        conv_k1 = int(self.cfg.get("conv_k1", 5))
        conv_s1 = int(self.cfg.get("conv_s1", 1))
        conv_c2 = int(self.cfg.get("conv_c2", 32))
        conv_k2 = int(self.cfg.get("conv_k2", 15))
        conv_s2 = int(self.cfg.get("conv_s2", 2))
        d_model = int(self.cfg.get("d_model", 36))
        nhead = int(self.cfg.get("nhead", 3))
        num_layers = int(self.cfg.get("num_layers", 3))
        dim_feedforward = int(self.cfg.get("dim_feedforward", 4 * d_model))
        dropout = float(self.cfg.get("dropout", 0.5))
        latent_dim = int(self.cfg.get("latent_dim", d_model))

        # Decoder hyperparameters (defaults mirror encoder)
        nhead_dec = self.cfg.get("nhead_dec", nhead)
        num_layers_dec = self.cfg.get("num_layers_dec", num_layers)
        dim_feedforward_dec = self.cfg.get("dim_feedforward_dec", dim_feedforward)
        dropout_dec = self.cfg.get("dropout_dec", dropout)
        out_ch = int(self.cfg.get("out_ch", 1))

        # Build encoder and decoder
        self.encoder = Encoder(
            in_ch=in_ch,
            conv_c1=conv_c1,
            conv_k1=conv_k1,
            conv_s1=conv_s1,
            conv_c2=conv_c2,
            conv_k2=conv_k2,
            conv_s2=conv_s2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=self.cfg.get("max_len", 8000),
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            d_model=d_model,
            conv_c1=conv_c1,
            conv_k1=conv_k1,
            conv_s1=conv_s1,
            conv_c2=conv_c2,
            conv_k2=conv_k2,
            conv_s2=conv_s2,
            spec_len=self.spec_len,
            nhead_dec=nhead_dec,
            num_layers_dec=num_layers_dec,
            dim_feedforward_dec=dim_feedforward_dec,
            dropout_dec=dropout_dec,
            max_len=self.cfg.get("max_len", 8000),
            out_ch=out_ch,
        )

    # Convenience methods -------------------------------------------------

    def encode(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        spectra: (B, in_ch, L_spec)
        """
        return self.encoder(spectra)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        """
        return self.decoder(z)

    def forward(
        self,
        spectra: torch.Tensor,
        return_latent: bool = False,
    ):
        """
        spectra: (B, in_ch, L_spec)  -- in transformed domain (e.g. safe-log)

        If return_latent=False:
            returns recon_flux: (B, out_ch, L_spec)

        If return_latent=True:
            returns (recon_flux, z)
        """
        if spectra.dim() != 3:
            raise ValueError(f"Expected spectra with shape (B, C, L), got {tuple(spectra.shape)}")
        if spectra.size(-1) != self.spec_len:
            raise ValueError(
                f"spectra length {spectra.size(-1)} does not match cfg['spec_len']={self.spec_len}"
            )

        z = self.encoder(spectra)
        recon = self.decoder(z)

        if return_latent:
            return recon, z
        return recon

# ==================================================
# Factory for convenience
# ==================================================

def build_autoencoder(cfg: Dict[str, Any]) -> Autoencoder:
    """
    Convenience function so the training script can do:

        from skyseek2_autoencoder import build_autoencoder
        model = build_autoencoder(CFG)

    where CFG is a plain dict.
    """
    return Autoencoder(cfg)


if __name__ == "__main__":
    # smoke test
    L_spec = 4000
    cfg = {
        "spec_len": L_spec,
        "in_ch": 3,
        "conv_c1": 16,
        "conv_k1": 5,
        "conv_s1": 1,
        "conv_c2": 32,
        "conv_k2": 15,
        "conv_s2": 2,
        "d_model": 36,
        "nhead": 3,
        "num_layers": 3,
        "dim_feedforward": 144,
        "dropout": 0.5,
        "latent_dim": 36,
        "out_ch": 1,
    }

    model = build_autoencoder(cfg)
    x = torch.randn(2, cfg["in_ch"], L_spec)
    recon, z = model(x, return_latent=True)
    print("Input:", x.shape)
    print("Latent:", z.shape)
    print("Recon:", recon.shape)