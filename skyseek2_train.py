#!/usr/bin/env python
# coding: utf-8
"""
SkySeek 2.0 — Autoencoder Training

Trains the SkySeek 2.0 autoencoder defined in `skyseek20_autoencoder.py`.

Data / design assumptions
-------------------------
- Input spectra come from .npz files (not FITS).
- Each .npz file contains arrays (N_i, L) for:
    - flux    (linear flux)
    - ivar    (raw inverse variance)
    - exptime (raw exposure time per pixel or spectrum)
- Training iterates over files in random order each epoch.
- Within each file, samples are visited in a random order (train)
    or fixed order (test).
- Input to the model is 3 channels:
    [flux_safe_log, ivar_raw, exptime_raw]
- Test set is provided as a separate folder of npz shards

All hyperparameters (including decoder transformer depth, grad clipping,
safe-log epsilon, and LR scheduler) live in the CFG dict and are
saved into checkpoints.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import warnings
import numpy as np
import torch
import torch.utils.data as tud
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from skyseek2_autoencoder import build_autoencoder

# ==================================================
# 1. Configuration dictionary
# ==================================================
iteration=3
loss_reports_per_epoch=4        #code will output train_loss from batches, set 0 to skip
CFG: Dict[str, Any] = {
    # ---- paths ----
    "train_dir": "../data/input_shards",            # Directory containing TRAIN .npz shards
    "test_dir": "../data/input_shards/test",        # Optional directory containing TEST .npz shards; if empty/None, no test is run
    "output_root": f"../skyseek2{iteration}",       # Where to store checkpoints and logs

    # ---- fixed spectrum/time scales ----
    # These are known constants from your preprocessing / design doc.
    "spec_len": 7781,       # fixed wavelength grid length
    "base_time": 450.0,     # base exposure time (stored for design parity; not used here)

    # ---- dataloader & shuffling ----
    "batch_size": 48,
    "num_workers": 4,
    "pin_memory": True,
    "shuffle_seed": 42,     # Base seed for file/order shuffling each epoch

    # ---- training ----
    "epochs": 50,
    "lr": 2e-4,
    "weight_decay": 5e-5,
    "use_amp": True,        # mixed precision
    "grad_clip": 5.0,

    # ---- model: encoder CNN ----
    "conv_c1": 18,
    "conv_k1": 6,
    "conv_s1": 1,
    "conv_c2": 36,
    "conv_k2": 18,
    "conv_s2": 2,

    # ---- model: encoder transformer ----
    "d_model": 36,
    "nhead": 3,
    "num_layers": 2,
    "dim_feedforward": 144,
    "dropout": 0.1,
    "max_len": 8000,
    "latent_dim": 48,

    # ---- model: decoder transformer ----
    # Explicit for tuning encoder/decoder depths independently.
    "nhead_dec": 3,
    "num_layers_dec": 1,
    "dim_feedforward_dec": 144,
    "dropout_dec": 0.1,

    # ---- LR scheduler (CosineAnnealingLR) ----
    # T_max is typically set to epochs (or slightly larger).
    "scheduler_T_max": 50,
    "scheduler_eta_min": 1e-6,

    # ---- safe-log transform ----
    # y = sign(x) * log1p(safe_log_eps + |x| / safe_log_scale)
    "safe_log_eps": 1e-4,
    "safe_log_scale": 1.0,

    # ---- npz keys for spectra ----
    # Change these if your .npz files use different names.
    "npz_flux_key": "flux",
    "npz_ivar_key": "ivar",
    "npz_exptime_key": "exptime",

    # ---- model: channels ----
    # Inputs: flux, ivar, exptime (3 channels)
    "in_ch": 3,
    # Output: flux only (1 channel)
    "out_ch": 1,

    # ---- checkpointing ----
    "checkpoint_every": 1,        # epochs between checkpoints
    "resume_path": "",            # path to a checkpoint to resume from, or "" to start fresh
}

# Suppress specific PyTorch warnings that clutter the log
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Plan failed with a cudnnException:.*CUDNN_STATUS_NOT_SUPPORTED.*",
)


# ==================================================
# 2. Safe log transformation for spectra
# ==================================================

def safe_log_flux(flux: np.ndarray, eps: float, scale: float) -> np.ndarray:
    """
    Safe symmetric log transform:

        y = sign(x) * log1p(eps + |x| / scale)

    eps > 0 and scale > 0.
    """
    return (
        np.sign(flux) * np.log1p(eps + np.abs(flux) / scale)
    ).astype(np.float32)


# ==================================================
# 3. NPZ scanning utilities
# ==================================================

def find_npz_files(directory: str) -> List[Path]:
    """
    Return a sorted list of .npz files in the given directory.
    """
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = sorted(list(d.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {directory}")
    return files


def scan_npz_lengths(directory: str, flux_key: str) -> Tuple[List[Path], int, int]:
    """
    Scan .npz shards in a directory to determine:
        - file list
        - spectrum length L
        - total number of samples across shards

    Uses mmap_mode='r' to avoid loading arrays into RAM.
    """
    files = find_npz_files(directory)

    spec_len_ref: Optional[int] = None
    total_N = 0

    for p in files:
        with np.load(p, mmap_mode="r") as data:
            flux = data[flux_key]
            if flux.ndim != 2:
                raise ValueError(f"{p} flux array must be 2D (N, L); got shape {flux.shape}")
            N_i, L_i = flux.shape
            if spec_len_ref is None:
                spec_len_ref = L_i
            elif L_i != spec_len_ref:
                raise ValueError(
                    f"Inconsistent spectrum length in {p}: {L_i} vs {spec_len_ref}"
                )
            total_N += N_i

    if spec_len_ref is None:
        raise ValueError(f"No valid flux arrays found in {directory}")

    return files, spec_len_ref, total_N


# ==================================================
# 4. Sharded AE datasets
# ==================================================

class SkySeekShardedIterableDataset(IterableDataset):
    """
    Iterable Dataset that streams spectra from a list of .npz files.

    Train mode:
        - Each epoch, file order is randomized using (shuffle_seed + epoch).
        - Within each file, indices are randomized with the same RNG.
        - Each worker gets a disjoint subset of files.

    Test mode:
        - File order is deterministic (sorted list).
        - Within-file indices are sequential (0..N_i-1).

    Yields per sample:
      - spectra: Tensor (3, L) float32
            [flux_safe_log, ivar_raw, exptime_raw]
      - flux_target: Tensor (1, L) float32
            [flux_safe_log]
    """

    def __init__(
        self,
        files: List[Path],
        flux_key: str,
        ivar_key: str,
        exptime_key: str,
        safe_log_eps: float,
        spec_len: int,
        shuffle_seed: int,
        epoch: int,
        is_train: bool,
        safe_log_scale: float,
    ):
        super().__init__()
        self.files = list(files)
        self.flux_key = flux_key
        self.ivar_key = ivar_key
        self.exptime_key = exptime_key
        self.safe_log_eps = float(safe_log_eps)
        self.safe_log_scale = float(safe_log_scale)
        self.spec_len = int(spec_len)
        self.shuffle_seed = int(shuffle_seed)
        self.epoch = int(epoch)
        self.is_train = bool(is_train)
    

    def __iter__(self):
        # Local RNG for this iterator (per worker process)

        rng = np.random.RandomState(self.shuffle_seed + self.epoch)

        files = list(self.files)
        # Randomize file order per epoch (train only)
        if self.is_train:
            rng.shuffle(files)
        else:
            # ensure deterministic order in test
            files = sorted(files)

        worker_info = tud.get_worker_info()
        if worker_info is not None:
            # Split file list across workers (disjoint subsets)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(files) / num_workers))
            start = worker_id * per_worker
            end = min(start + per_worker, len(files))
            files = files[start:end]

        for p in files:
            with np.load(p) as data:
                flux = np.asarray(data[self.flux_key], dtype=np.float32)
                ivar = np.asarray(data[self.ivar_key], dtype=np.float32)
                exptime = np.asarray(data[self.exptime_key], dtype=np.float32)

            if not (flux.shape == ivar.shape == exptime.shape):
                raise ValueError(
                    f"{p} flux/ivar/exptime shapes must match; got "
                    f"{flux.shape}, {ivar.shape}, {exptime.shape}"
                )
            N_i, L_i = flux.shape
            if L_i != self.spec_len:
                raise ValueError(
                    f"{p} has spectrum length {L_i}, but CFG['spec_len']={self.spec_len}"
                )

            if self.is_train:
                idxs = np.arange(N_i)
                rng.shuffle(idxs)
            else:
                idxs = np.arange(N_i)

            for idx in idxs:
                f = flux[idx]      # (L,)
                iv = ivar[idx]     # (L,)
                ex = exptime[idx]  # (L,)

                # safe-log transform on flux only
                f_t = safe_log_flux(f, eps=self.safe_log_eps, scale=self.safe_log_scale)

                # Input to model: 3 channels
                spectra = np.stack([f_t, iv, ex], axis=0)  # (3, L)
                spectra_t = torch.from_numpy(spectra)      # float32

                # Target: flux only, shape (1, L)
                flux_target = torch.from_numpy(f_t[None, :])  # (1, L)

                yield spectra_t, flux_target


def prepare_data(cfg: Dict[str, Any]):
    """
    Scan npz files in train/test directories, verify spectrum length, and
    return file lists and counts.

    Uses cfg["spec_len"] as the expected spectrum length and raises if the
    data disagrees.
    """
    train_dir = cfg["train_dir"]
    test_dir = cfg.get("test_dir")

    flux_key = cfg["npz_flux_key"]
    spec_len_expected = int(cfg["spec_len"])

    # ---- train files ----
    train_files, spec_len_train, N_train = scan_npz_lengths(train_dir, flux_key=flux_key)
    if spec_len_train != spec_len_expected:
        raise ValueError(
            f"CFG['spec_len']={spec_len_expected}, but train spectra have length {spec_len_train}"
        )

    # ---- optional test files ----
    if test_dir and str(test_dir).strip():
        test_files, spec_len_test, N_test = scan_npz_lengths(test_dir, flux_key=flux_key)
        if spec_len_test != spec_len_expected:
            raise ValueError(
                f"CFG['spec_len']={spec_len_expected}, but test spectra have length {spec_len_test}"
            )
    else:
        test_files, N_test = [], 0

    return train_files, test_files, N_train, N_test


def make_sharded_loader(
    files: List[Path],
    cfg: Dict[str, Any],
    epoch: int,
    is_train: bool,
) -> Optional[DataLoader]:
    """
    Construct a DataLoader wrapping a SkySeekShardedIterableDataset for either
    train or Test.

    If files is empty (e.g. no test_dir), returns None.
    """
    if not files:
        return None

    ds = SkySeekShardedIterableDataset(
        files=files,
        flux_key=cfg["npz_flux_key"],
        ivar_key=cfg["npz_ivar_key"],
        exptime_key=cfg["npz_exptime_key"],
        safe_log_eps=float(cfg["safe_log_eps"]),
        safe_log_scale=float(cfg["safe_log_scale"]),
        spec_len=int(cfg["spec_len"]),
        shuffle_seed=int(cfg.get("shuffle_seed", 42)),
        epoch=epoch,
        is_train=is_train,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg.get("pin_memory", True),
        # For IterableDataset, shuffle must be False and is ignored anyway.
        shuffle=False,
        drop_last=False,
        persistent_workers=True if cfg["num_workers"] > 0 else False,
        prefetch_factor=cfg.get("prefetch_factor", 2),
    )
    return loader


# ==================================================
# 5. Training utility functions
# ==================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, torch.nn.DataParallel) else m

def setup_logging(out_dir: Path):
    """
    If running under Slurm (SLURM_JOB_ID is set), send logs only to stdout
    so they end up in the Slurm --output file.

    If not under Slurm (local debugging), log to both stdout and a file
    in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # reset

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Always log to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # If not running under Slurm, also log to a separate file
    if "SLURM_JOB_ID" not in os.environ:
        log_path = out_dir / "train.log"
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info(f"Logging to {log_path}")
    else:
        # Under Slurm: Slurm captures stdout/stderr into the --output file
        logging.info("Detected Slurm job; logging only to stdout.")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_test_loss: Optional[float],
    cfg: Dict[str, Any],
    iteration: int,
):
    m = unwrap_model(model)
    state = {
        "version": "Skyseek 2",
        "iteration": iteration,
        "epoch": epoch,
        "cfg": cfg,
        "autoencoder_state": m.state_dict(),
        "encoder_state": m.encoder.state_dict(),
        "decoder_state": m.decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
	    "best_test_loss": best_test_loss,
        "rng_state": {
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    logging.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)
    return ckpt


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    cfg: Dict[str, Any],
    batches_per_report: int,
    reports_per_epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_reports = 0
    logged_baseline = False

    use_amp = cfg.get("use_amp", False)
    grad_clip = float(cfg.get("grad_clip", 0.0))
    ngpu = torch.cuda.device_count()

    for spectra, flux_target in loader:
        spectra = spectra.to(device, non_blocking=True)          # (B, 3, L)
        flux_target = flux_target.to(device, non_blocking=True)  # (B, 1, L)

        optimizer.zero_grad(set_to_none=True)

        B = spectra.size(0)
        
        with autocast(enabled=use_amp):
            if ngpu > 1 and B < ngpu:
                # bypass DP for tiny batch (runs on device0)
                recon_flux, _ = model.module(spectra, return_latent=True)
            else:
                recon_flux, _ = model(spectra, return_latent=True)

            loss = torch.mean((recon_flux - flux_target) ** 2)

        if not logged_baseline:
            logging.info(
                f"Batch report 00/{reports_per_epoch-1:02d}, Batch #0 (pre-update baseline): "
                f"train_loss={float(loss.item()):.4e}"
            )
            logged_baseline = True

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        if batches_per_report > 0 and (n_batches % batches_per_report == 0):
            n_reports += 1
            updating_batch_loss = total_loss / n_batches
            logging.info(
            f"Batch report {n_reports:02d}/{reports_per_epoch-1}, Batch #{n_batches}: "
            f"train_loss={updating_batch_loss:.4e}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    cfg: Dict[str, Any],
) -> Optional[float]:
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    n_batches = 0
    ngpu = torch.cuda.device_count()

    for spectra, flux_target in loader:
        spectra = spectra.to(device, non_blocking=True)
        flux_target = flux_target.to(device, non_blocking=True)
        
        B = spectra.size(0)
        
        if ngpu > 1 and B < ngpu:
            # bypass DP for tiny batch (runs on device0)
            recon_flux, _ = model.module(spectra, return_latent=True)
        else:
            recon_flux, _ = model(spectra, return_latent=True)

        loss = torch.mean((recon_flux - flux_target) ** 2)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)

# ==================================================
# 6. Main training
# ==================================================

def main():
    seed_everything(42)

    out_root = Path(CFG["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    setup_logging(out_root)

    device = get_device()
    logging.info(f"Using device: {device}")

    # ---- Scan data -----
    logging.info("Scanning npz shards...")
    train_files, test_files, N_train, N_test = prepare_data(CFG)
    logging.info(f"Train files: {len(train_files)}, total samples (approx): {N_train}")
    if test_files:
        logging.info(f"Test   files: {len(test_files)}, total samples (approx): {N_test}")
    else:
        logging.info("No Test directory provided; running without test set.")
    logging.info(f"Spectrum length (spec_len): {CFG['spec_len']}")

    reports_per_epoch = loss_reports_per_epoch
    if reports_per_epoch > 0:
        N_steps = N_train // CFG["batch_size"]
        if reports_per_epoch > N_steps: 
            reports_per_epoch = 0
            logging.info("Too many batch reports requested, skipping per-batch reports.")
        batches_per_report = N_steps // max(reports_per_epoch-1,1)
        logging.info(f"Reporting train loss every {batches_per_report} batches.")
    else:
        batches_per_report = 0
        logging.info(f"Skipping per-batch reports.")

    # ---- Model ----
    logging.info("Building model...")
    model = build_autoencoder(CFG)
    #logging.info("Compiling model...") <-try reimplimenting if it gets patched
    model.to(device)
    use_dp = isinstance(model, torch.nn.DataParallel)
    ngpu = torch.cuda.device_count()
    if device.type == "cuda" and ngpu > 1:
        logging.info(f"Using DataParallel over {ngpu} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(CFG["scheduler_T_max"]),
        eta_min=float(CFG.get("scheduler_eta_min", 0.0)),
    )

    scaler = GradScaler(enabled=CFG.get("use_amp", False))

    # ---- Resume if requested ----
    start_epoch = 1
    best_test_loss = None
    resume_path_str = CFG.get("resume_path", "")
    if resume_path_str:
        ckpt = load_checkpoint(Path(resume_path_str), device)
        unwrap_model(model).load_state_dict(ckpt["autoencoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        np.random.set_state(ckpt["rng_state"]["numpy_rng_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_test_loss = ckpt.get("best_test_loss", None)
        logging.info(
            f"Resumed from epoch {ckpt['epoch']} with best_test_loss={best_test_loss}"
        )

    epochs = CFG["epochs"]
    ckpt_every = CFG.get("checkpoint_every", 1)

    #Baseline Loss
    test_loader = make_sharded_loader(test_files, CFG, epoch=0, is_train=False) if test_files else None
    if not test_files:
        logging.info(
            "---(no test set)---"
        )
    else:
        test_loss = evaluate(model, test_loader, device=device, cfg=CFG)
        logging.info(
            f"---[Epoch 00/{epochs:02d}] "
            f"test_loss={test_loss:.4e}---"
        )


    for epoch in range(start_epoch, epochs + 1):
        # Build train loader fresh each epoch so file order depends on epoch
        train_loader = make_sharded_loader(train_files, CFG, epoch=epoch, is_train=True)
    
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            cfg=CFG,
            batches_per_report=batches_per_report,
            reports_per_epoch=reports_per_epoch,
        )

        test_loss = evaluate(model, test_loader, device=device, cfg=CFG)

        if test_loss is None:
            logging.info(
                f"---[Epoch {epoch:02d}/{epochs:02d}] "
                f"train_loss={train_loss:.4e} (no test set)---"
            )
        else:
            logging.info(
                f"---[Epoch {epoch:02d}/{epochs:02d}] "
                f"train_loss={train_loss:.4e}  test_loss={test_loss:.4e}---"
            )

        # Step scheduler once per epoch
        scheduler.step()

        # Track best test loss (if we have test)
        improved = False
        if test_loss is not None:
            if best_test_loss is None or test_loss < best_test_loss:
                best_test_loss = test_loss
                improved = True

        # Save checkpoints
        if (epoch % ckpt_every == 0) or improved:
            ckpt_name = f"skyseek2{iteration}_epoch{epoch:02d}.pt"
            ckpt_path = out_root / "checkpoints" / ckpt_name
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_test_loss=best_test_loss,
                cfg=CFG,
                iteration=iteration,
            )

    logging.info("Training complete.")
    

if __name__ == "__main__":
    main()
