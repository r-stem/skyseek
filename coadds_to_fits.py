#!/usr/bin/env python

"""
Extracts desired data from all DESI coadded spectra in a directory.
Parallelized across workers for efficiency.
Outputs compressed .fits.gz files.

Extracted features: FLUX, IVAR, TARGETID, COADD_EXPTIME
"""

from __future__ import annotations
import logging
from pathlib import Path
from astropy.table import Table
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ==================================================
# File paths
# ==================================================

COADD_DIR = Path("...")                             # input directory with coadded spectra
TARGET_DIR = Path("...")                            # output directory 
LOG_FILE = Path("./logs/hpx_main_batch1.5.log")     # text log file

# ==================================================
# Logging
# ==================================================

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("healpix_processing")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger

logger = setup_logger(LOG_FILE)

# ==================================================
# Core extraction logic
# ==================================================

def flux_ivar_exptime_extract(path: str, float32: bool = True):
    """
    Read coadd file at `path` and return an Astropy Table with
    TARGETID, COADD_EXPTIME, FLUX, IVAR
    with the latter two coadded into a single vector with DESI official tools
    """
    unneeded_hdus = ["RESOLUTION", "MASK", "EXP_FIBERMAP", "SCORES"]
    meta_columns = ["TARGETID", "COADD_EXPTIME"]

    spectra = read_spectra(
        path,
        single=float32,
        skip_hdus=unneeded_hdus,
        select_columns=dict(FIBERMAP=meta_columns),
    )

    combined_spectra = coadd_cameras(spectra)
    logger.info(
        "Retrieved %d spectra from %s",
        combined_spectra.num_spectra(),
        path,
    )

    band = combined_spectra.bands[0]

    extracted_data = Table()
    for col in meta_columns:
        extracted_data[col] = combined_spectra.fibermap[col]
    extracted_data["FLUX"] = combined_spectra.flux[band]
    extracted_data["IVAR"] = combined_spectra.ivar[band]

    return extracted_data
# ==================================================
# Per-file worker
# ==================================================

def process_coadd_file(path: Path) -> tuple[int, int]:
    """
    Process a single coadd file and return (saved, skipped).
    saved  = 1 if created a new output file, else 0
    skipped = 1 if skipped this file (existing output or bad name), else 0
    """
    name = path.stem  # e.g. "coadd-main-dark-17681"
    try:
        _, survey, program, healpixid = name.split("-")
    except ValueError:
        logger.warning("Skipping file with unexpected name format: %s", path)
        return (0, 1)

    out_path = TARGET_DIR / f"processed_spectra-{survey}-{program}-{healpixid}.fits.gz"

    if out_path.exists():
        logger.info("Skipping, already exists: %s", path)
        return (0, 1)

    try:
        extracted_data = flux_ivar_exptime_extract(
            str(path),
            float32=True
        )
        extracted_data.write(out_path, format="fits", overwrite=True)
        logger.info("Saved to path: %s", out_path)

        # Delete source coadd only after successful write
        if out_path.exists():
            path.unlink()
        else:
            logger.info(
                "Output %s missing after write; keeping original file %s",
                out_path,
                path,
            )

        return (1, 0)

    except Exception as e:
        logger.exception("Error while processing %s: %s", path, e)
        # do not mark as skipped (it didn't succeed), and don't delete source
        return (0, 0)

# ==================================================
# Main loop
# ==================================================

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(COADD_DIR.glob("coadd-*-*-*.fits"))
    logger.info("Found %d coadd files to process", len(files))

    if not files:
        logger.info("No files found, exiting.")
        return

    # Number of worker processes: from Slurm if available, else default to 4
    try:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    except ValueError:
        n_workers = 4

    if n_workers < 1:
        n_workers = 1

    logger.info("Using %d worker processes", n_workers)

    saved = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_path = {pool.submit(process_coadd_file, p): p for p in files}

        for fut in as_completed(future_to_path):
            path = future_to_path[fut]
            try:
                s, k = fut.result()
                saved += s
                skipped += k
            except Exception:
                logger.exception("Worker crashed on %s", path)

    logger.info("complete! saved %d, skipped %d", saved, skipped)

if __name__ == "__main__":
    main()
