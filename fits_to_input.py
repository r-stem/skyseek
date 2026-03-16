#!/usr/bin/env python

"""
Transforms the processed .fits shards into training-read .npz shards.

Loads input .fits files and VI .fits file.
Encodes EXPTIME as a sinusoidal channel.
Mixes in a slice of the VI table to allow for training on higher exposure.
Outputs .npz files.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from astropy.io import fits
from astropy.table import Table, vstack

# ==================================================
# Config
# ==================================================

SOURCE_DIR = Path("...")                # Input directory with .fits(.gz) files
TARGET_DIR = Path("...")                # Output directory for .npz files
VI_FILE_PATH = Path("...")              # .fits(.gz) file that contains aggregated VI input data

length = 7781              # spectrum length
base_time = 450            # EXPTIME scaling origin

TARGET_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Logging
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,   # <-- send logs to stdout instead of a file
)

logger = logging.getLogger("preprocess")


# ==================================================
# Worker function
# ==================================================

def process_file(args):
    """
    Process one FITS file → create one NPZ shard.
    Wrapped in a try/except so failure does not kill the pool.
    """
    f, vtable = args  # vtable is a *slice* of global vitable passed to this worker

    try:
        name = f.stem  # e.g. processed_spectra-main-dark-17681
        _, survey, program, healpixid = name.split("-")

        out_path = TARGET_DIR / f"preprocessed-{survey}-{program}-{healpixid}.npz"

        # -------------------
        # Load FITS → Table
        # -------------------
        with fits.open(f) as fts:
            table = Table(fts[1].data)

        table["SURVEY"] = survey

        # -------------------
        # EXPTIME sinusoidal encoding
        # -------------------
        xp = np.asarray(table["COADD_EXPTIME"])         # shape (N,)
        vec = np.arange(length)[None, :]                # shape (1,L)
        scale = np.pi / (base_time * length)
        time_sins = np.sin(vec * xp[:, None] * scale).astype(np.float32)

        table["EXPTIME_CHAN"] = time_sins
        table.remove_column("COADD_EXPTIME")

        # -------------------
        # Mix-in VI rows
        # vtable is unique to each worker
        # -------------------
        if len(vtable) > 0:
            table = vstack([table, vtable], join_type="exact")

        # -------------------
        # Extract arrays
        # -------------------
        targetid = np.asarray(table["TARGETID"])
        flux     = np.asarray(table["FLUX"])
        ivar     = np.asarray(table["IVAR"])
        exptime  = np.asarray(table["EXPTIME_CHAN"])

        # -------------------
        # Save NPZ
        # -------------------
        np.savez(
            out_path,
            targetid=targetid,
            flux=flux,
            ivar=ivar,
            exptime=exptime,
        )

        logger.info(
            f"SUCCESS | {f.name} → {out_path.name} | "
            f"rows={len(table)} (raw + admix={len(vtable)})"
        )

        return len(table)  # count for final summary

    except Exception as e:
        logger.error(f"ERROR | {f.name} | {type(e).__name__}: {e}")
        return 0


# ==================================================
# Main loop
# ==================================================

def main():
    # ==================================================
    # Load VI table
    # ==================================================
    with fits.open(VI_FILE_PATH) as vifile:
        vitable = Table(vifile[1].data)

    vitable = vitable[vitable["SURVEY"] == "sv1"]
    vitable = vitable["TARGETID","SURVEY","FLUX","IVAR","EXPTIME_CHAN"]

    files = sorted(SOURCE_DIR.glob("processed_spectra-*-*-*.fits"))
    total_files = len(files)

    # Number of VI rows to mix into each output shard
    if total_files > 0:
        admix = len(vitable) // total_files + 1
    else:
        admix = 0

    # Determine number of workers from SLURM or local environment
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        workers = int(slurm_cpus)
    else:
        workers = cpu_count()

    logger.info(f"Starting preprocessing with {workers} workers, {total_files} files.")

    # Slice vitable into chunks — each worker gets admix rows uniquely
    vtable_slices = []
    start = 0
    for _ in range(total_files):
        end = min(start + admix, len(vitable))
        vtable_slices.append(vitable[start:end])
        start = end

    # ==================================================
    # Run parallel processing
    # ==================================================
    with Pool(processes=workers) as pool:
        total_rows = sum(pool.map(process_file, zip(files, vtable_slices)))

    logger.info(f"Completed preprocessing.")
    logger.info(f"Total output rows across all NPZ shards: {total_rows}")


if __name__ == "__main__":
    main()
