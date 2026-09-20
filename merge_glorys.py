"""
merge_glorys.py  (storage-efficient version)
=============================================
Merges the GLORYS temperature files (glorys_temp_nio_YYYY.nc) across years into
a single aligned file: the TARGET for training (this is NOT the ARGO test set).

WHY THIS VERSION IS DIFFERENT:
The original approach (build a full uncompressed on-disk memmap, THEN write a
compressed .npz from it) needs roughly 2x the final file size in free disk space
at once, because both copies exist simultaneously for part of the run. On a
storage-constrained instance that fails partway through - wasting the compute
time you already paid for.

This version writes directly into a single COMPRESSED HDF5 file, one year at a
time, with no separate uncompressed shadow copy. Peak extra disk usage is just
the size of the (still-growing) output file itself, nothing more.

Output: merged_glorys.h5  ->  datasets:
    thetao      float32 (T, D, H, W)   D = depth levels (see TARGET_DEPTHS below)
    depth       float32 (D,)
    dates       int64 (T,)             pandas datetime64[ns] view (nanoseconds since epoch)
    latitude    float64 (H,)
    longitude   float64 (W,)

To read it later:
    import h5py
    with h5py.File("merged_glorys.h5", "r") as f:
        thetao = f["thetao"][:]        # or f["thetao"][t0:t1] to read a slice
                                        # WITHOUT loading the whole file into RAM
        dates = pd.to_datetime(f["dates"][:])

DEPTH SELECTION:
    Set to the exact 15 standard depths specified in the SIH26066 problem
    statement (PS ID 26066): 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200,
    300, 500, 700, 1000 meters. GLORYS's own depth levels don't line up with
    these, so each target is obtained via LINEAR INTERPOLATION along the
    depth axis - giving the exact requested value, not a nearest-neighbor
    substitute. The two targets outside GLORYS's actual depth coverage
    (0m is above the shallowest GLORYS level 0.494m; 1000m is below the
    deepest level 902.339m) can't be interpolated - true extrapolation past
    the measured depth range isn't physically reliable for ocean temperature,
    so those two are clamped to GLORYS's boundary values (0.494m and
    902.339m) instead. Every other depth in the list gets its exact value.
    Set TARGET_DEPTHS = None to keep all 35 raw GLORYS depths instead (needs
    ~2.3x more disk space).
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h5py

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "content" / "drive" / "MyDrive" / "SIH 2026 DATASET" / "Fixed Datasets"
GLORYS_DIR = DATA_ROOT / "glorys_regridded"

OUT_PATH = BASE_DIR / "merged_glorys.h5"

# Official SIH26066 standard depths (meters). Set to None to keep all 35
# depths instead (uses ~2.3x more disk space).
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

COMPRESSION = "gzip"
COMPRESSION_LEVEL = 4  # 0-9; higher = smaller file but slower. 4 is a fair trade-off.

YEAR_RE = re.compile(r"(19|20)\d{2}")


def sorted_year_files(dirpath: Path):
    files = sorted(dirpath.glob("*.nc"))
    def year_of(f):
        m = YEAR_RE.search(f.stem)
        if not m:
            raise ValueError(f"Could not find a year in filename: {f.name}")
        return int(m.group())
    return sorted(files, key=year_of)


def main():
    files = sorted_year_files(GLORYS_DIR)
    if not files:
        raise FileNotFoundError(f"No .nc files found in {GLORYS_DIR}")
    print(f"Found {len(files)} GLORYS year files.")

    # ---- inspect the first file to fix grid/depth shapes ----
    with xr.open_dataset(files[0]) as ds0:
        lat = ds0["latitude"].values
        lon = ds0["longitude"].values
        full_depth = ds0["depth"].values

        if TARGET_DEPTHS is None:
            depth_out = full_depth
        else:
            lo, hi = full_depth.min(), full_depth.max()
            depth_out = np.clip(np.asarray(TARGET_DEPTHS, dtype="float64"), lo, hi)
            depth_out = np.unique(depth_out)  # guards against two targets clipping to the same edge
            clipped = [(t, d) for t, d in zip(TARGET_DEPTHS, np.clip(TARGET_DEPTHS, lo, hi)) if t != d]
            print(f"Requested {len(TARGET_DEPTHS)} standard depths, GLORYS range is "
                  f"[{lo:.3f}, {hi:.3f}] m.")
            if clipped:
                print(f"  Out of range, clamped to GLORYS boundary: "
                      + ", ".join(f"{t}m -> {d:.3f}m" for t, d in clipped))
            print(f"  All other {len(TARGET_DEPTHS) - len(clipped)} depths will be exact "
                  f"via linear interpolation along the depth axis.")

    H, W, D = len(lat), len(lon), len(depth_out)

    # ---- total days across all files (for the output dataset shape) ----
    day_counts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            day_counts.append(ds.sizes["time"])
    T_total = sum(day_counts)
    print(f"Total days across all years: {T_total}  |  output shape will be "
          f"({T_total}, {D}, {H}, {W})")

    # ---- rough disk-space sanity check BEFORE writing anything ----
    uncompressed_bytes = T_total * D * H * W * 4
    # gzip level 4 on this kind of smooth geophysical float data typically gets
    # 30-50% reduction; assume a conservative 30% to avoid under-promising
    est_bytes = uncompressed_bytes * 0.7
    free_bytes = shutil.disk_usage(BASE_DIR).free
    print(f"Estimated output size: ~{est_bytes / 1e9:.1f} GB "
          f"(uncompressed would be {uncompressed_bytes / 1e9:.1f} GB)")
    print(f"Free disk space available: {free_bytes / 1e9:.1f} GB")
    if est_bytes > free_bytes * 0.9:
        raise SystemExit(
            "ABORTING before writing anything: estimated output size is too "
            "close to your free disk space. Free up more space or reduce "
            "TARGET_DEPTHS further before re-running."
        )

    # ---- create the HDF5 file and datasets up front ----
    with h5py.File(OUT_PATH, "w") as hf:
        thetao_ds = hf.create_dataset(
            "thetao", shape=(T_total, D, H, W), dtype="float32",
            chunks=(1, D, H, W),  # one day at a time, matches how it'll be read
            compression=COMPRESSION, compression_opts=COMPRESSION_LEVEL,
        )
        hf.create_dataset("depth", data=depth_out.astype("float32"))
        hf.create_dataset("latitude", data=lat)
        hf.create_dataset("longitude", data=lon)

        all_dates = []
        t_cursor = 0
        for f, n_days in zip(files, day_counts):
            print(f"Processing {f.name} ({n_days} days) ...")
            with xr.open_dataset(f) as ds:
                da = ds["thetao"]
                if TARGET_DEPTHS is not None:
                    # linear interpolation along depth -> exact values at each
                    # target depth, except the two boundary-clamped ones above
                    da = da.interp(depth=depth_out, method="linear")
                da = da.transpose("time", "depth", "latitude", "longitude")
                arr = da.values.astype("float32")  # only this one year in RAM
                thetao_ds[t_cursor:t_cursor + n_days] = arr  # streams straight to disk, compressed
                all_dates.append(pd.DatetimeIndex(ds["time"].values))
            t_cursor += n_days

        dates = all_dates[0].append(all_dates[1:]) if len(all_dates) > 1 else all_dates[0]
        dates = pd.DatetimeIndex(dates).normalize()

        if not dates.is_monotonic_increasing:
            order = np.argsort(dates.values)
            # re-order in place, chunk by chunk, to avoid loading everything at once
            reordered = thetao_ds[:][order]
            thetao_ds[:] = reordered
            del reordered
            dates = dates[order]

        dup = dates.duplicated()
        if dup.any():
            print(f"WARNING: {dup.sum()} duplicate dates found across year files - "
                  f"check for overlapping files.")

        hf.create_dataset("dates", data=dates.values.astype("int64"))

    print(f"Saved -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
