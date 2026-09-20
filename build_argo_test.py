"""
Build merged_argo_test.npz -- the independent ARGO-backed test set for
2022-2023, following the design agreed on:

  - Inputs: reuse the REAL daily surface fields from merged_glorys_full.h5
    (not ARGO's monthly cadence) -- train/test must see the same input
    distribution, only the target source differs.
  - Target: real ARGO temperature, monthly, matched to the NEAREST calendar
    day in the daily input axis (not averaged into a monthly window).
  - Depth: ARGO's native 19 levels (5-1000m) interpolated onto your 15
    standard depths. Levels shallower than ARGO's minimum (5m) are
    edge-clamped rather than extrapolated (done here by clipping the
    target depths into ARGO's native range before interpolating).
  - Most rows in the resulting daily target array have no real ARGO match
    (only ~24 months of data spread across 730 days) -- those rows are
    left as NaN, and `test_indices` records only the rows with a genuine
    ARGO sample. train.py must use `test_indices` directly instead of
    all_valid_indices(), or it will silently evaluate against
    NaN->0 placeholder targets on ~700+ days that were never real.

BEFORE RUNNING:
  1. Fill in STANDARD_DEPTHS below with your actual 15 depth values
     (in meters, in the same order as the GLORYS target's depth axis).
     This script will refuse to run with the placeholder value.
  2. Fill in ARGO_RAW_PATH and GLORYS_H5_PATH.
  3. Check the auto-detected variable/coord names printed at the top of
     the run -- if ARGO's depth or lat/lon coord isn't named what's
     auto-detected, set ARGO_TEMP_VAR / ARGO_DEPTH_COORD manually.
"""

import os
import numpy as np
import xarray as xr
import h5py

# ----------------------- CONFIG (edit these) -----------------------
GLORYS_H5_PATH = "merged_glorys_full.h5"
ARGO_RAW_PATH = "argo/argo_regridded_025deg.nc"   # confirmed: 241 monthly steps
                                                    # 2004-01-15..2024-01-15, ZAX=19 depths,
                                                    # lat 5-30 lon 45-105 @0.25deg -- same grid
                                                    # as GLORYS, no regridding needed
OUT_PATH = "merged_argo_test.npz"

TEST_START = "2022-01-01"
TEST_END = "2023-12-31"

# how many extra days of GLORYS inputs to pull in *before* TEST_START so
# that windows near the start of the test period still have enough prior
# days for n_days windowing (must be >= the largest n_days you plan to try)
BUFFER_DAYS = 20

ARGO_TEMP_VAR = "TEMP"     # confirmed from the uploaded file
ARGO_DEPTH_COORD = "ZAX"   # confirmed from the uploaded file

# Confirmed from the SIH problem statement. 14 of these 15 are exact matches
# to ARGO's own 19 native depths (5,10,20,30,50,75,100,125,150,200,300,500,
# 700,1000) -- only 0m falls outside ARGO's range (min=5m) and gets
# edge-clamped to the 5m value.
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
# ---------------------------------------------------------------------


EXCLUDE = {"time", "lat", "lon", "latitude", "longitude", "depth", "pressure",
           "nav_lat", "nav_lon", "crs", "bnds", "time_bnds"}


def auto_temp_var(ds):
    candidates = [v for v in ds.data_vars if v.lower() not in EXCLUDE]
    if len(candidates) == 0:
        raise ValueError(f"No data variable found in ARGO file. Available: {list(ds.data_vars)}")
    if len(candidates) > 1:
        print(f"  [warn] multiple data vars found {candidates}, using first: {candidates[0]}")
    return candidates[0]


def auto_depth_coord(ds):
    for cand in ("depth", "deptht", "pressure", "lev", "z"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    raise ValueError(f"Could not auto-detect a depth coordinate. Coords were: {list(ds.coords)}")


def main():
    if STANDARD_DEPTHS is None:
        raise ValueError(
            "STANDARD_DEPTHS is not set. Fill in your actual 15 standard depth "
            "values (meters) at the top of this script before running."
        )
    standard_depths = np.asarray(STANDARD_DEPTHS, dtype=np.float64)
    if len(standard_depths) != 15:
        raise ValueError(f"Expected 15 standard depths, got {len(standard_depths)}.")

    # ---------------- load GLORYS inputs + dates ----------------
    print(f"Loading GLORYS inputs from {GLORYS_H5_PATH} ...")
    with h5py.File(GLORYS_H5_PATH, "r") as f:
        glorys_inputs = f["inputs"][:].astype(np.float32)         # (T, 7, H, W)
        glorys_dates_ns = f["dates"][:]
    glorys_dates = glorys_dates_ns.astype("datetime64[ns]").astype("datetime64[D]")

    buffered_start = np.datetime64(TEST_START) - np.timedelta64(BUFFER_DAYS, "D")
    test_end = np.datetime64(TEST_END)

    slice_mask = (glorys_dates >= buffered_start) & (glorys_dates <= test_end)
    if not slice_mask.any():
        raise ValueError(
            f"No GLORYS dates found in [{buffered_start}, {test_end}] -- "
            f"check GLORYS_H5_PATH covers the test period."
        )
    inputs_slice = glorys_inputs[slice_mask]     # (T_local, 7, H, W)
    dates_slice = glorys_dates[slice_mask]       # (T_local,)
    print(f"  sliced GLORYS inputs: {inputs_slice.shape}, "
          f"{dates_slice[0]} -> {dates_slice[-1]} ({len(dates_slice)} days, "
          f"includes {BUFFER_DAYS}-day buffer before {TEST_START})")

    H, W = inputs_slice.shape[-2], inputs_slice.shape[-1]

    # ---------------- load + interpolate ARGO ----------------
    print(f"\nLoading ARGO raw data from {ARGO_RAW_PATH} ...")
    argo_ds = xr.open_dataset(ARGO_RAW_PATH)

    temp_var = ARGO_TEMP_VAR or auto_temp_var(argo_ds)
    depth_coord = ARGO_DEPTH_COORD or auto_depth_coord(argo_ds)
    print(f"  using temperature variable: '{temp_var}', depth coord: '{depth_coord}'")
    print(f"  ARGO native depths ({len(argo_ds[depth_coord])} levels): "
          f"{argo_ds[depth_coord].values}")

    argo_da = argo_ds[temp_var]
    if "time" not in argo_da.dims:
        raise ValueError(f"'{temp_var}' has no 'time' dimension. Dims: {argo_da.dims}")

    argo_da = argo_da.sel(time=slice(TEST_START, TEST_END))
    if argo_da.sizes.get("time", 0) == 0:
        raise ValueError(f"No ARGO samples found in [{TEST_START}, {TEST_END}].")
    argo_times = argo_da["time"].values.astype("datetime64[D]")
    print(f"  ARGO samples in test window: {len(argo_times)} months, "
          f"{argo_times[0]} -> {argo_times[-1]}")

    # edge-clamp: clip target depths into ARGO's native range so interp()
    # returns the boundary value instead of NaN for out-of-range depths
    argo_depth_vals = argo_ds[depth_coord].values.astype(np.float64)
    clamped_depths = np.clip(standard_depths, argo_depth_vals.min(), argo_depth_vals.max())
    n_clamped = int(np.sum(clamped_depths != standard_depths))
    if n_clamped:
        print(f"  [note] {n_clamped} standard depth(s) fall outside ARGO's native "
              f"range [{argo_depth_vals.min()}, {argo_depth_vals.max()}] -- "
              f"edge-clamped rather than extrapolated.")

    argo_interp = argo_da.interp({depth_coord: clamped_depths}, method="linear")
    argo_interp = argo_interp.transpose("time", depth_coord, ...).values.astype(np.float32)
    print(f"  interpolated ARGO target shape: {argo_interp.shape}  (expect (n_months, 15, {H}, {W}))")

    if argo_interp.shape[-2:] != (H, W):
        raise ValueError(
            f"ARGO grid {argo_interp.shape[-2:]} doesn't match GLORYS grid ({H}, {W}). "
            f"Regridding is needed before this script can pair them."
        )

    # ---------------- nearest-day matching ----------------
    target_full = np.full((len(dates_slice), 15, H, W), np.nan, dtype=np.float32)
    matched_indices = []
    max_gap_days = []

    for i, argo_date in enumerate(argo_times):
        pos = np.searchsorted(dates_slice, argo_date)
        # searchsorted gives insertion point; check neighbors for true nearest
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(dates_slice)]
        best = min(candidates, key=lambda p: abs((dates_slice[p] - argo_date) / np.timedelta64(1, "D")))
        gap = abs((dates_slice[best] - argo_date) / np.timedelta64(1, "D"))
        max_gap_days.append(gap)

        if best in matched_indices:
            print(f"  [warn] ARGO month {argo_date} matched to already-used day index {best} "
                  f"(nearest-day collision) -- this sample will overwrite the previous one.")

        target_full[best] = argo_interp[i]
        matched_indices.append(best)

    matched_indices = np.array(sorted(set(matched_indices)))
    print(f"\nMatched {len(matched_indices)} unique days out of {len(argo_times)} ARGO months "
          f"(max nearest-day gap: {max(max_gap_days):.1f} days, "
          f"mean: {np.mean(max_gap_days):.1f} days)")

    # keep only indices with enough preceding history for windowing
    min_valid_idx = BUFFER_DAYS  # conservative: BUFFER_DAYS was chosen >= max n_days you'll try
    test_indices = matched_indices[matched_indices >= min_valid_idx]
    if len(test_indices) < len(matched_indices):
        print(f"  [warn] dropped {len(matched_indices) - len(test_indices)} matched day(s) "
              f"too close to the start of the buffered window to have a full n_days history.")

    # ---------------- ocean mask: ARGO's own valid-data footprint ----------------
    ocean_mask = np.any(~np.isnan(argo_interp), axis=(0, 1))  # (H, W)
    print(f"ARGO ocean_mask valid pixels: {ocean_mask.sum()} / {ocean_mask.size} "
          f"({100 * ocean_mask.mean():.1f}%)")

    # ---------------- save ----------------
    np.savez_compressed(
        OUT_PATH,
        inputs=inputs_slice,
        target=target_full,
        ocean_mask=ocean_mask,
        test_indices=test_indices,
        dates=dates_slice.astype(str),
    )
    print(f"\nSaved -> {OUT_PATH}")
    print(f"  inputs: {inputs_slice.shape}, target: {target_full.shape}, "
          f"test_indices: {test_indices.shape} (only these rows have real ARGO data)")


if __name__ == "__main__":
    main()