"""
merge_glorys_with_inputs.py
============================
Combines the 7 surface INPUT variables (merged_inputs.npz) with the GLORYS
subsurface TARGET (merged_glorys.h5) into a single date-aligned array,
matching what train.py's load_glorys_arrays() expects:

    inputs:     (T, 7, H, W)   float32  -- SST, SSS, SSH, U, V, uwnd, vwnd
    target:     (T, 15, H, W)  float32  -- GLORYS temperature, 15 std depths
    ocean_mask: (H, W)         bool     -- valid in ALL 7 inputs AND all 15
                                            target depths (single reference
                                            day -- land/ocean doesn't change
                                            day to day, so scanning every
                                            timestep would just be expensive
                                            for the same answer)
    dates:      (T,)           datetime64[D]

Output is HDF5, not .npz -- combined raw size is ~23.5 GB (7.5 GB inputs +
16 GB target), too large to comfortably round-trip through savez_compressed.
NOTE: train.py's load_glorys_arrays() will need updating to read this via
h5py instead of np.load() -- see the printed reminder at the end.

Run from the folder containing both merged_inputs.npz and merged_glorys.h5,
or edit the paths below.
"""

from pathlib import Path

import h5py
import numpy as np

INPUTS_PATH = Path("merged_inputs.npz")
GLORYS_PATH = Path("merged_glorys.h5")
OUT_PATH = Path("merged_glorys_full.h5")


def main():
    print("Loading merged_inputs.npz ...")
    inp = np.load(INPUTS_PATH, allow_pickle=True)
    inputs = inp["inputs"]                                  # (7, T_in, H, W)
    input_dates = np.asarray(inp["dates"], dtype="datetime64[D]")
    var_names = [str(v) for v in inp["var_names"]]
    latitude = inp["latitude"]
    longitude = inp["longitude"]
    print(f"  inputs shape (7,T,H,W): {inputs.shape}, vars: {var_names}")

    print("Loading merged_glorys.h5 ...")
    with h5py.File(GLORYS_PATH, "r") as f:
        thetao = f["thetao"][:]                             # (T_g, 15, H, W)
        depth = f["depth"][:]
        glorys_dates_raw = f["dates"][:]                     # int64 ns
    glorys_dates = glorys_dates_raw.astype("datetime64[ns]").astype("datetime64[D]")
    print(f"  target shape (T,15,H,W): {thetao.shape}")

    # ---- align on common dates ----
    common_dates = np.intersect1d(input_dates, glorys_dates)
    print(f"  common dates: {len(common_dates)} "
          f"({common_dates.min()} to {common_dates.max()})")
    if len(common_dates) != len(input_dates) or len(common_dates) != len(glorys_dates):
        print(f"  NOTE: input_dates had {len(input_dates)}, glorys_dates had "
              f"{len(glorys_dates)} -- trimming both to the {len(common_dates)} "
              f"dates common to both. Investigate if this drop is larger than "
              f"a handful of days.")

    in_idx = np.searchsorted(input_dates, common_dates)
    gl_idx = np.searchsorted(glorys_dates, common_dates)

    inputs = inputs.transpose(1, 0, 2, 3)                   # (7,T,H,W) -> (T,7,H,W)
    inputs = inputs[in_idx]
    target = thetao[gl_idx]
    dates_out = common_dates

    # ---- combined ocean mask, from a single reference day ----
    print("Computing combined ocean mask ...")
    ref = 0
    input_valid = ~np.isnan(inputs[ref]).any(axis=0)        # (H, W)
    target_valid = ~np.isnan(target[ref]).any(axis=0)       # (H, W)
    ocean_mask = input_valid & target_valid
    print(f"  ocean fraction: {ocean_mask.mean():.3f}")

    # ---- write ----
    print(f"Writing {OUT_PATH} ...")
    with h5py.File(OUT_PATH, "w") as f:
        f.create_dataset("inputs", data=inputs, dtype="float32",
                          chunks=(1, 7, inputs.shape[2], inputs.shape[3]),
                          compression="gzip", compression_opts=4)
        f.create_dataset("target", data=target, dtype="float32",
                          chunks=(1, 15, target.shape[2], target.shape[3]),
                          compression="gzip", compression_opts=4)
        f.create_dataset("ocean_mask", data=ocean_mask, dtype="bool")
        f.create_dataset("dates", data=dates_out.astype("datetime64[ns]").astype("int64"))
        f.create_dataset("depth", data=depth)
        f.create_dataset("latitude", data=latitude)
        f.create_dataset("longitude", data=longitude)
        f.attrs["var_names"] = var_names

    print(f"Done. inputs {inputs.shape}, target {target.shape}, "
          f"ocean_mask {ocean_mask.mean():.1%} ocean, "
          f"file size {OUT_PATH.stat().st_size / 1e9:.2f} GB")
    print("\nREMINDER: train.py's load_glorys_arrays() currently does "
          "np.load('...npz') -- update it to h5py.File(OUT_PATH, 'r') and "
          "read inputs/target/ocean_mask/dates as datasets, not npz keys.")


if __name__ == "__main__":
    main()