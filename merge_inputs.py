"""
merge_inputs.py
================
Merges the 7 surface INPUT variables (excludes ARGO, excludes GLORYS target)
across all years into a single aligned array:

    SST, SSS, SSH(SLA), U-current, V-current, U-wind, V-wind

Output: merged_inputs.npz  ->  keys:
    inputs      float32 (7, T, H, W)   channel order = var_names below
    var_names   list[str]              ['sst','sss','ssh','u','v','uwnd','vwnd']
    dates       datetime64[ns] (T,)
    latitude    float64 (H,)
    longitude   float64 (W,)
    ocean_mask  bool (H, W)            True = ocean, based on SST NaN pattern

Run this from the OceanEmbed root (same folder as dataset.py/model.py/train.py),
or edit BASE_DIR below.

ASSUMPTIONS YOU SHOULD VERIFY:
  1. analysed_sst is in Kelvin (standard for OSTIA L4) -> auto-converted to Celsius
     below. If your SST is already in Celsius, set CONVERT_SST_KELVIN = False.
  2. SSS file has a size-1 'depth' dim (depth=0.0) -> we select depth=0 (surface).
  3. currents file has dims (time, longitude, latitude) with 'lat'/'lon' as plain
     coordinate variables, and 'time' often decoded as cftime (e.g.
     cftime.DatetimeJulian) rather than numpy datetime64 due to how OSCAR files
     label their calendar -> both are fixed below.
  4. All variables share the same 101 x 241 grid, lat 5-30N, lon 45-105E. This is
     checked with an assertion; it will raise loudly if a grid doesn't match
     instead of silently misaligning data.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG - edit these if your folder names differ from the screenshot
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "content" / "drive" / "MyDrive" / "SIH 2026 DATASET" / "Fixed Datasets"

DIRS = {
    "sst": DATA_ROOT / "SST_regridded",
    "sss": DATA_ROOT / "fixed_extrapolated_sss",
    "ssh": DATA_ROOT / "SSH_regridded",
    "currents": DATA_ROOT / "currents_regridded",
    "winds": DATA_ROOT / "regridded_fixed_winds",
}

OUT_PATH = BASE_DIR / "merged_inputs.npz"
CONVERT_SST_KELVIN = True  # analysed_sst from OSTIA is normally in Kelvin

YEAR_RE = re.compile(r"(19|20)\d{2}")


def sorted_year_files(dirpath: Path):
    """Return files in a directory sorted by the 4-digit year found in the filename."""
    files = sorted(dirpath.glob("*.nc"))
    def year_of(f):
        m = YEAR_RE.search(f.stem)
        if not m:
            raise ValueError(f"Could not find a year in filename: {f.name}")
        return int(m.group())
    return sorted(files, key=year_of)


def load_concat(dirpath: Path, extract_fn):
    """Open every .nc in dirpath (sorted by year), run extract_fn(ds) -> standardized
    DataArray with dims (time, latitude, longitude), then concat along time."""
    files = sorted_year_files(dirpath)
    if not files:
        raise FileNotFoundError(f"No .nc files found in {dirpath}")
    das = []
    for f in files:
        with xr.open_dataset(f) as ds:
            da = extract_fn(ds).load()  # load into memory now, dataset closes after
        das.append(da)
    full = xr.concat(das, dim="time")
    full = full.sortby("time")
    return full


# ---------------------------------------------------------------------------
# Per-variable extraction functions -> standardized (time, latitude, longitude)
# ---------------------------------------------------------------------------

def extract_sst(ds):
    da = ds["analysed_sst"].astype("float32")
    if CONVERT_SST_KELVIN:
        # crude sanity check on a single scalar rather than the whole array
        sample = float(da.isel(time=0).mean(skipna=True))
        if sample > 100:  # looks like Kelvin
            da = da - 273.15
    return da


def extract_sss(ds):
    da = ds["sos"]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da.astype("float32")


def extract_ssh(ds):
    return ds["sla"].astype("float32")


def _to_datetime_index(time_values):
    """Convert a time coordinate's raw values to a pandas DatetimeIndex,
    handling both numpy datetime64 and cftime object arrays.

    OSCAR currents files often mislabel their time calendar, so xarray
    decodes "time" as cftime objects (e.g. cftime.DatetimeJulian) instead
    of numpy datetime64 -- pd.DatetimeIndex() raises a TypeError on those
    directly. Pull year/month/day straight off each object instead (both
    cftime and plain python datetime expose these attributes), which
    sidesteps the calendar mismatch. This is safe here since the actual
    calendar system only differs from proleptic Gregorian before 1582,
    nowhere near this dataset's 1993-2023 range.
    """
    time_values = np.asarray(time_values)
    if time_values.dtype == object:
        return pd.to_datetime([
            f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in time_values
        ])
    return pd.DatetimeIndex(time_values)


def extract_currents(ds):
    # Raw dims: (time, longitude, latitude); lat/lon are coordinate variables
    # mapped onto those dims but not set as the index -> fix both issues.
    ds = ds.set_index(latitude="lat", longitude="lon")
    ds = ds.assign_coords(time=_to_datetime_index(ds["time"].values))
    u = ds["u"].transpose("time", "latitude", "longitude").astype("float32")
    v = ds["v"].transpose("time", "latitude", "longitude").astype("float32")
    return u, v


def extract_winds(ds):
    uwnd = ds["uwnd"].astype("float32")
    vwnd = ds["vwnd"].astype("float32")
    return uwnd, vwnd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading SST ...")
    sst = load_concat(DIRS["sst"], extract_sst)

    print("Loading SSS ...")
    sss = load_concat(DIRS["sss"], extract_sss)

    print("Loading SSH ...")
    ssh = load_concat(DIRS["ssh"], extract_ssh)

    print("Loading currents (u, v) ...")
    u_list, v_list = [], []
    for f in sorted_year_files(DIRS["currents"]):
        with xr.open_dataset(f) as ds:
            u, v = extract_currents(ds)
            u_list.append(u.load())
            v_list.append(v.load())
    u_full = xr.concat(u_list, dim="time").sortby("time")
    v_full = xr.concat(v_list, dim="time").sortby("time")

    print("Loading winds (uwnd, vwnd) ...")
    uw_list, vw_list = [], []
    for f in sorted_year_files(DIRS["winds"]):
        with xr.open_dataset(f) as ds:
            uw, vw = extract_winds(ds)
            uw_list.append(uw.load())
            vw_list.append(vw.load())
    uwnd_full = xr.concat(uw_list, dim="time").sortby("time")
    vwnd_full = xr.concat(vw_list, dim="time").sortby("time")

    variables = {
        "sst": sst,
        "sss": sss,
        "ssh": ssh,
        "u": u_full,
        "v": v_full,
        "uwnd": uwnd_full,
        "vwnd": vwnd_full,
    }

    # ---- align everyone to the intersection of available dates ----
    print("Aligning dates across all 7 variables ...")
    common_dates = None
    for name, da in variables.items():
        dates = pd.DatetimeIndex(da["time"].values).normalize()
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    common_dates = common_dates.sort_values()
    print(f"  Common date range: {common_dates.min()} to {common_dates.max()} "
          f"({len(common_dates)} days)")

    # ---- check grids match, then reindex each var to the common dates ----
    ref_lat = variables["sst"]["latitude"].values
    ref_lon = variables["sst"]["longitude"].values

    stacked = []
    var_names = ["sst", "sss", "ssh", "u", "v", "uwnd", "vwnd"]
    for name in var_names:
        da = variables[name]
        assert da.sizes["latitude"] == len(ref_lat) and da.sizes["longitude"] == len(ref_lon), \
            f"{name} grid shape mismatch: got {da.sizes}, expected ({len(ref_lat)},{len(ref_lon)})"
        np.testing.assert_allclose(da["latitude"].values, ref_lat, atol=1e-3,
                                    err_msg=f"{name} latitude grid does not match SST grid")
        np.testing.assert_allclose(da["longitude"].values, ref_lon, atol=1e-3,
                                    err_msg=f"{name} longitude grid does not match SST grid")

        da = da.assign_coords(time=pd.DatetimeIndex(da["time"].values).normalize())
        da = da.sel(time=common_dates)
        stacked.append(da.values.astype("float32"))

    inputs = np.stack(stacked, axis=0)  # (7, T, H, W)
    print(f"Final inputs array shape: {inputs.shape}")

    ocean_mask = ~np.isnan(stacked[0][0])  # from first SST timestep

    np.savez_compressed(
        OUT_PATH,
        inputs=inputs,
        var_names=np.array(var_names),
        dates=common_dates.values,
        latitude=ref_lat,
        longitude=ref_lon,
        ocean_mask=ocean_mask,
    )
    print(f"Saved -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()