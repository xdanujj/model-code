"""
Dataset for surface-to-subsurface ocean temperature reconstruction.

--------------------------------------------------------------------
CHANGES FROM THE SPEED-FIX VERSION (mask correctness fix)
--------------------------------------------------------------------
The old code used a single STATIC 2D ocean_mask (H, W), reused for every
one of the 15 depth levels. That's wrong: bathymetry means a grid cell
can be valid ocean at the surface (depth0) but nonexistent at depth14
(the seafloor is shallower than that level there). Per-channel NaN%
in the raw target array climbs steadily with depth (51% at depth0 up
to ~63% at depth14 in this dataset) -- exactly the bathymetry signature.

Combined with the old `np.nan_to_num(y, nan=0.0)` call, every depth
level where the true value was NaN but the flat 2D mask still said
"ocean" was being trained against a fabricated target of 0.0 (close to
the global mean in normalized units) as if it were a real observation.
That's real, wrong gradient signal, worse at deeper levels.

Fix: build the loss mask per-sample, per-depth-level, directly from
`~isnan(target)` at that (t_end) timestep -- shape (15, H, W) instead
of a shared (H, W). NaNs in the input window are still filled with 0
after normalization (needed so the conv net gets finite numbers), but
the TARGET mask now correctly excludes any (depth, pixel) with no data,
whether that's from bathymetry or from a missing day in that channel.

The static 2D `ocean_mask` argument is still accepted (kept for
backward compat with callers / the ARGO test array) but is no longer
used to build the training mask -- it's ANDed into the per-depth mask
only as an extra safety filter, so it can never let load an on-land
static ocean_mask=True cell that the target NaN check would exclude.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Indices (within the 7-channel input) of the "eastward" velocity
# components that must be sign-flipped when the grid is mirrored along
# longitude. Matches the channel order documented in train.py:
# 0:SST, 1:SSS, 2:SSH, 3:U_current, 4:V_current, 5:U_wind, 6:V_wind
FLIP_NEGATE_CHANNELS = (3, 5)


class OceanTSDataset(Dataset):
    def __init__(self, inputs, target, ocean_mask, n_days=26, indices=None,
                 input_mean=None, input_std=None, target_mean=None, target_std=None,
                 augment=False, noise_std=0.03, flip_prob=0.5):
        """
        inputs: np.ndarray (T, 7, H, W)  float32
        target: np.ndarray (T, 15, H, W) float32
        ocean_mask: np.ndarray (H, W)    bool -- static safety filter only,
                 see module docstring. The real per-depth mask is built
                 from isnan(target) at __getitem__ time.
        n_days: number of consecutive days to stack as input channels
        indices: which time indices are valid "end of window" days
        input_mean/std, target_mean/std: precomputed normalization stats
        augment: if True, apply random longitude-flip + Gaussian noise
                 augmentation. Set True for TRAIN ONLY.
        noise_std: std (in already-normalized units) of the Gaussian
                 noise added to the input window when augment=True.
        flip_prob: probability of applying the longitude flip per sample.
        """
        self.inputs = inputs
        self.target = target
        self.static_ocean_mask = ocean_mask.astype(bool)
        self.n_days = n_days
        self.T, self.C, self.H, self.W = inputs.shape
        self.augment = augment
        self.noise_std = noise_std
        self.flip_prob = flip_prob

        if indices is None:
            indices = np.arange(n_days - 1, self.T)
        self.indices = indices

        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std
        # precompute reshaped stats ONCE instead of every __getitem__ call
        if input_mean is not None:
            self._im = input_mean.reshape(self.C, 1, 1, 1).astype(np.float32)
            self._istd = input_std.reshape(self.C, 1, 1, 1).astype(np.float32)
        else:
            self._im = self._istd = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t_end = self.indices[i]
        t_start = t_end - self.n_days + 1

        # NOTE: minimal CPU work on purpose -- see module docstring.
        # No augmentation happens here anymore; it runs on the GPU in
        # train.py's augment_batch() instead.
        window = self.inputs[t_start:t_end + 1]             # (n_days, 7, H, W)
        window = window.transpose(1, 0, 2, 3)                # (7, n_days, H, W) view

        y = self.target[t_end]                                # (15, H, W)

        # per-depth-level valid mask, built BEFORE any NaN filling below.
        # True where this (depth, pixel) has a real observation.
        valid_mask = ~np.isnan(y)                             # (15, H, W) bool
        valid_mask &= self.static_ocean_mask[None, :, :]       # extra safety AND

        if self._im is not None:
            window = (window - self._im) / self._istd
        if self.target_mean is not None:
            y = (y - self.target_mean) / self.target_std

        # optional CPU-side augmentation path, OFF by default -- kept only
        # for debugging/comparison, not used by the current train.py
        if self.augment:
            if np.random.rand() < self.flip_prob:
                window = window[:, :, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
                valid_mask = valid_mask[:, :, ::-1].copy()
                for ch in FLIP_NEGATE_CHANNELS:
                    window[ch] = -window[ch]
            if self.noise_std > 0:
                window = window + np.random.normal(
                    0.0, self.noise_std, size=window.shape).astype(np.float32)

        window = np.ascontiguousarray(np.nan_to_num(window, nan=0.0), dtype=np.float32)
        y = np.ascontiguousarray(np.nan_to_num(y, nan=0.0), dtype=np.float32)
        valid_mask = np.ascontiguousarray(valid_mask, dtype=np.float32)

        return (
            torch.from_numpy(window),
            torch.from_numpy(y),
            torch.from_numpy(valid_mask),   # now (15, H, W), NOT (H, W)
        )


def compute_norm_stats(inputs, target, train_indices):
    """Compute per-channel mean/std using ONLY the training split."""
    train_inputs = inputs[train_indices]
    train_target = target[train_indices]

    input_mean = np.nanmean(train_inputs, axis=(0, 2, 3), keepdims=True)[0]
    input_std = np.nanstd(train_inputs, axis=(0, 2, 3), keepdims=True)[0]
    input_std[input_std < 1e-6] = 1e-6

    target_mean = np.nanmean(train_target, axis=(0, 2, 3), keepdims=True)[0]
    target_std = np.nanstd(train_target, axis=(0, 2, 3), keepdims=True)[0]
    target_std[target_std < 1e-6] = 1e-6

    return input_mean, input_std, target_mean, target_std


def date_based_split(dates, n_days, train_range, val_range):
    """Split by explicit CALENDAR boundaries, not fractions."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    T = len(dates)
    valid_start = n_days - 1
    valid_mask = np.zeros(T, dtype=bool)
    valid_mask[valid_start:] = True

    train_start, train_end = np.datetime64(train_range[0]), np.datetime64(train_range[1])
    val_start, val_end = np.datetime64(val_range[0]), np.datetime64(val_range[1])

    train_mask = valid_mask & (dates >= train_start) & (dates <= train_end)
    val_mask = valid_mask & (dates >= val_start) & (dates <= val_end)

    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]

    if len(train_idx) == 0:
        raise ValueError(f"No training days found in range {train_range}")
    if len(val_idx) == 0:
        raise ValueError(f"No validation days found in range {val_range}")

    return train_idx, val_idx


def all_valid_indices(T, n_days):
    """Every valid 'end of window' index for a dataset used entirely as
    one split (e.g. the ARGO test array)."""
    return np.arange(n_days - 1, T)