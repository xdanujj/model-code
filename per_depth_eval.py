"""
Per-depth-level sanity check.

Computes val_RMSE(degC) SEPARATELY for each of the 15 depth levels,
using the current best checkpoint, and compares each depth's RMSE
against that depth's own natural std (computed on the training split).

Why this matters: your training loop reports ONE combined RMSE across
all 15 depths. That number can hide very different behavior per depth
-- if a deep level's RMSE is close to its own natural std, the model
has learned almost nothing there beyond predicting the mean; that's a
structural ceiling (missing information), not something more
regularization or tuning can fix.

Run this on the Jarvis Labs machine, same directory as train.py:
    python3 per_depth_eval.py
"""
import numpy as np
import torch
import h5py

from dataset import OceanTSDataset, compute_norm_stats, date_based_split
from model import UNetOcean3D

GLORYS_PATH = "/home/jl_fs/OceanEmbed/merged_glorys_full.h5"
CHECKPOINT_PATH = "/home/jl_fs/OceanEmbed/checkpoints/best_model.pt"


def load_glorys_arrays():
    with h5py.File(GLORYS_PATH, "r") as f:
        inputs = f["inputs"][:].astype(np.float32)
        target = f["target"][:].astype(np.float32)
        ocean_mask = f["ocean_mask"][:].astype(bool)
        dates_ns = f["dates"][:]
    dates = dates_ns.astype("datetime64[ns]").astype("datetime64[D]")
    return inputs, target, ocean_mask, dates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    n_days = ckpt["n_days"]
    input_mean, input_std = ckpt["input_mean"], ckpt["input_std"]
    target_mean, target_std = ckpt["target_mean"], ckpt["target_std"]

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val_RMSE(degC) at save time = {ckpt['val_rmse_degC']:.4f}")
    print(f"n_days used: {n_days}")

    inputs, target, ocean_mask, dates = load_glorys_arrays()
    train_idx, val_idx = date_based_split(
        dates, n_days,
        train_range=("1993-01-01", "2019-12-31"),
        val_range=("2020-01-01", "2021-12-31"),
    )

    val_ds = OceanTSDataset(inputs, target, ocean_mask, n_days=n_days, indices=val_idx,
                             input_mean=input_mean, input_std=input_std,
                             target_mean=target_mean, target_std=target_std,
                             augment=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False,
                                              num_workers=4)

    model = UNetOcean3D(in_channels=7, out_channels=15, base_ch=20,
                         n_days=n_days, use_cbam=True, use_checkpoint=False,
                         dropout=0.0, bottleneck_dropout=0.0).to(device)
    # strip potential torch.compile prefix from checkpoint keys
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    n_depths = target.shape[1]
    sq_err_sum = torch.zeros(n_depths, device=device)
    valid_count = torch.zeros(n_depths, device=device)

    tstd = torch.tensor(target_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    with torch.no_grad():
        for x, y, mask in val_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)  # (B, 15, H, W)

            pred = model(x)
            pred_degC = pred.float() * tstd
            y_degC = y.float() * tstd

            diff2 = (pred_degC - y_degC) ** 2 * mask
            sq_err_sum += diff2.sum(dim=(0, 2, 3))
            valid_count += mask.sum(dim=(0, 2, 3))

    per_depth_rmse = torch.sqrt(sq_err_sum / (valid_count + 1e-8)).cpu().numpy()

    # natural std per depth, in degC, on the TRAIN split (matches what
    # compute_norm_stats already computed, just displayed per-depth here)
    train_target = target[train_idx]
    natural_std = np.nanstd(train_target, axis=(0, 2, 3))

    print()
    print(f"{'depth':>6} | {'RMSE(degC)':>11} | {'natural_std':>12} | {'RMSE/std ratio':>14}")
    print("-" * 52)
    for d in range(n_depths):
        ratio = per_depth_rmse[d] / (natural_std[d] + 1e-8)
        flag = "  <-- near ceiling (model ~= predicting mean)" if ratio > 0.7 else ""
        print(f"{d:>6} | {per_depth_rmse[d]:>11.4f} | {natural_std[d]:>12.4f} | {ratio:>14.3f}{flag}")

    print()
    print(f"Overall combined RMSE (matches training loop's calc): "
          f"{np.sqrt((sq_err_sum.sum() / valid_count.sum()).item()):.4f}")


if __name__ == "__main__":
    main()