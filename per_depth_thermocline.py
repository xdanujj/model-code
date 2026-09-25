"""
Per-depth sanity check for the UNetOceanV2 (base + thermocline expert)
checkpoint produced by train.py.

Prints base-model RMSE and final (base + expert correction) RMSE side
by side, per depth, plus the natural std for each depth (train split).
The "delta" column should be ~0 for every depth outside the thermocline
band and negative (improved) for depths inside it. If any depth outside
the band shows a nonzero delta, that means you loaded a checkpoint that
went through the optional joint-fine-tune phase (base was unfrozen) --
check the checkpoint's "joint_finetune" flag.

Run on the Jarvis Labs machine, same directory as train.py:
    python3 per_depth_eval.py
"""
import numpy as np
import torch
import h5py

from dataset import OceanTSDataset, compute_norm_stats, date_based_split
from thermocline_expert_model import UNetOceanV2

GLORYS_PATH = "/home/jl_fs/OceanEmbed/merged_glorys_full.h5"
EXPERT_CHECKPOINT = "/home/jl_fs/OceanEmbed/checkpoints/best_model_thermocline_expert_5_12.pt"


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
    ckpt = torch.load(EXPERT_CHECKPOINT, map_location=device, weights_only=False)
    n_days = ckpt["n_days"]
    input_mean, input_std = ckpt["input_mean"], ckpt["input_std"]
    target_mean, target_std = ckpt["target_mean"], ckpt["target_std"]
    lo, hi = ckpt["thermocline_band"]
    base_ch = ckpt.get("base_ch", 20)
    expert_hidden = ckpt.get("expert_hidden", 32)
    expert_dropout = ckpt.get("expert_dropout", 0.1)

    print(f"Loaded expert checkpoint from epoch {ckpt['epoch']}, "
          f"val_RMSE(degC)={ckpt['val_rmse_degC']:.4f}, "
          f"thermocline_band=[{lo},{hi}), joint_finetune={ckpt.get('joint_finetune', False)}")
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
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    n_depths = target.shape[1]
    model = UNetOceanV2(in_channels=7, out_channels=n_depths, base_ch=base_ch,
                         use_cbam=True, use_checkpoint=False,
                         dropout=0.0, bottleneck_dropout=0.0,
                         thermocline_band=(lo, hi),
                         expert_hidden=expert_hidden, expert_dropout=expert_dropout).to(device)

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    sq_err_final = torch.zeros(n_depths, device=device)
    sq_err_base = torch.zeros(n_depths, device=device)
    valid_count = torch.zeros(n_depths, device=device)
    tstd = torch.tensor(target_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred, base_pred = model(x, return_base=True)

            pred_degC = pred.float() * tstd
            base_degC = base_pred.float() * tstd
            y_degC = y.float() * tstd

            sq_err_final += ((pred_degC - y_degC) ** 2 * mask).sum(dim=(0, 2, 3))
            sq_err_base += ((base_degC - y_degC) ** 2 * mask).sum(dim=(0, 2, 3))
            valid_count += mask.sum(dim=(0, 2, 3))

    per_depth_final = torch.sqrt(sq_err_final / (valid_count + 1e-8)).cpu().numpy()
    per_depth_base = torch.sqrt(sq_err_base / (valid_count + 1e-8)).cpu().numpy()

    train_target = target[train_idx]
    natural_std = np.nanstd(train_target, axis=(0, 2, 3))

    print()
    header = f"{'depth':>6} | {'base RMSE':>10} | {'final RMSE':>10} | {'delta':>8} | {'natural_std':>12} | {'final/std':>10}"
    print(header)
    print("-" * len(header))
    for d in range(n_depths):
        delta = per_depth_final[d] - per_depth_base[d]
        ratio = per_depth_final[d] / (natural_std[d] + 1e-8)
        band_flag = "  <- thermocline band" if lo <= d < hi else ""
        warn = "  !! unexpected change outside band" if (not (lo <= d < hi)) and abs(delta) > 1e-4 else ""
        print(f"{d:>6} | {per_depth_base[d]:>10.4f} | {per_depth_final[d]:>10.4f} | "
              f"{delta:>+8.4f} | {natural_std[d]:>12.4f} | {ratio:>10.3f}{band_flag}{warn}")

    overall_final = np.sqrt((sq_err_final.sum() / valid_count.sum()).item())
    overall_base = np.sqrt((sq_err_base.sum() / valid_count.sum()).item())
    band_final = per_depth_final[lo:hi].mean()
    band_base = per_depth_base[lo:hi].mean()
    print()
    print(f"Overall RMSE(degC): base={overall_base:.4f}  final={overall_final:.4f}  "
          f"({overall_final - overall_base:+.4f})")
    print(f"Thermocline band [{lo},{hi}) mean RMSE(degC): base={band_base:.4f}  final={band_final:.4f}  "
          f"({band_final - band_base:+.4f})")


if __name__ == "__main__":
    main()