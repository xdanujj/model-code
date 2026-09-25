"""
Trains ThermoclineExpert on top of a FROZEN, already-trained base model.

Workflow this script implements:
  1. Load your existing best checkpoint (the plain 15-depth U-Net you've
     already trained) into UNetOceanV2.base.
  2. Freeze every parameter in .base. Only .expert trains.
  3. Train with a depth-weighted masked MSE loss: the thermocline band
     gets a higher weight, so the optimizer spends its effort where the
     error actually lives, without changing what the loss means for the
     other 7 levels (their weight stays 1.0, same as your original loss).
  4. Because the base is frozen AND pred outside [lo, hi) is set to
     base_pred by construction (see model.py), the only way this run
     can make things worse is by overfitting the expert itself - it
     cannot regress any level outside the band. Watch per-depth eval
     (per_depth_eval.py) to confirm.

Optional phase 2 (OFF by default, see JOINT_FINETUNE below): unfreeze
the base for a few epochs at a very low LR. This can squeeze out a bit
more, but it breaks the "other levels literally cannot regress"
guarantee, since the shared backbone would move for all 15 depths.
Only turn it on once phase 1 is stable and you've confirmed the expert
alone already helps.

Run on the Jarvis Labs machine, same directory as dataset.py:
    python3 train.py
"""

import os
import time

import h5py
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from dataset import OceanTSDataset, compute_norm_stats, date_based_split
from thermocline_expert_model import UNetOceanV2, FLIP_NEGATE_CHANNELS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

GLORYS_PATH = "/home/jl_fs/OceanEmbed/merged_glorys_full.h5"
BASE_CHECKPOINT = "/home/jl_fs/OceanEmbed/checkpoints/best_model.pt"          # your existing model
CHECKPOINT_DIR = "/home/jl_fs/OceanEmbed/checkpoints"
EXPERT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model_thermocline_expert_5_12_targeted.pt")

# ---- thermocline band: (5,12), unchanged from the last run - depth 4
# stayed at delta=0.0000 with this band, so no reason to touch it again.
THERMOCLINE_LO, THERMOCLINE_HI = 5, 12

# ---- PER-DEPTH loss weights, replacing the old flat THERMOCLINE_WEIGHT.
# Your last per-depth eval on this band:
#   depth :  5      6      7      8      9      10     11
#   delta : -0.0093 -0.0031 -0.0056 -0.0104 -0.0097 -0.0109 -0.0061
#   base  :  0.7511  0.9689  1.0892  1.0820  0.9716  0.7187  0.4929
# Depths 7, 8, 9 are the ones you actually want down (the ~1.0-1.09
# levels) - they were getting the SAME loss weight as 5, 10, 11, which
# already had smaller errors to begin with. This pushes more of the
# optimizer's effort specifically at 7/8/9, a little extra at 6 (still
# a high absolute error), and pulls 5/10/11 back toward flat weighting
# since they're already comparatively closer to done. Depths outside
# the band are NOT in this dict and always get weight 1.0 - editing
# this dict can never change that.
THERMOCLINE_DEPTH_WEIGHTS = {
    5: 1.3,
    6: 1.9,
    7: 2.4,
    8: 2.4,
    9: 2.2,
    10: 1.4,
    11: 1.1,
}

# unchanged from the last run - hidden=48 clearly helped spread the
# improvement more evenly across the band, keep it.
EXPERT_HIDDEN = 48
EXPERT_DROPOUT = 0.1

JOINT_FINETUNE = False          # keep False until phase-1 alone helps
JOINT_FINETUNE_EPOCHS = 8
JOINT_FINETUNE_LR = 5e-6


def load_glorys_arrays():
    with h5py.File(GLORYS_PATH, "r") as f:
        inputs = f["inputs"][:].astype(np.float32)
        target = f["target"][:].astype(np.float32)
        ocean_mask = f["ocean_mask"][:].astype(bool)
        dates_ns = f["dates"][:]
    dates = dates_ns.astype("datetime64[ns]").astype("datetime64[D]")
    return inputs, target, ocean_mask, dates


def augment_batch(x, y, mask, noise_std=0.03, flip_prob=0.5):
    """Same longitude-flip + input-noise augmentation as your original
    pipeline, GPU-side. Must run BEFORE the model sees x, so the SST/SSH
    the expert reads off x stay aligned with y."""
    B = x.shape[0]
    device = x.device
    do_flip = torch.rand(B, device=device) < flip_prob

    x_flipped = torch.flip(x, dims=[-1]).clone()
    x_flipped[:, FLIP_NEGATE_CHANNELS, :, :, :] *= -1
    x = torch.where(do_flip.view(B, 1, 1, 1, 1), x_flipped, x)

    y = torch.where(do_flip.view(B, 1, 1, 1), torch.flip(y, dims=[-1]), y)
    mask = torch.where(do_flip.view(B, 1, 1, 1), torch.flip(mask, dims=[-1]), mask)

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x, y, mask


def make_depth_weights(n_depths, lo, hi, per_depth_overrides, device):
    """Weight is 1.0 everywhere by default. Only indices inside
    per_depth_overrides get a different weight - indices outside [lo, hi)
    can never appear in that dict in normal use, but we clamp anyway so a
    typo in the dict can't accidentally leak weighting outside the band."""
    w = torch.ones(n_depths, device=device)
    for depth_idx, weight in per_depth_overrides.items():
        if lo <= depth_idx < hi:
            w[depth_idx] = weight
    return w.view(1, n_depths, 1, 1)


def weighted_masked_mse(pred, target, mask, depth_weights):
    pred, target, mask = pred.float(), target.float(), mask.float()
    diff2 = (pred - target) ** 2 * mask * depth_weights
    return diff2.sum() / (mask.sum() + 1e-8)


def masked_rmse(pred, target, mask):
    pred, target, mask = pred.float(), target.float(), mask.float()
    mse = ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return torch.sqrt(mse)


@torch.no_grad()
def evaluate(model, loader, device, target_std_t, n_depths):
    """Returns (final_rmse_degC, base_rmse_degC, per_depth_final_rmse_degC).
    base_rmse_degC lets you confirm the frozen backbone's own output is
    unchanged from your original checkpoint's numbers."""
    model.eval()
    sq_err_final = torch.zeros(n_depths, device=device)
    sq_err_base = torch.zeros(n_depths, device=device)
    valid_count = torch.zeros(n_depths, device=device)

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred, base_pred = model(x, return_base=True)

        pred_degC = pred.float() * target_std_t
        base_degC = base_pred.float() * target_std_t
        y_degC = y.float() * target_std_t

        sq_err_final += ((pred_degC - y_degC) ** 2 * mask).sum(dim=(0, 2, 3))
        sq_err_base += ((base_degC - y_degC) ** 2 * mask).sum(dim=(0, 2, 3))
        valid_count += mask.sum(dim=(0, 2, 3))

    per_depth_final = torch.sqrt(sq_err_final / (valid_count + 1e-8))
    per_depth_base = torch.sqrt(sq_err_base / (valid_count + 1e-8))
    final_rmse = torch.sqrt(sq_err_final.sum() / valid_count.sum()).item()
    base_rmse = torch.sqrt(sq_err_base.sum() / valid_count.sum()).item()
    return final_rmse, base_rmse, per_depth_final.cpu().numpy(), per_depth_base.cpu().numpy()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    torch.backends.cudnn.benchmark = True

    batch_size = 16
    n_epochs = 60
    lr = 2e-4          # expert is small and starts at zero-correction, so
                        # this can be a fairly normal LR even though the
                        # backbone underneath is frozen
    weight_decay = 1e-4
    patience = 12
    grad_clip_norm = 1.0
    num_workers = min(8, os.cpu_count() or 4)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- load base checkpoint first: n_days / norm stats must match it ----
    base_ckpt = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    n_days = base_ckpt["n_days"]
    input_mean, input_std = base_ckpt["input_mean"], base_ckpt["input_std"]
    target_mean, target_std = base_ckpt["target_mean"], base_ckpt["target_std"]
    print(f"Loaded base checkpoint from epoch {base_ckpt['epoch']}, "
          f"val_RMSE(degC) at save time = {base_ckpt['val_rmse_degC']:.4f}, n_days={n_days}")

    glorys_inputs, glorys_target, glorys_mask, glorys_dates = load_glorys_arrays()
    train_idx, val_idx = date_based_split(
        glorys_dates, n_days,
        train_range=("1993-01-01", "2019-12-31"),
        val_range=("2020-01-01", "2021-12-31"),
    )
    print(f"train days: {len(train_idx)}, val days: {len(val_idx)}")

    train_ds = OceanTSDataset(glorys_inputs, glorys_target, glorys_mask, n_days=n_days, indices=train_idx,
                               input_mean=input_mean, input_std=input_std,
                               target_mean=target_mean, target_std=target_std, augment=False)
    val_ds = OceanTSDataset(glorys_inputs, glorys_target, glorys_mask, n_days=n_days, indices=val_idx,
                             input_mean=input_mean, input_std=input_std,
                             target_mean=target_mean, target_std=target_std, augment=False)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True,
                          persistent_workers=(num_workers > 0),
                          prefetch_factor=4 if num_workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    n_depths = glorys_target.shape[1]

    # ---- model: same base_ch/use_cbam as your original best_model.pt ----
    model = UNetOceanV2(in_channels=7, out_channels=n_depths, base_ch=20,
                         use_cbam=True, use_checkpoint=False,
                         dropout=0.20, bottleneck_dropout=0.35,
                         thermocline_band=(THERMOCLINE_LO, THERMOCLINE_HI),
                         expert_hidden=EXPERT_HIDDEN, expert_dropout=EXPERT_DROPOUT).to(device)

    # strict=True on purpose: a silently-partial load is worse than a crash
    # here, since it would mean "frozen base" is actually partly random.
    missing, unexpected = model.load_pretrained_base(base_ckpt["model_state_dict"], strict=True)
    print("Base weights loaded cleanly into model.base (strict match).")

    model.set_base_trainable(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable (expert) params: {n_trainable:,}  |  Frozen (base) params: {n_frozen:,}")

    target_std_t = torch.tensor(target_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    depth_weights = make_depth_weights(n_depths, THERMOCLINE_LO, THERMOCLINE_HI, THERMOCLINE_DEPTH_WEIGHTS, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # sanity: confirm frozen-base RMSE matches what the old checkpoint reported,
    # before spending any epochs training
    _, base_rmse0, _, per_depth_base0 = evaluate(model, val_loader, device, target_std_t, n_depths)
    print(f"Sanity check - frozen base val_RMSE(degC) = {base_rmse0:.4f} "
          f"(checkpoint reported {base_ckpt['val_rmse_degC']:.4f}; should match closely)")

    best_val_rmse_degC = base_rmse0
    epochs_no_improve = 0
    joint_phase = False
    joint_phase_start_epoch = None

    for epoch in range(1, n_epochs + 1):
        if JOINT_FINETUNE and not joint_phase and epochs_no_improve >= patience:
            print(f"Entering optional joint fine-tune phase at epoch {epoch} "
                  f"(unfreezing base, lr={JOINT_FINETUNE_LR:.1e}). "
                  f"NOTE: this removes the 'other depths can't regress' guarantee.")
            model.set_base_trainable(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=JOINT_FINETUNE_LR, weight_decay=weight_decay)
            joint_phase = True
            joint_phase_start_epoch = epoch
            epochs_no_improve = 0
        if joint_phase and (epoch - joint_phase_start_epoch) >= JOINT_FINETUNE_EPOCHS:
            print("Joint fine-tune budget exhausted, stopping.")
            break

        model.train()
        if not joint_phase:
            # base is frozen: keep its Dropout off too, so base_pred stays
            # exactly deterministic (matches your original checkpoint's
            # behavior) and the expert only ever sees clean base features
            model.base.eval()
        train_loss_sum = torch.zeros((), device=device)
        t0 = time.time()

        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x, y, mask = augment_batch(x, y, mask)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(x)
                loss = weighted_masked_mse(pred, y, mask, depth_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
            optimizer.step()

            train_loss_sum += loss.detach() * x.size(0)

        train_loss = (train_loss_sum / len(train_ds)).item()

        val_rmse_degC, base_rmse_degC, per_depth_final, per_depth_base = evaluate(
            model, val_loader, device, target_std_t, n_depths)
        scheduler.step(val_rmse_degC)
        elapsed = time.time() - t0

        band_final = per_depth_final[THERMOCLINE_LO:THERMOCLINE_HI].mean()
        band_base = per_depth_base[THERMOCLINE_LO:THERMOCLINE_HI].mean()
        print(f"Epoch {epoch:3d} | weighted_train_loss={train_loss:.5f} "
              f"| val_RMSE(degC) final={val_rmse_degC:.4f} base={base_rmse_degC:.4f} "
              f"| thermocline band mean: final={band_final:.4f} base={band_base:.4f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
              f"{' [joint]' if joint_phase else ' [expert-only]'}")

        if val_rmse_degC < best_val_rmse_degC:
            best_val_rmse_degC = val_rmse_degC
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse_degC": val_rmse_degC,
                "base_rmse_degC_at_save": base_rmse_degC,
                "input_mean": input_mean, "input_std": input_std,
                "target_mean": target_mean, "target_std": target_std,
                "n_days": n_days,
                "thermocline_band": (THERMOCLINE_LO, THERMOCLINE_HI),
                "thermocline_depth_weights": THERMOCLINE_DEPTH_WEIGHTS,
                "base_ch": 20,
                "expert_hidden": EXPERT_HIDDEN,
                "expert_dropout": EXPERT_DROPOUT,
                "joint_finetune": joint_phase,
            }, EXPERT_CHECKPOINT)
            print(f"  -> saved new best (val_RMSE degC = {val_rmse_degC:.4f}, "
                  f"vs frozen-base-alone {base_rmse0:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not JOINT_FINETUNE:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Training complete. Best val_RMSE(degC): {best_val_rmse_degC:.4f} "
          f"(frozen-base-alone was {base_rmse0:.4f})")


if __name__ == "__main__":
    train()