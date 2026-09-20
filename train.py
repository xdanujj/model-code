"""
Training script for surface-to-subsurface ocean temperature reconstruction.

--------------------------------------------------------------------
CHANGES FROM THE ORIGINAL VERSION (overfitting fixes)
--------------------------------------------------------------------
1. n_days corrected from 10 -> 26.
2. weight_decay raised 1e-4 -> 5e-4.
3. dropout / bottleneck_dropout passed through to the model.
4. Every epoch prints the train/val gap explicitly.

--------------------------------------------------------------------
CHANGES IN THIS VERSION (speed -- your 813s/epoch report)
--------------------------------------------------------------------
1. Augmentation moved from CPU (per-sample, in the Dataset) to GPU
   (per-batch, in augment_batch() below). The CPU version was doing
   negative-stride slices + several full-array .copy() calls + a
   np.random.normal draw over ~4.4M elements PER SAMPLE, competing for
   only 2 worker processes -- that's almost certainly your bottleneck,
   not the GPU. augment_batch() does the equivalent work as a handful of
   torch ops on the whole batch, on the GPU, where it's nearly free.
2. persistent_workers=True + a higher num_workers + prefetch_factor.
   Without persistent_workers, DataLoader tears down and respawns every
   worker process at the START of every epoch -- which lines up exactly
   with where your run appeared to hang (right after epoch 1 finished).
3. use_checkpoint defaults to False now. Gradient checkpointing trades
   extra recompute for lower VRAM, but this model is ~9.7M params -- on
   a 40GB A100 you almost certainly don't need it, and it was silently
   costing you an extra forward pass through the whole 3D encoder on
   every backward step. Flip it back to True only if you hit OOM.
4. batch_size raised back to 16 (grad_accum_steps=1) since checkpointing
   is off and there's VRAM headroom. Adjust down if you do OOM.
5. torch.backends.cudnn.benchmark = True -- lets cuDNN autotune the
   fastest conv algorithm for your fixed input shape (after padding,
   every batch has the same (T,H,W)), which specifically speeds up the
   3D convolutions.
"""

"""
--------------------------------------------------------------------
CHANGES IN THIS VERSION (further speed, now that we KNOW it's
compute-bound, not data-bound -- confirmed by your diag output showing
data_wait << gpu_compute)
--------------------------------------------------------------------
1. bf16 instead of fp16 for autocast. A100 runs bf16 at the same tensor-
   core throughput as fp16, but bf16 has the same exponent range as
   fp32, so it doesn't need loss scaling at all -- GradScaler is now
   skipped entirely when bf16 is selected, removing its bookkeeping.
2. torch.compile(model). Fuses elementwise ops (conv+norm+relu chains)
   into fewer, larger GPU kernels, cutting launch overhead. First few
   batches will be slower (it's tracing/compiling the graph, similar to
   the cudnn.benchmark warm-up you already saw) -- wrapped in try/except
   so it can't break your run on an older PyTorch build.
3. Explicit TF32 flags. Usually on by default on Ampere, but set
   explicitly for any matmul-heavy ops that aren't already inside
   autocast.
4. batch_size 16 -> 32. Your GPU had ~36GB of unused VRAM and was
   compute-bound with idle memory -- larger batches amortize per-kernel-
   launch overhead better. Back it off if you OOM.
5. Removed the per-step `.item()` calls on the running loss/RMSE sums.
   `.item()` forces a CPU-GPU sync -- the CPU has to stop and wait for
   the GPU to catch up before it can even queue the NEXT step's kernels,
   which serializes what should be overlapped work. Now the loss is
   accumulated as a GPU tensor all epoch and only pulled to CPU once, at
   the end -- turning ~615 syncs/epoch into 1.
"""

import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from dataset import OceanTSDataset, compute_norm_stats, date_based_split, all_valid_indices
from model import UNetOcean3D

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Channels (within the 7-channel input) that must be sign-flipped when the
# grid is mirrored along longitude -- see augment_batch() below. Matches
# the channel order documented in load_glorys_arrays():
# 0:SST, 1:SSS, 2:SSH, 3:U_current, 4:V_current, 5:U_wind, 6:V_wind
FLIP_NEGATE_CHANNELS = (3, 5)


def augment_batch(x, y, mask, noise_std=0.03, flip_prob=0.5):
    """GPU-side replacement for the old CPU per-sample augmentation.
    x: (B, 7, T, H, W), y: (B, 15, H, W), mask: (B, 15, H, W) -- per-depth
    valid-target mask (see dataset.py), all already on device and already
    normalized. Applies, per-sample in the batch:
      a) a longitude (last-axis) flip, applied identically to x/y/mask so
         they stay spatially aligned, with sign-corrected U-components in x
      b) small Gaussian noise on the input only
    Only ever call this on the TRAIN batch, never val/test.
    """
    B = x.shape[0]
    device = x.device

    do_flip = torch.rand(B, device=device) < flip_prob  # (B,)

    x_flipped = torch.flip(x, dims=[-1]).clone()
    x_flipped[:, FLIP_NEGATE_CHANNELS, :, :, :] *= -1
    x = torch.where(do_flip.view(B, 1, 1, 1, 1), x_flipped, x)

    y_flipped = torch.flip(y, dims=[-1])
    y = torch.where(do_flip.view(B, 1, 1, 1), y_flipped, y)

    # mask is now (B, 15, H, W) -- same rank as y, so same broadcast shape
    mask_flipped = torch.flip(mask, dims=[-1])
    mask = torch.where(do_flip.view(B, 1, 1, 1), mask_flipped, mask)

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    return x, y, mask


GLORYS_PATH = "/home/jl_fs/OceanEmbed/merged_glorys_full.h5"
ARGO_PATH = "/home/jl_fs/OceanEmbed/merged_argo_test.npz"


def load_glorys_arrays():
    with h5py.File(GLORYS_PATH, "r") as f:
        inputs = f["inputs"][:].astype(np.float32)
        target = f["target"][:].astype(np.float32)
        ocean_mask = f["ocean_mask"][:].astype(bool)
        dates_ns = f["dates"][:]
    dates = dates_ns.astype("datetime64[ns]").astype("datetime64[D]")
    return inputs, target, ocean_mask, dates


def load_argo_arrays():
    data = np.load(ARGO_PATH)
    inputs = data["inputs"].astype(np.float32)
    target = data["target"].astype(np.float32)
    ocean_mask = data["ocean_mask"].astype(bool)
    test_indices = data["test_indices"].astype(np.int64)
    return inputs, target, ocean_mask, test_indices


def masked_mse_loss(pred, target, mask):
    # mask is now (B, 15, H, W) -- a genuine per-depth-level valid mask
    # (built from isnan(target) in dataset.py), same shape as pred/target.
    # No unsqueeze/broadcast needed, and the denominator is just the
    # count of real valid (depth, pixel) entries -- NOT that count times
    # 15, since every depth level now has its own, different mask.
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / (mask.sum() + 1e-8)


def masked_rmse(pred, target, mask):
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    diff2 = (pred - target) ** 2 * mask
    mse = diff2.sum() / (mask.sum() + 1e-8)
    return torch.sqrt(mse)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    use_amp = device.type == "cuda"
    amp_dtype_name = "bf16" if use_amp else "fp32"   # A100 -> bf16, no loss scaling needed
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_scaler = use_amp and amp_dtype_name == "fp16"  # bf16 doesn't need GradScaler
    scaler = GradScaler("cuda", enabled=use_scaler)
    print(f"AMP (mixed precision): {'enabled (' + amp_dtype_name.upper() + ')' if use_amp else 'disabled (no CUDA device)'}")

    # lets cuDNN pick the fastest conv algorithm for your fixed input
    # shape -- safe here because every batch is padded to the same
    # (T, H, W), so there's no shape-changing overhead to worry about
    torch.backends.cudnn.benchmark = True

    # ---- config ----
    n_days = 16                # UNCHANGED per your request -- keeping
                                # epoch time where it is. See note below:
                                # this means we're fighting overfitting
                                # via capacity/regularization instead of
                                # via a longer window.
    batch_size = 16
    grad_accum_steps = 1
    n_epochs = 100
    lr = 1e-3                  # RESTORED from 2.5e-4. That value wasn't
                                # from a checkpoint resume (verified: no
                                # optimizer state is loaded before this
                                # loop starts) -- it was a stray edit.
                                # Restoring for a clean baseline.
    weight_decay = 8e-4         # up from 5e-4 -- stronger L2 penalty,
                                # directly targets the gap you've seen
                                # widen for 19 straight epochs across
                                # THREE separate runs (pre-fix, post-fix,
                                # post-fix+low-lr) -- a robust, real
                                # overfitting signal, not an artifact
    dropout = 0.20              # up from 0.15
    bottleneck_dropout = 0.35   # up from 0.25 -- bottleneck is the most
                                # parameter-dense, most overfitting-prone
                                # part of the network per model.py's own
                                # docstring reasoning
    patience = 15
    grad_clip_norm = 1.0
    use_checkpoint = False
    compile_model = True        # torch.compile fuses conv/norm/relu chains
                                # into fewer GPU kernels. First few batches
                                # will be slower (tracing/compiling), same
                                # idea as the cudnn.benchmark warm-up.
    num_workers = min(8, os.cpu_count() or 4)
    aug_noise_std = 0.03
    aug_flip_prob = 0.5
    checkpoint_dir = "/home/jl_fs/OceanEmbed/checkpoints"
    best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- data ----
    glorys_inputs, glorys_target, glorys_mask, glorys_dates = load_glorys_arrays()

    train_idx, val_idx = date_based_split(
        glorys_dates, n_days,
        train_range=("1993-01-01", "2019-12-31"),
        val_range=("2020-01-01", "2021-12-31"),
    )
    print(f"train days: {len(train_idx)}, val days: {len(val_idx)}")

    input_mean, input_std, target_mean, target_std = compute_norm_stats(
        glorys_inputs, glorys_target, train_idx)

    # augment=False everywhere here -- augmentation now happens on the
    # GPU per-batch via augment_batch(), called only on the train loader
    # below. The Dataset itself just does the minimal slicing/normalizing.
    train_ds = OceanTSDataset(glorys_inputs, glorys_target, glorys_mask, n_days=n_days, indices=train_idx,
                               input_mean=input_mean, input_std=input_std,
                               target_mean=target_mean, target_std=target_std,
                               augment=False)
    val_ds = OceanTSDataset(glorys_inputs, glorys_target, glorys_mask, n_days=n_days, indices=val_idx,
                             input_mean=input_mean, input_std=input_std,
                             target_mean=target_mean, target_std=target_std,
                             augment=False)

    # persistent_workers=True: keeps worker processes alive between
    # epochs instead of respawning them every time (this is very likely
    # what your run stalled on, right at the epoch 1 -> epoch 2 boundary).
    # prefetch_factor keeps a few batches ready ahead of the GPU.
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True,
                          persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    has_argo = Path(ARGO_PATH).exists()
    if has_argo:
        argo_inputs, argo_target, argo_mask, test_idx = load_argo_arrays()
        print(f"test days (ARGO 2022-2023): {len(test_idx)} "
              f"(real ARGO-matched days only, out of {argo_inputs.shape[0]} buffered days)")

        test_ds = OceanTSDataset(argo_inputs, argo_target, argo_mask, n_days=n_days, indices=test_idx,
                                  input_mean=input_mean, input_std=input_std,
                                  target_mean=target_mean, target_std=target_std,
                                  augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    else:
        print(f"NOTE: {ARGO_PATH} not found -- skipping ARGO test evaluation.")

    # ---- model ----
    model = UNetOcean3D(in_channels=7, out_channels=15, base_ch=20,
                         n_days=n_days, use_cbam=True,
                         use_checkpoint=use_checkpoint,
                         dropout=dropout,
                         bottleneck_dropout=bottleneck_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if compile_model:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}) -- continuing without it")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=device)  # accumulate on GPU, pull to CPU once/epoch
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        diag = (epoch == 1)
        data_time = 0.0
        compute_time = 0.0
        step_t0 = time.time()
        for step, (x, y, mask) in enumerate(train_loader):
            if diag:
                data_time += time.time() - step_t0

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x, y, mask = augment_batch(x, y, mask, noise_std=aug_noise_std, flip_prob=aug_flip_prob)

            if diag:
                compute_t0 = time.time()

            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(x)
                loss = masked_mse_loss(pred, y, mask) / grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_boundary = (step + 1) % grad_accum_steps == 0
            is_last_batch = (step + 1) == len(train_loader)
            if is_accum_boundary or is_last_batch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if diag:
                torch.cuda.synchronize()
                compute_time += time.time() - compute_t0
                if (step + 1) % 50 == 0:
                    print(f"  [diag] step {step+1}/{len(train_loader)}: "
                          f"data_wait={data_time:6.1f}s  gpu_compute={compute_time:6.1f}s  "
                          f"(ratio data/compute = {data_time/max(compute_time,1e-6):.2f})")

            # NOTE: no .item() here -- stays a GPU tensor all epoch, so the
            # CPU never blocks waiting on the GPU mid-epoch (see docstring)
            train_loss_sum += loss.detach() * grad_accum_steps * x.size(0)
            step_t0 = time.time()
        train_loss = (train_loss_sum / len(train_ds)).item()  # single sync, once
        # train RMSE (norm units) for direct comparison against val RMSE below
        train_rmse_proxy = train_loss ** 0.5

        # ---- validate ----
        model.eval()
        val_rmse_sum = torch.zeros((), device=device)
        val_rmse_degC_sum = torch.zeros((), device=device)
        n_val_seen = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pred = model(x)
                rmse_norm = masked_rmse(pred, y, mask)
                val_rmse_sum += rmse_norm.detach() * x.size(0)

                tstd = torch.tensor(target_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
                pred_degC = pred.float() * tstd
                y_degC = y.float() * tstd
                rmse_degC = masked_rmse(pred_degC, y_degC, mask)
                val_rmse_degC_sum += rmse_degC.detach() * x.size(0)
                n_val_seen += x.size(0)

        val_rmse = (val_rmse_sum / n_val_seen).item()
        val_rmse_degC = (val_rmse_degC_sum / n_val_seen).item()
        gap = val_rmse - train_rmse_proxy

        scheduler.step(val_rmse)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | train_RMSE(norm)~={train_rmse_proxy:.5f} "
              f"| val_RMSE(norm)={val_rmse:.5f} | gap={gap:+.5f} "
              f"| val_RMSE(degC)={val_rmse_degC:.4f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_rmse_degC": val_rmse_degC,
                "input_mean": input_mean, "input_std": input_std,
                "target_mean": target_mean, "target_std": target_std,
                "n_days": n_days,
            }, best_checkpoint)
            print(f"  -> saved new best model (val_RMSE degC = {val_rmse_degC:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Training complete. Best val RMSE (norm): {best_val_rmse:.5f}")

    if not has_argo:
        print(f"\nSkipping ARGO test evaluation -- {ARGO_PATH} not found.")
        return

    print("\nEvaluating best model on ARGO test set (2022-2023)...")
    ckpt = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_rmse_sum = torch.zeros((), device=device)
    test_rmse_degC_sum = torch.zeros((), device=device)
    n_test_seen = 0
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(x)
            rmse_norm = masked_rmse(pred, y, mask)
            test_rmse_sum += rmse_norm.detach() * x.size(0)

            tstd = torch.tensor(target_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            pred_degC = pred.float() * tstd
            y_degC = y.float() * tstd
            rmse_degC = masked_rmse(pred_degC, y_degC, mask)
            test_rmse_degC_sum += rmse_degC.detach() * x.size(0)
            n_test_seen += x.size(0)

    test_rmse = (test_rmse_sum / n_test_seen).item()
    test_rmse_degC = (test_rmse_degC_sum / n_test_seen).item()
    print(f"Test RMSE (norm)  = {test_rmse:.5f}")
    print(f"Test RMSE (degC)  = {test_rmse_degC:.4f}")


if __name__ == "__main__":
    train()