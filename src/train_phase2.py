import os
import csv
import math
import time
import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_features import Human36MFeatureClips
from model import PHDFor3DJoints
from utils import save_checkpoint, set_seed, load_phase1_checkpoint, freeze_module, phase2_losses

""" Coded by Luísa Ferreira, 2026 with assistance of ChatGPT 5.2 (OpenAI)."""

def get_core_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

# ---------------------------------------------------------
# Curriculum / rollout
# ---------------------------------------------------------
def curriculum_steps(epoch, max_steps, curriculum_epochs):
    # decides how many future steps to predict in the current epoch, for curriculum learning.
    # early training => easy task, predict just few steps
    # later training => harder task, predict more steps, up to max_steps.
    if max_steps <= 1:
        return 1
    if curriculum_epochs <= 1:
        return max_steps
    alpha = min(1.0, max(0.0, epoch / float(curriculum_epochs - 1)))
    steps = 1 + round(alpha * (max_steps - 1))
    return int(max(1, min(max_steps, steps)))
# for example, with max_steps=25 and curriculum_epochs=15:
# epoch 0 => pred_steps=1 (easy, just next step)then approximately:
# epoch 1 => pred_steps=3
# ...
# epoch 15 => pred_steps=25 (full difficulty)
# after epoch 15, it stays at pred_steps=25.

@torch.no_grad()
def sanity_check_sequence_length(seq_len, pred_steps):
    if pred_steps >= seq_len:
        raise ValueError(
            f"pred_steps ({pred_steps}) must be smaller than sequence length ({seq_len})."
        )


def rollout_latents_autoregressive(f_ar,phi,pred_steps,condition_len):
    # predict future latents autoregressively, starting from a teacher-forced prefix.
    # condition_len is how many initial steps to keep teacher-forced before switching to autoregressive predictions.
    B, T, D = phi.shape # batch size, total sequence length, latent dimension
    assert condition_len >= 1
    assert condition_len + pred_steps == T

    # start with teacher-forced prefix
    seq = phi[:, :condition_len, :]   # (B, condition_len, D)
    preds = []

    for _ in range(pred_steps):
        ar_out = f_ar(seq)            # (B, current_len, D)
        pred_next = ar_out[:, -1, :]  # predict next latent from available past
        preds.append(pred_next.unsqueeze(1))
        seq = torch.cat([seq, pred_next.unsqueeze(1)], dim=1)

    # all predictions concatenated, with teacher-forced prefix at the start
    phi_hat = torch.cat([phi[:, :condition_len, :]] + preds, dim=1)
    return phi_hat



def forward_phase2(model, feats, joints3d, pred_steps, w_recon_3d, w_future_3d, w_latent, w_vel):
    condition_len = feats.shape[1] - pred_steps
    if condition_len < 1:
        raise ValueError(
            f"Condition length became {condition_len}. "
            f"Check seq_len={feats.shape[1]} and pred_steps={pred_steps}."
        )

    phi = model.f_movie(feats)
    joints_phi = model.f_3D(phi)

    phi_hat = rollout_latents_autoregressive(
        model.f_AR,
        phi,
        pred_steps=pred_steps,
        condition_len=condition_len,
    )
    joints_hat = model.f_3D(phi_hat)

    loss, stats = phase2_losses(
        joints_phi=joints_phi,
        joints_hat=joints_hat,
        phi=phi,
        phi_hat=phi_hat,
        gt_joints=joints3d,
        condition_len=condition_len,
        w_recon_3d=w_recon_3d,
        w_future_3d=w_future_3d,
        w_latent=w_latent,
        w_vel=w_vel,
    )
    return loss, stats

def train(*, model, loader, optimizer, scaler, device, pred_steps, max_pred_steps, use_amp, grad_clip, w_recon_3d, w_future_3d, w_latent, w_vel):
    model.train()
    core_model = get_core_model(model)
    core_model.f_3D.eval()   # frozen in phase 2

    meter = {
        "loss": 0.0,
        "recon_3d": 0.0,
        "future_3d": 0.0,
        "latent": 0.0,
        "vel": 0.0,
        "count": 0,
    }

    autocast_enabled = use_amp and device.type == "cuda"
    seq_len = None

    for batch in loader:
        feats, joints3d, _joints2d, _K = batch
        feats = feats.to(device, non_blocking=True)
        joints3d = joints3d.to(device, non_blocking=True)

        if seq_len is None:
            seq_len = feats.shape[1]
            sanity_check_sequence_length(seq_len, max_pred_steps)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            loss, stats = forward_phase2(
                model=model,
                feats=feats,
                joints3d=joints3d,
                pred_steps=pred_steps,
                w_recon_3d=w_recon_3d,
                w_future_3d=w_future_3d,
                w_latent=w_latent,
                w_vel=w_vel,
            )

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        bs = feats.size(0)
        meter["loss"] += stats["loss"] * bs
        meter["recon_3d"] += stats["recon_3d"] * bs
        meter["future_3d"] += stats["future_3d"] * bs
        meter["latent"] += stats["latent"] * bs
        meter["vel"] += stats["vel"] * bs
        meter["count"] += bs

    n = max(1, meter["count"])
    return {
        "loss": meter["loss"] / n,
        "recon_3d": meter["recon_3d"] / n,
        "future_3d": meter["future_3d"] / n,
        "latent": meter["latent"] / n,
        "vel": meter["vel"] / n,
    }

@torch.no_grad()
def evaluate(*, model, loader, device, pred_steps, max_pred_steps, use_amp, w_recon_3d, w_future_3d, w_latent, w_vel):
    model.eval()

    meter = {
        "loss": 0.0,
        "recon_3d": 0.0,
        "future_3d": 0.0,
        "latent": 0.0,
        "vel": 0.0,
        "count": 0,
    }

    autocast_enabled = use_amp and device.type == "cuda"
    seq_len = None

    for batch in loader:
        feats, joints3d, _joints2d, _K = batch
        feats = feats.to(device, non_blocking=True)
        joints3d = joints3d.to(device, non_blocking=True)

        if seq_len is None:
            seq_len = feats.shape[1]
            sanity_check_sequence_length(seq_len, max_pred_steps)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            loss, stats = forward_phase2(
                model=model,
                feats=feats,
                joints3d=joints3d,
                pred_steps=pred_steps,
                w_recon_3d=w_recon_3d,
                w_future_3d=w_future_3d,
                w_latent=w_latent,
                w_vel=w_vel,
            )

        bs = feats.size(0)
        meter["loss"] += stats["loss"] * bs
        meter["recon_3d"] += stats["recon_3d"] * bs
        meter["future_3d"] += stats["future_3d"] * bs
        meter["latent"] += stats["latent"] * bs
        meter["vel"] += stats["vel"] * bs
        meter["count"] += bs

    n = max(1, meter["count"])
    return {
        "loss": meter["loss"] / n,
        "recon_3d": meter["recon_3d"] / n,
        "future_3d": meter["future_3d"] / n,
        "latent": meter["latent"] / n,
        "vel": meter["vel"] / n,
    }


def main():
    parser = argparse.ArgumentParser("Train Phase 2 of adapted PHD on Human3.6M")

    parser.add_argument("--root", type=str, required=True, help="Root folder with cached H36M feature clips")
    parser.add_argument("--phase1_ckpt", type=str, required=True, help="Checkpoint from phase 1 training")
    parser.add_argument("--outdir", type=str, default="checkpoints_phase2")
    parser.add_argument("--log_csv", type=str, default="")

    parser.add_argument("--train_subjects", type=int, nargs="+", default=[1, 6, 7, 8])
    parser.add_argument("--val_subjects", type=int, nargs="+", default=[5])
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--max_val_clips", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--joints_num", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_pred_steps", type=int, default=25)
    parser.add_argument("--curriculum_epochs", type=int, default=15)

    parser.add_argument("--w_recon_3d", type=float, default=1.0)
    parser.add_argument("--w_future_3d", type=float, default=1.0)
    parser.add_argument("--w_latent", type=float, default=1.0)
    parser.add_argument("--w_vel", type=float, default=0.1)

    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = Human36MFeatureClips(
        root=args.root,
        subjects=args.train_subjects,
        max_clips=args.max_train_clips,
        test_set=False,
    )
    val_set = Human36MFeatureClips(
        root=args.root,
        subjects=args.val_subjects,
        max_clips=args.max_val_clips,
        test_set=False,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")

    model = PHDFor3DJoints(
        latent_dim=args.latent_dim,
        joints_num=args.joints_num,
        dropout=args.dropout,
    )

    load_phase1_checkpoint(model, args.phase1_ckpt)

    # ------------------------------------------------
    # PHASE 2: freeze f_3d and train f_movie with autoregressive f_AR on future prediction
    # ------------------------------------------------
    freeze_module(model.f_3D)

    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable_params)}")
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    log_csv = args.log_csv or os.path.join(args.outdir, "train_phase2_log.csv")
    write_header = not os.path.exists(log_csv)

    best_val = float("inf")
    best_path = os.path.join(args.outdir, "best_phase2.pt")
    last_path = os.path.join(args.outdir, "last_phase2.pt")

    print("===== Phase 2 =====")
    print(f"Root: {args.root}")
    print(f"Phase1 ckpt: {args.phase1_ckpt}")
    print(f"Train subjects: {args.train_subjects}")
    print(f"Val subjects: {args.val_subjects}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Max pred steps: {args.max_pred_steps}")
    print(f"Curriculum epochs: {args.curriculum_epochs}")
    print(f"Loss weights: recon={args.w_recon_3d} future={args.w_future_3d} latent={args.w_latent} vel={args.w_vel}")
    print("===================")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_pred_steps = curriculum_steps(epoch, args.max_pred_steps, args.curriculum_epochs)

        train_stats = train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            pred_steps=train_pred_steps,
            max_pred_steps=args.max_pred_steps,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
            w_recon_3d=args.w_recon_3d,
            w_future_3d=args.w_future_3d,
            w_latent=args.w_latent,
            w_vel=args.w_vel,
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            pred_steps=args.max_pred_steps,
            max_pred_steps=args.max_pred_steps,
            use_amp=args.amp,
            w_recon_3d=args.w_recon_3d,
            w_future_3d=args.w_future_3d,
            w_latent=args.w_latent,
            w_vel=args.w_vel,
        )

        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"pred_steps={train_pred_steps:02d} | "
            f"train loss={train_stats['loss']:.5f} "
            f"(recon={train_stats['recon_3d']:.5f} future={train_stats['future_3d']:.5f} "
            f"latent={train_stats['latent']:.5f} vel={train_stats['vel']:.5f}) | "
            f"val loss={val_stats['loss']:.5f} "
            f"(recon={val_stats['recon_3d']:.5f} future={val_stats['future_3d']:.5f} "
            f"latent={val_stats['latent']:.5f} vel={val_stats['vel']:.5f}) | "
            f"lr={lr_now:.2e} | {dt:.1f}s"
        )

        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch",
                    "lr",
                    "train_pred_steps",
                    "train_loss",
                    "train_recon_3d",
                    "train_future_3d",
                    "train_latent",
                    "train_vel",
                    "val_loss",
                    "val_recon_3d",
                    "val_future_3d",
                    "val_latent",
                    "val_vel",
                ])
                write_header = False
            writer.writerow([
                epoch + 1,
                lr_now,
                train_pred_steps,
                train_stats["loss"],
                train_stats["recon_3d"],
                train_stats["future_3d"],
                train_stats["latent"],
                train_stats["vel"],
                val_stats["loss"],
                val_stats["recon_3d"],
                val_stats["future_3d"],
                val_stats["latent"],
                val_stats["vel"],
            ])

        # Save best according to future prediction error on validation.
        if val_stats["future_3d"] < best_val:
            best_val = val_stats["future_3d"]
            save_checkpoint(best_path, model, optimizer, epoch + 1, best_val, args)
            print(f"  -> Saved new best checkpoint to {best_path} (val future MPJPE={best_val:.5f})")

        if ((epoch + 1) % args.save_every) == 0:
            save_checkpoint(last_path, model, optimizer, epoch + 1, best_val, args)

    print("Training finished.")
    print(f"Best validation future MPJPE: {best_val:.5f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"CSV log: {log_csv}")


if __name__ == "__main__":
    main()
