import os
import time
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    H36M_ROOT,
    SEQ_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
    JOINTS_NUM,
    FRAME_SKIP
)
from dataset_features import Human36MFeatureClips
from model import PHDFor3DJoints as PHD
from utils import save_checkpoint, mpjpe_3d, bone_length_loss

""" Coded by Luísa Ferreira, 2026 with assistance of ChatGPT 5.2 (OpenAI)."""


def train(model, loader, optim, scaler, sched, device, lambda_bone: float = 1.0, log_every: int = 500):
    model.train()
    epoch_start = time.time()

    running_loss = 0.0
    running_l3d = 0.0
    running_lbone = 0.0
    running_mpjpe = 0.0
    n_batches = 0

    # timers for each thing (data / forward / backward / total)
    timers = defaultdict(float)

    # this measures the time between iterations (i.e., DataLoader + host work)
    end_data = time.time()

    for it, batch in enumerate(loader):
        t_iter_start = time.time()

        # --------------------
        # Data loading time
        # --------------------
        timers["data"] += (t_iter_start - end_data)

        feats, joints3d, joints2d, K = batch

        feats = feats.to(device, non_blocking=True)         # (B,T,2048)
        joints3d = joints3d.to(device, non_blocking=True)   # (B,T,J,3)
        joints2d = joints2d.to(device, non_blocking=True)   # (B,T,J,2) 
        K = K.to(device, non_blocking=True)              # (3,3) or (B,3,3) or (B,T,3,3)

        optim.zero_grad(set_to_none=True)

        # --------------------
        # Forward + Loss time
        # --------------------
        t_fwd = time.time()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            # IMPORTANT: I precomputed ResNet features, so we must NOT pass video into the model
            # We use a dedicated path that assumes feats = ResNet output (2048D)
            _phi, _phi_hat, joints_pred, _joints_hat = model(feats, predict_future=False)
            # joints_pred: (B,T,J,3) assumed camera coordinates that match K

        # 3D loss
        l3d = mpjpe_3d(joints_pred, joints3d)
        lbone = bone_length_loss(joints_pred, joints3d)


        loss = l3d + lambda_bone * lbone
        timers["forward+loss"] += (time.time() - t_fwd)

        # --------------------
        # Backward + Optim time
        # --------------------
        t_bwd = time.time()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        sched.step()

        timers["backward"] += (time.time() - t_bwd)

        # --------------------
        # Logging / metrics
        # --------------------
        running_loss += float(loss.item())
        running_l3d += float(l3d.item())
        running_lbone += float(lbone.item())
        running_mpjpe += mpjpe_3d(joints_pred, joints3d).item()
        n_batches += 1

        t_iter_end = time.time()
        timers["iter"] += (t_iter_end - t_iter_start)
        end_data = t_iter_end

        if log_every > 0 and (it + 1) % log_every == 0:
            dt_epoch = time.time() - epoch_start
            print(
                f"[3D + bone] iter {it+1:05d}/{len(loader):05d} | "
                f"loss {running_loss/n_batches:.6f} (3d {running_l3d/n_batches:.6f} + "
                f"{lambda_bone:.3g}*bone {running_lbone/n_batches:.6f}) | "
                f"mpjpe {running_mpjpe/n_batches:.3f} | "
                f"time/iter {timers['iter']/n_batches:.4f}s | "
                f"epoch {dt_epoch:.1f}s"
            )

    # print time to each thing (per epoch)
    epoch_time = time.time() - epoch_start
    print("\n[Train timing]")
    print(f"  data loading:    {timers['data']:.2f}s")
    print(f"  forward+loss:    {timers['forward+loss']:.2f}s")
    print(f"  backward+optim:  {timers['backward']:.2f}s")
    print(f"  total iter time: {timers['iter']:.2f}s")
    print(f"  total epoch:     {epoch_time:.2f}s")
    print(f"  avg iter time:   {timers['iter']/max(n_batches,1):.4f}s\n")

    return running_loss / max(n_batches, 1), running_mpjpe / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, lambda_bone: float = 1.0, test_set: bool = False):
    model.eval()

    total_loss = 0.0
    total_l3d = 0.0
    total_lbone = 0.0
    total_mpjpe = 0.0
    n_batches = 0

    # timing for evaluation
    t_eval_start = time.time()
    timers = defaultdict(float)
    end_data = time.time()

    for batch in loader:
        t_iter_start = time.time()
        timers["data"] += (t_iter_start - end_data)
        if test_set:
            feats, joints3d, joints2d, K, meta = batch
        else:
            feats, joints3d, joints2d, K = batch

        feats = feats.to(device, non_blocking=True)         # (B,T,2048)
        joints3d = joints3d.to(device, non_blocking=True)   # (B,T,J,3)
        joints2d = joints2d.to(device, non_blocking=True)   # (B,T,J,2)
        K = K.to(device, non_blocking=True)            # (B,3,3) 

        t_fwd = time.time()
        # IMPORTANT: I precomputed ResNet features, so we must NOT pass video into the model
        # We use a dedicated path that assumes feats = ResNet output (2048D)
        _phi, _phi_hat, joints_pred, _joints_hat = model(feats, predict_future=False)
        # joints_pred: (B,T,J,3) assumed camera coordinates that match K

        # 3D loss
        l3d = mpjpe_3d(joints_pred, joints3d)
        lbone = bone_length_loss(joints_pred, joints3d)

        loss = l3d + lambda_bone * lbone

        total_loss += float(loss.item())
        total_l3d += float(l3d.item())
        total_lbone += float(lbone.item())
        total_lbone += float(lbone.item())
        total_mpjpe += mpjpe_3d(joints_pred, joints3d).item()
        n_batches += 1

        t_iter_end = time.time()
        timers["iter"] += (t_iter_end - t_iter_start)
        end_data = t_iter_end

    eval_time = time.time() - t_eval_start
    print("[Val timing]")
    print(f"  data loading:  {timers['data']:.2f}s")
    print(f"  forward:       {timers['forward']:.2f}s")
    print(f"  total:         {eval_time:.2f}s")
    print(f"  avg iter time: {timers['iter']/max(n_batches,1):.4f}s\n")

    # (optionally you can also print l3d/l2d outside)
    return (
        total_loss / max(n_batches, 1),
        total_mpjpe / max(n_batches, 1),
        total_l3d / max(n_batches, 1),
        total_lbone / max(n_batches, 1),
    )


def main():
    parser = argparse.ArgumentParser("Phase-1 training: freeze ResNet, train f_movie + f_3D (3D joints + 2D reprojection)")
    parser.add_argument("--root", type=str, default=H36M_ROOT)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lambda-bone", type=float, default=1.0, help="bone length loss weight")
    parser.add_argument("--outdir", type=str, default="./runs/phase1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=500)

    parser.add_argument("--train-subjects", type=int, nargs="+", default=[1, 6, 7, 8], help="Subjects to use for training")
    parser.add_argument("--val-subjects", type=int, nargs="+", default=[5], help="Subjects to use for validation")
    args = parser.parse_args()

    # set device and multi-GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        device = torch.device("cpu")
        num_gpus = 0
        gpu_ids = []
    
    use_multi_gpu = num_gpus > 1 

    os.makedirs(args.outdir, exist_ok=True)

    effective_batch_size = args.batch_size
    if use_multi_gpu:
        # each GPU gets batch_size / num_gpus samples
        per_gpu_batch_size = args.batch_size // num_gpus
        effective_batch_size = per_gpu_batch_size * num_gpus
        print(f"Multi-GPU training: {num_gpus} GPUs (IDs: {gpu_ids})")
        print(f"Effective batch size: {effective_batch_size} ({per_gpu_batch_size} per GPU)")

    # load data
    train_set = Human36MFeatureClips(
        root=args.root,
        subjects=args.train_subjects,
    )
    val_set = Human36MFeatureClips(
        root=args.root,
        subjects=args.val_subjects,
    )

    # DataLoader optimizations
    loader_kwargs = dict(
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        drop_last=False,
        **loader_kwargs
    )

    model = PHD(joints_num=JOINTS_NUM)

    # ----------------------------------
    # TRAINING PHASE 1 : freeze ResNet, train f_movie + f_3D
    # ----------------------------------
    for p in model.f_AR.parameters():
        p.requires_grad = False

    # move to device before wrapping in DataParallel (if using)
    model = model.to(device)
    
    # multi-GPU with nn.DataParallel
    if use_multi_gpu:
        print(f"Wrapping model in DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found. Did you accidentally freeze everything?")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        # handle nn.DataParallel case for loading state_dict
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt["model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_val={best_val:.4f})")

    print("===== Phase-1 training =====")
    print(f"Device: {device}")
    print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")
    print(f"Seq len: {args.seq_len} | Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"Lambda Bone: {args.lambda_bone:.3g}")
    print("============================")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        t_epoch = time.time()

        tr_loss, tr_mpjpe = train(
            model, train_loader, optim, scaler, sched, device,
            lambda_bone=args.lambda_bone,
            log_every=args.log_every
        )
        va_loss, va_mpjpe, va_l3d, va_lbone = evaluate(
            model, val_loader, device,
            lambda_bone=args.lambda_bone,
        )

        print(f"Train: loss={tr_loss:.6f} | mpjpe={tr_mpjpe:.3f}")
        print(f"Val:   loss={va_loss:.6f} (3d {va_l3d:.6f} + {args.lambda_bone:.3g}*bone {va_lbone:.6f}) | mpjpe={va_mpjpe:.3f}")
        print(f"Epoch time: {time.time() - t_epoch:.2f}s")

        save_checkpoint(
            os.path.join(args.outdir, "last.pt"),
            model, optim, epoch, best_val, args
        )

        if va_mpjpe < best_val:
            best_val = va_mpjpe
            save_checkpoint(
                os.path.join(args.outdir, "best.pt"),
                model, optim, epoch, best_val, args
            )
            print(f"New best val MPJPE: {best_val:.3f} (saved best.pt)")

    print("\nDone.")
    print(f"Best val MPJPE: {best_val:.3f}")


if __name__ == "__main__":
    main()