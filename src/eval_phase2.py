import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_features import Human36MFeatureClips
from model import PHDFor3DJoints
from utils import set_seed, mpjpe_per_frame_mm, pa_mpjpe_per_frame_mm, dtw_aligned_metrics


""" 
Evaluate phase 2 calculating MPJPE and PA-MPJPE at multiple horizons, optionally with DTW alignment.
Also saves one example batch of predictions for visualization if --save_debug_npz is provided. The saved batch includes:
    - pred_seq: (B,T,J,3) predicted 3D joints for the whole sequence
    - gt_seq: (B,T,J,3) ground truth 3D joints for the whole sequence
    - pred_future: (B,F,J,3) predicted 3D joints for the future (after condition_len)
    - gt_future: (B,F,J,3) ground truth 3D joints for the future (after condition_len)
    - condition_len: int, the number of frames used as conditioning
    - horizons: List[int], the evaluation horizons in frames
    - meta: dict or list, the metadata for the batch (e.g. subject, action, cam)

Coded by Luísa Ferreira, 2026 with assistance of ChatGPT 5.2 (OpenAI).
"""

# ---------------------------------------------------------
# aux 
# ---------------------------------------------------------

def get_core_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint_strictish(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print("[load] Missing keys:")
        for k in missing:
            print("  ", k)
    if unexpected:
        print("[load] Unexpected keys:")
        for k in unexpected:
            print("  ", k)


def extract_meta_batch(meta):
    out = {}
    box = meta["box"]
    if isinstance(box, torch.Tensor):
        out["box"] = box.detach().cpu().numpy()
    else:
        out["box"] = np.asarray(box)

    subject = meta["subject"]
    if isinstance(subject, torch.Tensor):
        out["subject"] = subject.detach().cpu().numpy()
    else:
        out["subject"] = np.asarray(subject)

    out["action"] = np.asarray(meta["action"], dtype=object)
    out["cam"] = np.asarray(meta["cam"], dtype=object)

    start = meta["start"]
    if isinstance(start, torch.Tensor):
        out["start"] = start.detach().cpu().numpy()
    else:
        out["start"] = np.asarray(start)

    end = meta["end"]
    if isinstance(end, torch.Tensor):
        out["end"] = end.detach().cpu().numpy()
    else:
        out["end"] = np.asarray(end)

    return out

# ---------------------------------------------------------
# AR rollouts
# ---------------------------------------------------------

@torch.no_grad()
def rollout_latents_autoregressive(f_ar, phi, pred_steps, condition_len):
    """
    phi: (B, T, D)
    returns phi_hat: (B, T, D)
      - teacher prefix [0:condition_len)
      - predicted future [condition_len:T)
    """
    B, T, D = phi.shape
    assert condition_len >= 1
    assert condition_len + pred_steps == T

    seq = phi[:, :condition_len, :]
    preds = []

    for _ in range(pred_steps):
        ar_out = f_ar(seq)            # (B, cur_len, D)
        pred_next = ar_out[:, -1, :]  # (B, D)
        preds.append(pred_next.unsqueeze(1))
        seq = torch.cat([seq, pred_next.unsqueeze(1)], dim=1)

    phi_hat = torch.cat([phi[:, :condition_len, :]] + preds, dim=1)
    return phi_hat


@torch.no_grad()
def rollout_constant_pose(joints_est_full, condition_len):
    """
    joints_est_full: (B, T, J, 3)
    returns joints_hat: (B, T, J, 3)

    Keeps the last conditioned pose fixed for all future steps.
    """
    B, T, J, D = joints_est_full.shape
    assert D == 3
    assert condition_len >= 1
    assert condition_len < T

    prefix = joints_est_full[:, :condition_len].clone()
    future_len = T - condition_len
    last_pose = joints_est_full[:, condition_len - 1:condition_len].clone()
    future = last_pose.repeat(1, future_len, 1, 1)

    joints_hat = torch.cat([prefix, future], dim=1)
    return joints_hat

# ---------------------------------------------------------
# Print aux function
# ---------------------------------------------------------

def fmt_row(name, vals, horizons):
    parts = [name.ljust(18)]
    for h in horizons:
        v = vals.get(h, float("nan"))
        parts.append(f"{v:8.2f}")
    return " ".join(parts)

# ---------------------------------------------------------
# Per-mode prediction
# ---------------------------------------------------------

@torch.no_grad()
def predict_all_modes(core, feats, condition_len, use_amp, device):
    """
    returns dict with:
      latent_ar -> (B,T,J,3) or None
      constant  -> (B,T,J,3) or None
    """
    B, T = feats.shape[:2]
    future_len = T - condition_len
    autocast_enabled = use_amp and device.type == "cuda"

    outputs = {
        "latent_ar": None,
        "constant": None,
    }

    with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
        phi = core.f_movie(feats)
        joints_est = core.f_3D(phi)

        outputs["constant"] = rollout_constant_pose(
            joints_est_full=joints_est,
            condition_len=condition_len,
        )

        if hasattr(core, "f_AR") and core.f_AR is not None:
            phi_hat = rollout_latents_autoregressive(
                core.f_AR,
                phi,
                pred_steps=future_len,
                condition_len=condition_len,
            )
            outputs["latent_ar"] = core.f_3D(phi_hat)

    return outputs


# ---------------------------------------------------------
# Eval all modes
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_phase2_all_modes(model, loader, device, condition_len, horizons, use_amp, use_dtw, save_debug_npz):
    core = get_core_model(model)
    core.eval()

    if hasattr(core, "f_movie"):
        core.f_movie.eval()
    if hasattr(core, "f_3D"):
        core.f_3D.eval()
    if hasattr(core, "f_AR"):
        core.f_AR.eval()

    modes = ["constant", "latent_ar"]

    sums = {
        mode: {
            "MPJPE_mm": {h: 0.0 for h in horizons},
            "PA_MPJPE_mm": {h: 0.0 for h in horizons},
            "DTW_MPJPE_mm": {h: 0.0 for h in horizons},
            "DTW_PA_MPJPE_mm": {h: 0.0 for h in horizons},
            "cnt": {h: 0 for h in horizons},
            "cnt_dtw": {h: 0 for h in horizons},
        }
        for mode in modes
    }

    saved_debug = False

    for batch_idx, batch in enumerate(loader):
        feats, joints3d, _joints2d, _K, meta = batch
        feats = feats.to(device, non_blocking=True)
        joints3d = joints3d.to(device, non_blocking=True)

        B, T = feats.shape[:2]
        future_len = T - condition_len
        if future_len <= 0:
            raise ValueError(f"condition_len={condition_len} must be smaller than T={T}")

        pred_dict = predict_all_modes(
            core=core,
            feats=feats,
            condition_len=condition_len,
            use_amp=use_amp,
            device=device,
        )

        gt_full = joints3d.detach().cpu().numpy()
        gt_future = gt_full[:, condition_len:]

        if save_debug_npz and not saved_debug:
            os.makedirs(os.path.dirname(save_debug_npz) or ".", exist_ok=True)

            meta_batch = extract_meta_batch(meta)

            save_dict = {
                "gt_seq": gt_full,                         # (B,T,J,3)
                "gt_future": gt_future,                   # (B,F,J,3)
                "condition_len": np.array(condition_len, dtype=np.int64),
                "horizons": np.array(horizons, dtype=np.int64),

                "meta_box": meta_batch["box"],
                "meta_subject": meta_batch["subject"],
                "meta_action": meta_batch["action"],
                "meta_cam": meta_batch["cam"],
                "meta_start": meta_batch["start"],
                "meta_end": meta_batch["end"],
            }

            if pred_dict["constant"] is not None:
                const_full = pred_dict["constant"].detach().cpu().numpy()
                save_dict["pred_seq_constant"] = const_full                  # (B,T,J,3)
                save_dict["pred_future_constant"] = const_full[:, condition_len:]  # (B,F,J,3)

            if pred_dict["latent_ar"] is not None:
                latent_full = pred_dict["latent_ar"].detach().cpu().numpy()
                save_dict["pred_seq_phi"] = latent_full                      # (B,T,J,3)
                save_dict["pred_future_phi"] = latent_full[:, condition_len:]      # (B,F,J,3)

            np.savez(save_debug_npz, **save_dict)
            print(f"[debug] Saved full batch prediction bundle to {save_debug_npz}")
            saved_debug = True

        for mode in modes:
            pred_tensor = pred_dict[mode]
            if pred_tensor is None:
                continue

            pred_full = pred_tensor.detach().cpu().numpy()
            pred_future = pred_full[:, condition_len:]

            for b in range(B):
                mpjpe_pf = mpjpe_per_frame_mm(pred_future[b], gt_future[b])
                pa_pf = pa_mpjpe_per_frame_mm(pred_future[b], gt_future[b])

                for h in horizons:
                    if h <= len(mpjpe_pf):
                        sums[mode]["MPJPE_mm"][h] += float(mpjpe_pf[h - 1])
                        sums[mode]["PA_MPJPE_mm"][h] += float(pa_pf[h - 1])
                        sums[mode]["cnt"][h] += 1

                if use_dtw:
                    dtw_mpjpe_seq, dtw_pa_seq = dtw_aligned_metrics(pred_future[b], gt_future[b])

                    for h in horizons:
                        if h <= len(dtw_mpjpe_seq):
                            sums[mode]["DTW_MPJPE_mm"][h] += float(dtw_mpjpe_seq[:h].mean())
                            sums[mode]["DTW_PA_MPJPE_mm"][h] += float(dtw_pa_seq[:h].mean())
                            sums[mode]["cnt_dtw"][h] += 1

    metrics = {}
    for mode in modes:
        mode_metrics = {}

        ran = any(sums[mode]["cnt"][h] > 0 for h in horizons)

        if not ran:
            mode_metrics["MPJPE_mm"] = {h: float("nan") for h in horizons}
            mode_metrics["PA_MPJPE_mm"] = {h: float("nan") for h in horizons}
            if use_dtw:
                mode_metrics["DTW_MPJPE_mm"] = {h: float("nan") for h in horizons}
                mode_metrics["DTW_PA_MPJPE_mm"] = {h: float("nan") for h in horizons}
            metrics[mode] = mode_metrics
            continue

        mode_metrics["MPJPE_mm"] = {
            h: sums[mode]["MPJPE_mm"][h] / max(sums[mode]["cnt"][h], 1)
            for h in horizons
        }
        mode_metrics["PA_MPJPE_mm"] = {
            h: sums[mode]["PA_MPJPE_mm"][h] / max(sums[mode]["cnt"][h], 1)
            for h in horizons
        }

        if use_dtw:
            mode_metrics["DTW_MPJPE_mm"] = {
                h: sums[mode]["DTW_MPJPE_mm"][h] / max(sums[mode]["cnt_dtw"][h], 1)
                for h in horizons
            }
            mode_metrics["DTW_PA_MPJPE_mm"] = {
                h: sums[mode]["DTW_PA_MPJPE_mm"][h] / max(sums[mode]["cnt_dtw"][h], 1)
                for h in horizons
            }

        metrics[mode] = mode_metrics

    return metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Evaluate Phase 2 like PHD paper (adapted to 3D joints)")

    parser.add_argument("--root", type=str, required=True, help="Root folder of cached H36M clips")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint")
    parser.add_argument("--subjects", type=int, nargs="+", default=[9, 11], help="Test subjects")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--joints_num", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--condition_len", type=int, default=15)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--use_dtw", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_out", type=str, default="")
    parser.add_argument(
        "--save_debug_npz",
        type=str,
        default="",
        help="Path to save one example prediction .npz for visualization"
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = Human36MFeatureClips(
        root=args.root,
        subjects=args.subjects,
        max_clips=args.max_clips,
        test_set=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    sample = dataset[0]
    seq_len = sample[0].shape[0]
    future_len = seq_len - args.condition_len
    if future_len <= 0:
        raise ValueError(
            f"Sequence length {seq_len} must be greater than condition_len {args.condition_len}"
        )

    horizons = sorted(set(int(h) for h in args.horizons if h <= future_len))
    dropped = [h for h in args.horizons if h > future_len]
    if dropped:
        print(
            f"[warn] Dropping horizons {dropped} because seq_len={seq_len} "
            f"and condition_len={args.condition_len} allow only {future_len} future frames."
        )

    print("===== Eval Phase 2 =====")
    print(f"Root: {args.root}")
    print(f"Ckpt: {args.ckpt}")
    print(f"Subjects: {args.subjects}")
    print(f"Num clips: {len(dataset)}")
    print(f"Seq len: {seq_len}")
    print(f"Condition len: {args.condition_len}")
    print(f"Future len available: {future_len}")
    print(f"Horizons: {horizons}")
    print(f"Use DTW: {args.use_dtw}")
    print("========================")

    model = PHDFor3DJoints(
        latent_dim=args.latent_dim,
        joints_num=args.joints_num,
        dropout=args.dropout,
    )
    load_checkpoint_strictish(model, args.ckpt)

    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    metrics = evaluate_phase2_all_modes(
        model=model,
        loader=loader,
        device=device,
        condition_len=args.condition_len,
        horizons=horizons,
        use_amp=args.amp,
        use_dtw=args.use_dtw,
        save_debug_npz=args.save_debug_npz,
    )

    print()
    print("Paper-style evaluation on Human3.6M test")
    print("(adapted for 3D joints; errors in mm)")

    for mode_name, pretty_name in [
        ("constant", "Constant baseline"),
        ("latent_ar", "AR on phi"),
    ]:
        print()
        print(f"[{pretty_name}]")
        header = "Metric".ljust(18) + "".join([f"{h:>8d}" for h in horizons])
        print(header)
        print("-" * len(header))
        print(fmt_row("MPJPE", metrics[mode_name]["MPJPE_mm"], horizons))
        print(fmt_row("PA-MPJPE", metrics[mode_name]["PA_MPJPE_mm"], horizons))
        if args.use_dtw:
            print(fmt_row("DTW-MPJPE", metrics[mode_name]["DTW_MPJPE_mm"], horizons))
            print(fmt_row("DTW-PA-MPJPE", metrics[mode_name]["DTW_PA_MPJPE_mm"], horizons))

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "metric"] + [str(h) for h in horizons])

            for mode_name in ["constant", "latent_ar"]:
                writer.writerow(
                    [mode_name, "MPJPE_mm"] +
                    [metrics[mode_name]["MPJPE_mm"][h] for h in horizons]
                )
                writer.writerow(
                    [mode_name, "PA_MPJPE_mm"] +
                    [metrics[mode_name]["PA_MPJPE_mm"][h] for h in horizons]
                )

                if args.use_dtw:
                    writer.writerow(
                        [mode_name, "DTW_MPJPE_mm"] +
                        [metrics[mode_name]["DTW_MPJPE_mm"][h] for h in horizons]
                    )
                    writer.writerow(
                        [mode_name, "DTW_PA_MPJPE_mm"] +
                        [metrics[mode_name]["DTW_PA_MPJPE_mm"][h] for h in horizons]
                    )

        print(f"\nSaved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()