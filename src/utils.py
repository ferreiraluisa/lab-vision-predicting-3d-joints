import os
from typing import Optional, Tuple, Dict
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from torchvision.io import VideoReader
import torchvision.transforms.functional as F

import numpy as np
# -----------------------------
# Metrics / Losses
# -----------------------------

H36M_EDGES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

# Pre-build edge index tensors once at module load time (moved to device inside each loss)
_EDGE_SRC = torch.tensor([e[0] for e in H36M_EDGES], dtype=torch.long)  # (E,)
_EDGE_DST = torch.tensor([e[1] for e in H36M_EDGES], dtype=torch.long)  # (E,)


# Bone length loss: encourage predicted bone lengths to match GT at every timestep.
# pred, gt: (B,T,J,3)
def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    src = _EDGE_SRC.to(pred.device)
    dst = _EDGE_DST.to(pred.device)
    pred_bones = pred[:, :, dst] - pred[:, :, src]   # (B,T,E,3)
    gt_bones   = gt[:, :, dst]   - gt[:, :, src]     # (B,T,E,3)

    return torch.mean(torch.norm(pred_bones - gt_bones, dim=-1))  # mean L2 bone length error

def mpjpe_3d(predicted, target):
    """
    Mean per-joint position error (mean Euclidean distance).
    predicted/target: (..., 3)
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))

# phase 2 temporal velocity loss
def temporal_velocity_loss(pred, target):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_v = pred[:, 1:] - pred[:, :-1]
    targ_v = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(pred_v - targ_v, dim=-1))

def phase2_losses(joints_phi, joints_hat, phi, phi_hat, gt_joints,condition_len, w_recon_3d, w_future_3d, w_latent, w_vel,):
    recon_3d = mpjpe_3d(joints_phi, gt_joints)
    future_3d = mpjpe_3d(joints_hat[:, condition_len:], gt_joints[:, condition_len:])
    latent = torch.nn.functional.smooth_l1_loss(
        phi_hat[:, condition_len:],
        phi[:, condition_len:],
    )
    vel = temporal_velocity_loss(
        joints_hat[:, condition_len:],
        gt_joints[:, condition_len:],
    )

    total = (
        w_recon_3d * recon_3d
        + w_future_3d * future_3d
        + w_latent * latent
        + w_vel * vel
    )

    stats = {
        "loss": float(total.detach().item()),
        "recon_3d": float(recon_3d.detach().item()),
        "future_3d": float(future_3d.detach().item()),
        "latent": float(latent.detach().item()),
        "vel": float(vel.detach().item()),
    }
    # return Tuple[torch.Tensor, Dict[str, float]]
    return total, stats


def project_with_K_torch(P_cam, K, eps=1e-6):
    # perpective projection with camera intrinsics K(no distortion)
    # TESTED ON TOY DATA, WORKS AS EXPECTED
    assert P_cam.shape[-1] == 3, f"P_cam must end with 3, got {P_cam.shape}"
    lead = P_cam.shape[:-1]

    if K.dim() == 2:
        Kb = K
    else:
        assert K.shape[-2:] == (3, 3), f"K must end with (3,3), got {K.shape}"
        Kb = K
        missing = len(lead) - (Kb.dim() - 2)
        if missing > 0:
            Kb = Kb.view(*Kb.shape[:-2], *([1] * missing), 3, 3)

    P_col = P_cam.unsqueeze(-1)                      # (..., 3, 1)
    P_h = torch.matmul(Kb, P_col).squeeze(-1)        # (..., 3)
    z = P_h[..., 2:3].clamp(min=eps)
    uv = P_h[..., 0:2] / z
    return uv

def masked_l2d_loss_sq(pred2d, gt2d, vis, eps=1e-6):
    """
    Paper 2D loss:
      L2D = || v ⊙ (x - x_hat) ||_2^2

    """
    diff = pred2d - gt2d                               # (B,T,J,2)
    se = diff.pow(2).sum(dim=-1)                       # (B,T,J) squared L2 per joint
    se = se * vis.clamp(0, 1)
    denom = vis.sum().clamp_min(1.0)                   # number of visible joints across batch
    return se.sum() / denom


def weak_persp_fit_and_project(xy3d, gt2d, vis, eps=1e-6):
    # Weak perspective fit and project
    w = vis.clamp(0, 1).unsqueeze(-1)                  # (B,T,J,1)
    wsum = w.sum(dim=2, keepdim=True).clamp_min(eps)   # (B,T,1,1)

    xy_mu = (w * xy3d).sum(dim=2, keepdim=True) / wsum
    uv_mu = (w * gt2d).sum(dim=2, keepdim=True) / wsum

    xy0 = xy3d - xy_mu
    uv0 = gt2d - uv_mu

    # scalar s minimizing || w*(s*xy0 - uv0) ||^2
    num = (w * (xy0 * uv0)).sum(dim=(2, 3), keepdim=True)                  # (B,T,1,1)
    den = (w * (xy0 * xy0)).sum(dim=(2, 3), keepdim=True).clamp_min(eps)   # (B,T,1,1)
    s = num / den

    # translation
    t = uv_mu - s * xy_mu                                                   # (B,T,1,2)

    proj = s * xy3d + t                                                     # (B,T,J,2)
    return proj



# ------------------------------------------------
# Checkpointing & Freezing
# ------------------------------------------------
def save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer,
                    epoch: int, best_val: float, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model": model_state,
            "optim": optim.state_dict(),
            "args": vars(args),
        },
        path,
    )

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def freeze_module(module: nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

def load_phase1_checkpoint(model: nn.Module, ckpt_path: str) -> Tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Loaded phase-1 checkpoint from: {ckpt_path}")
    if missing:
        print("[load] Missing keys:")
        for k in missing:
            print("  ", k)
    if unexpected:
        print("[load] Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    start_epoch = 0
    best_val = float("inf")
    if isinstance(ckpt, dict):
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
    return start_epoch, best_val

# ------------------------------------------------
# Seed setting
# ------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------
# Evaluation utils
# ---------------------------------

# phase 2, procrustes alignment + dtW
# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def mpjpe_per_frame_mm(pred, gt):
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    return err * 1000.0


def compute_similarity_transform(pred, gt, eps=1e-8):
    """
    pred, gt: (J, 3)
    returns aligned pred: (J, 3)
    Solves: min_{s,R,t} || gt - (s * pred @ R + t) ||_F
    """

    pred = pred.astype(np.float64, copy=False)
    gt   = gt.astype(np.float64, copy=False)

    mu_pred = pred.mean(axis=0, keepdims=True)   # (1,3)
    mu_gt   = gt.mean(axis=0, keepdims=True)     # (1,3)

    X = pred - mu_pred
    Y = gt   - mu_gt

    var_X = np.sum(X ** 2)
    if var_X < eps:
        return pred.copy()

    K = X.T @ Y
    U, s, Vt = np.linalg.svd(K)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    scale = np.sum(s) / (var_X + eps)
    t = mu_gt - scale * (mu_pred @ R)

    aligned = scale * (pred @ R) + t
    return aligned


def pa_mpjpe_per_frame_mm(pred, gt):
    vals = []
    for t in range(pred.shape[0]):
        pa_pred = compute_similarity_transform(pred[t], gt[t])
        err = np.linalg.norm(pa_pred - gt[t], axis=-1).mean() * 1000.0
        vals.append(err)
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------
# DTW
# ---------------------------------------------------------

def dtw_path(cost):
    T1, T2 = cost.shape
    dp = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    i, j = T1, T2
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dp[i - 1, j], i - 1, j),
            (dp[i, j - 1], i, j - 1),
            (dp[i - 1, j - 1], i - 1, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    while i > 0:
        i -= 1
        path.append((i, 0))

    while j > 0:
        j -= 1
        path.append((0, j))

    path.reverse()
    return path


def dtw_cost_matrix(pred, gt):
    T1, T2 = pred.shape[0], gt.shape[0]
    cost = np.zeros((T1, T2), dtype=np.float64)

    for i in range(T1):
        for j in range(T2):
            cost[i, j] = np.linalg.norm(pred[i] - gt[j], axis=-1).mean()

    return cost


def dtw_aligned_metrics(pred_future, gt_future):
    cost = dtw_cost_matrix(pred_future, gt_future)
    path = dtw_path(cost)

    dtw_mpjpe_seq = []
    dtw_pa_seq = []

    for i, j in path:
        p = pred_future[i:i + 1]
        g = gt_future[j:j + 1]
        dtw_mpjpe_seq.append(float(mpjpe_per_frame_mm(p, g)[0]))
        dtw_pa_seq.append(float(pa_mpjpe_per_frame_mm(p, g)[0]))

    return np.asarray(dtw_mpjpe_seq), np.asarray(dtw_pa_seq)





# ------------------------------------------------
# Visualization utils
# ------------------------------------------------
def _as_numpy(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def to_uint8_rgb(x):
    # x: (3, H, W)
    x = np.transpose(x, (1, 2, 0))  # (H, W, 3)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

def read_clip_uint8(video_path: str, start: int, end: int, frame_skip: int) -> torch.Tensor:
    # return uint8 frames shaped (T,H,W,C) with T=end-start.
    target_T = int(end - start)
    if target_T <= 0:
        raise ValueError(f"Invalid clip range: start={start}, end={end}")

    # faster using VideoReader, but can fail on some videos
    try:
        reader = VideoReader(video_path, "video")
        md = reader.get_metadata()
        fps = md["video"]["fps"][0]

        start_time = (start * frame_skip) / float(fps)
        reader.seek(start_time)

        frames = []
        frame_idx = 0

        for fr in reader:
            if frame_idx % frame_skip == 0:
                x = fr["data"]         # (C,H,W) uint8
                x = x.permute(1, 2, 0) # (H,W,C)
                frames.append(x)
                if len(frames) >= target_T:
                    break

            frame_idx += 1

            if frame_idx > target_T * frame_skip * 5:
                break

        if len(frames) == target_T:
            return torch.stack(frames, dim=0)

        raise RuntimeError(f"VideoReader got {len(frames)}/{target_T} frames")

    except Exception as e:
        print(f"[WARN] VideoReader failed for {video_path}. Falling back to read_video. Error: {e}")

    # fallback if goes wrong(much slower)
    frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")  # (Tv,H,W,C) uint8
    if frames.numel() == 0:
        raise RuntimeError(f"read_video returned 0 frames for {video_path}")

    frames = frames[::frame_skip]

    if frames.shape[0] < end:
        raise RuntimeError(
            f"Decoded too few frames for {video_path}: decoded_sub={frames.shape[0]}, need_end={end} "
            f"(start={start}, end={end}, frame_skip={frame_skip})"
        )

    frames = frames[start:end]

    if frames.shape[0] != target_T:
        raise RuntimeError(
            f"Frame count mismatch reading {video_path}: got {frames.shape[0]}, expected {target_T} "
            f"for slice [{start}:{end}] after frame_skip={frame_skip}"
        )

    return frames


def crop_and_resize_video_uint8(frames_uint8: torch.Tensor, box: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    # crop and resize uint8 video frames. frames_uint8: (T,H,W,C) uint8, box: (4,) int [top,left,hh,ww]
    top, left, hh, ww = box.tolist()

    frames = frames_uint8.permute(0, 3, 1, 2)  # (T,C,H,W)
    frames = frames[:, :, top:top + hh, left:left + ww]
    frames = F.resize(frames, [out_size, out_size], antialias=False)
    frames = frames.to(torch.float32) / 255.0
    return frames


def load_videos_from_meta(metas, root_dir="../", out_size=224):
    videos = []
    kept_indices = []
    not_found = []

    for i, meta_item in enumerate(metas):
        subject = meta_item["subject"]
        action = meta_item["action"]
        cam = meta_item["cam"]
        box = torch.as_tensor(meta_item["box"], dtype=torch.int64)

        start = int(meta_item["start"])
        end = int(meta_item["end"])
        frame_skip = int(meta_item.get("frame_skip", 2))

        video_path = os.path.join(
            root_dir,
            f"S{subject}",
            action,
            cam,
            f"S{subject}_{action}_{cam}.mp4"
        )

        if not os.path.exists(video_path):
            print(f"[WARN] Video file not found: {video_path}")
            not_found.append(i)
            continue

        try:
            frames_uint8 = read_clip_uint8(video_path, start, end, frame_skip)
            video_crop = crop_and_resize_video_uint8(frames_uint8, box, out_size=out_size)

            videos.append(video_crop)
            kept_indices.append(i)

        except Exception as e:
            print(f"[WARN] Skipping clip {i} due to error in {video_path}\n       Error: {e}")
            not_found.append(i)

    if len(videos) == 0:
        raise RuntimeError("No videos could be loaded from meta.")

    videos = torch.stack(videos, dim=0)  # (B,T,3,H,W)
    return videos, kept_indices, not_found