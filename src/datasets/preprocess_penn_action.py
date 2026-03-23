import os
import glob
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from queue import Queue
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image

from scipy.io import loadmat

""""
Script to preprocess PennAction dataset by extracting per-clip ResNet50 features and mapping 2D joints to H36M format.

ATTENTION: did not use in final project since I got nice results when using the resnetv50 pre-trained with HMR weights.

Coded by Luisa Ferreira, 2026
"""


# -----------------------------
# Async writer 
# -----------------------------
class AsyncFileWriter:
    def __init__(self, max_queue_size=200):
        self.queue = Queue(maxsize=max_queue_size)
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.count = 0

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            payload, save_path = item
            torch.save(payload, save_path, _use_new_zipfile_serialization=False)
            self.queue.task_done()

    def save(self, payload, save_path):
        self.queue.put((payload, save_path))
        self.count += 1

    def wait(self):
        self.queue.join()

    def stop(self):
        self.queue.put(None)
        self.thread.join()


# -----------------------------
# PennAction joints (13) -> H36M(17) mapping
# Penn indices 0..12:
# 0 head
# 1 l_shoulder, 2 r_shoulder
# 3 l_elbow,    4 r_elbow
# 5 l_wrist,    6 r_wrist
# 7 l_hip,      8 r_hip
# 9 l_knee,    10 r_knee
# 11 l_ankle,  12 r_ankle
#
# H36M-17 indices:
# 0 pelvis/root
# 1 r_hip,2 r_knee,3 r_ankle
# 4 l_hip,5 l_knee,6 l_ankle
# 7 spine,8 thorax,9 neck,10 head
# 11 l_shoulder,12 l_elbow,13 l_wrist
# 14 r_shoulder,15 r_elbow,16 r_wrist
# -----------------------------
PENN_TO_H36M = {
    0: 10,  # head -> head
    1: 11,  # l_shoulder
    2: 14,  # r_shoulder
    3: 12,  # l_elbow
    4: 15,  # r_elbow
    5: 13,  # l_wrist
    6: 16,  # r_wrist
    7: 4,   # l_hip
    8: 1,   # r_hip
    9: 5,   # l_knee
    10: 2,  # r_knee
    11: 6,  # l_ankle
    12: 3,  # r_ankle
}


def _mat_struct_to_dict(m) -> Dict:
    if isinstance(m, dict):
        return m
    if hasattr(m, "_fieldnames"):  # mat_struct
        return {f: getattr(m, f) for f in m._fieldnames}
    if hasattr(m, "dtype") and m.dtype.names is not None:
        out = {}
        for name in m.dtype.names:
            out[name] = m[name].item() if np.size(m[name]) == 1 else m[name]
        return out
    raise ValueError("Unsupported .mat struct format for annotation.")


def load_annotation(mat_path: str) -> Dict:
    d = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    keys = [k for k in d.keys() if not k.startswith("__")]

    expected = {"action","pose","x","y","visibility","train","bbox","dimensions","nframes"}
    if set(keys).issuperset(expected):
        return {k: d[k] for k in keys}

    raise KeyError(f"No usable annotation format in {mat_path}. Keys: {keys}")


def bbox_to_xyxy(b: np.ndarray) -> np.ndarray:
    # pennAction bbox format is inconsistent across videos: some are (x1,y1,x2,y2) while others are (x,y,w,h).
    b = b.astype(np.float32)
    xyxy_like = np.mean((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])) > 0.9
    if xyxy_like:
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    else:
        x1, y1, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x2, y2 = x1 + w, y1 + h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def square_bbox_xyxy(bxyxy: np.ndarray, img_w: int, img_h: int, margin: float) -> np.ndarray:
    # square box around subject with margin, clipped to image boundaries
    x1, y1, x2, y2 = bxyxy.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * margin)

    x1 = cx - 0.5 * side
    x2 = cx + 0.5 * side
    y1 = cy - 0.5 * side
    y2 = cy + 0.5 * side

    x1 = float(np.clip(x1, 0, img_w - 1))
    y1 = float(np.clip(y1, 0, img_h - 1))
    x2 = float(np.clip(x2, 1, img_w))
    y2 = float(np.clip(y2, 1, img_h))

    if x2 <= x1 + 1:
        x2 = min(img_w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(img_h, y1 + 2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def penn_to_h36m_2d(x: np.ndarray, y: np.ndarray, vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 2d penn action joints mapped to h36m, returning visibility vector as well(for 17 joints in h36m format)
    Tn = x.shape[0]
    j2d = np.zeros((Tn, 17, 2), dtype=np.float32)
    v17 = np.zeros((Tn, 17), dtype=np.float32)
    for pj, hj in PENN_TO_H36M.items():
        j2d[:, hj, 0] = x[:, pj].astype(np.float32)
        j2d[:, hj, 1] = y[:, pj].astype(np.float32)
        v17[:, hj] = vis[:, pj].astype(np.float32)
    return j2d, v17


def joints_to_crop_coords(j2d: np.ndarray, vis: np.ndarray, bbox_xyxy: np.ndarray, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    # correct joints to inside bbox crop
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)

    out = j2d.copy()
    out[:, 0] = (out[:, 0] - x1) * (out_size / bw)
    out[:, 1] = (out[:, 1] - y1) * (out_size / bh)

    out[vis < 0.5] = 0.0
    return out.astype(np.float32), vis.astype(np.float32)


@dataclass
class ClipIndex:
    seq_id: str
    split: str              # train/test
    action: str
    pose: str
    start: int              # frame index in [0..T-1]
    end: int                # inclusive end
    frame_skip: int
    seq_len: int


class PennActionPreprocessedClips(Dataset):
    def __init__(
        self,
        root: str,
        seq_len: int = 40,
        frame_skip: int = 2,
        stride: int = 5,
        img_size: int = 224,
        bbox_margin: float = 0.15,
        only_split: str = "",  # "", "train", or "test"
        max_clips: Optional[int] = None,
    ):
        self.root = root
        self.frames_root = os.path.join(root, "frames")
        self.labels_root = os.path.join(root, "labels")
        self.seq_len = int(seq_len)
        self.frame_skip = int(frame_skip)
        self.stride = int(stride)
        self.img_size = int(img_size)
        self.bbox_margin = float(bbox_margin)
        self.only_split = only_split

        mats = sorted(glob.glob(os.path.join(self.labels_root, "*.mat")))
        if len(mats) == 0:
            raise RuntimeError(f"No .mat files found in {self.labels_root}")

        self._seq_cache = {}  # optional small cache (paths/ann)
        self.index: List[ClipIndex] = []

        for mat_path in mats:
            seq_id = os.path.splitext(os.path.basename(mat_path))[0]
            ann = load_annotation(mat_path)

            train_flag = int(np.array(ann.get("train", 1)).item())
            split = "train" if train_flag == 1 else "test"
            if self.only_split and split != self.only_split:
                continue

            action = str(ann.get("action", "unknown"))
            pose = str(ann.get("pose", "unknown"))

            x = np.array(ann["x"], dtype=np.float32)
            Tn = x.shape[0]

            # clip start positions in original frame index space
            # we sample frames: start, start+fs, ..., start+(seq_len-1)*fs
            max_start = Tn - (self.seq_len - 1) * self.frame_skip - 1
            if max_start < 0:
                continue

            starts = list(range(0, max_start + 1, self.stride))
            for s in starts:
                e = s + (self.seq_len - 1) * self.frame_skip
                self.index.append(ClipIndex(
                    seq_id=seq_id,
                    split=split,
                    action=action,
                    pose=pose,
                    start=s,
                    end=e,
                    frame_skip=self.frame_skip,
                    seq_len=self.seq_len,
                ))

            # cache mat path for lazy load
            self._seq_cache[seq_id] = {"mat_path": mat_path}

            if max_clips is not None and len(self.index) >= max_clips:
                self.index = self.index[:max_clips]
                break

        # ImageNet normalization (match your ResNet extraction)
        # 20.03: ATTENTION: I only used this in first experiments when I was still using pytorch resnet pretrained in imagenet.
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.index)

    def _load_seq_data(self, seq_id: str):
        # load annotation
        mat_path = self._seq_cache[seq_id]["mat_path"]
        ann = load_annotation(mat_path)

        x = np.array(ann["x"], dtype=np.float32)             # (T,13)
        y = np.array(ann["y"], dtype=np.float32)             # (T,13)
        vis = np.array(ann["visibility"], dtype=np.float32)  # (T,13)
        bbox = np.array(ann["bbox"], dtype=np.float32)       # (T,4)
        bbox_xyxy = bbox_to_xyxy(bbox)                       # (T,4)

        joints2d_17, vis_17 = penn_to_h36m_2d(x, y, vis)      # (T,17,2), (T,17)

        # frame paths
        seq_dir = os.path.join(self.frames_root, seq_id)
        frame_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
        if len(frame_paths) < x.shape[0]:
            raise RuntimeError(f"{seq_id}: found {len(frame_paths)} frames but expected >= {x.shape[0]} in {seq_dir}")
        frame_paths = frame_paths[:x.shape[0]]

        return frame_paths, joints2d_17, vis_17, bbox_xyxy

    def __getitem__(self, idx: int):
        clip = self.index[idx]
        frame_paths, joints2d_17, vis_17, bbox_xyxy = self._load_seq_data(clip.seq_id)

        # select frames for this clip
        frame_idxs = [clip.start + k * clip.frame_skip for k in range(clip.seq_len)]

        video = torch.empty((clip.seq_len, 3, self.img_size, self.img_size), dtype=torch.float32)
        j2d_out = np.zeros((clip.seq_len, 17, 2), dtype=np.float32)
        v2d_out = np.zeros((clip.seq_len, 17), dtype=np.float32)
        box_out = np.zeros((clip.seq_len, 4), dtype=np.float32)

        for t, fi in enumerate(frame_idxs):
            img = Image.open(frame_paths[fi]).convert("RGB")
            w, h = img.size

            b = square_bbox_xyxy(bbox_xyxy[fi], w, h, margin=self.bbox_margin)
            box_out[t] = b

            # crop & resize
            x1, y1, x2, y2 = b.tolist()
            crop = img.crop((x1, y1, x2, y2)).resize((self.img_size, self.img_size), Image.BILINEAR)

            # to tensor + normalize
            ten = TF.to_tensor(crop)  # (3,H,W) in [0,1]
            ten = (ten - self.mean) / self.std
            video[t] = ten

            # joints -> crop coords
            j2d_crop, v_crop = joints_to_crop_coords(joints2d_17[fi], vis_17[fi], b, self.img_size)
            j2d_out[t] = j2d_crop
            v2d_out[t] = v_crop

        meta = {
            "seq": clip.seq_id,
            "split": clip.split,
            "action": clip.action,
            "pose": clip.pose,
            "start": clip.start,
            "end": clip.end,
            "frame_skip": clip.frame_skip,
            "seq_len": clip.seq_len,
            "img_size": self.img_size,
            "bbox_margin": self.bbox_margin,
            "source": "PennAction",
        }

        return video, torch.from_numpy(j2d_out), torch.from_numpy(v2d_out), torch.from_numpy(box_out), meta


def _meta_at(meta_batch, b: int):
    out = {}
    for k, v in meta_batch.items():
        if torch.is_tensor(v):
            vb = v[b]
            out[k] = vb.item() if vb.numel() == 1 else vb
        else:
            out[k] = v[b]
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Precompute per-clip ResNet50 features for PennAction")
    parser.add_argument("--root", type=str, required=True, help="PennAction root containing frames/ and labels/")
    parser.add_argument("--out", type=str, required=True, help="Output directory for cached features")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--bbox-margin", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-fp16", action="store_true")
    parser.add_argument("--only-split", type=str, default="", help="train or test (optional)")
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_root = Path(args.out)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    ds = PennActionPreprocessedClips(
        root=args.root,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=args.stride,
        img_size=args.img_size,
        bbox_margin=args.bbox_margin,
        only_split=args.only_split,
        max_clips=args.max_clips,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        prefetch_factor=2,
        # persistent_workers=True,
    )

    # ResNet50 backbone (avgpool)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

    use_dp = device.startswith("cuda") and (torch.cuda.device_count() > 1)
    if use_dp:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        backbone = nn.DataParallel(backbone)

    writer = AsyncFileWriter(max_queue_size=300)

    print(f"Device: {device}")
    print(f"Processing {len(ds)} clips...")
    t_all = time.time()
    global_i = 0
    t_last = time.time()

    # Warmup (optional)
    if len(ds) > 0:
        warm = next(iter(loader))
        warm_video = warm[0].to(device, non_blocking=True)
        B, T, C, H, W = warm_video.shape
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            enabled=device.startswith("cuda"),
        ):
            _ = backbone(warm_video.view(B * T, C, H, W).contiguous())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        print("✓ Warmup complete")

    for it, batch in enumerate(loader):
        video, joints2d, vis2d, box, meta = batch
        B, T, C, H, W = video.shape
        video = video.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            enabled=device.startswith("cuda"),
        ):
            x = video.view(B * T, C, H, W).contiguous()
            feats = backbone(x).flatten(1)          # (B*T,2048)
            feats = feats.view(B, T, -1)            # (B,T,2048)

        feats_to_save = feats.half() if args.save_fp16 else feats.float()

        for b in range(B):
            clip = ds.index[global_i]
            global_i += 1

            rel_dir = Path(clip.split) / clip.seq_id
            save_dir = out_root / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"clip_{clip.start}_{clip.end}_fs{clip.frame_skip}_len{clip.seq_len}.pt"

            payload = {
                "feats": feats_to_save[b].cpu(),          # (T,2048)
                "joints2d": joints2d[b].cpu(),            # (T,17,2) in crop coords
                "vis2d": vis2d[b].cpu(),                  # (T,17)
                "box": box[b].cpu(),                      # (T,4) original coords bbox (square+margin)
                "meta": {**_meta_at(meta, b)},
            }

            writer.save(payload, save_path)

        # progress
        if global_i % 200 == 0 or global_i == len(ds):
            dt = time.time() - t_last
            clips_per_sec = 200 / dt if dt > 0 else 0.0
            t_last = time.time()
            progress = 100.0 * global_i / len(ds)
            eta = (len(ds) - global_i) / clips_per_sec if clips_per_sec > 0 else 0.0
            print(f"[{progress:5.1f}%] {global_i:6d}/{len(ds)} clips | "
                  f"{clips_per_sec:6.1f} clips/s | ETA: {eta:6.1f}s")

    print("\nWaiting for file writes to complete...")
    writer.wait()
    writer.stop()

    total_time = time.time() - t_all
    print("-" * 60)
    print(f"✓ Done! Saved {global_i} clips to: {out_root}")
    print(f"✓ Total time: {total_time:.1f}s ({len(ds)/total_time:.1f} clips/s)")


if __name__ == "__main__":
    main()