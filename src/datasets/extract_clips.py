import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Human36MPreprocessedClips
"""
This script extracts clips from the preprocessed Human36M dataset and saves them as compressed .npz files. Each .npz file will contain:
 - video_u8: The video clip as a uint8 array (shape: [T, 3, 224, 224])
 - joints3d: The 3D joint positions (shape: [T, num_joints, 3])
 - joints2d: The 2D joint positions (shape: [T, num_joints, 2])
 - K: The camera intrinsics (shape: [3, 3] or [T, 3, 3])
 - box: The bounding box (if available, shape: [T, 4] or [4])
 - metadata: subject, action, cam, start frame, end frame

This was the easiest way I found to provide the video clips to the TensorFlow ResNet used for feature extraction. Although the feature extraction model is implemented in TensorFlow, the fastest way to load and preprocess the video data was using PyTorch, so the clips are first prepared in this format and then passed to the TensorFlow pipeline.

Coded by Luisa Ferreira, 2026 with assistance of ChatGPT 5.2 (OpenAI).
"""


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1,5,6,7,8,9,11])
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ds = Human36MPreprocessedClips(
        root=args.root,
        subjects=args.subjects,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        stride=args.stride,
        max_clips=None,
        augment=args.augment,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    total = len(ds)

    print("===================================")
    print("Starting clip export")
    print(f"Dataset clips : {total}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Workers       : {args.num_workers}")
    print("Saving video as uint8")
    print("===================================")

    t0 = time.time()
    i = 0

    for batch in loader:

        if len(batch) == 5:
            video, joints3d, joints2d, K, box = batch
        else:
            video, joints3d, joints2d, K, box, _meta = batch

        B, T, C, H, W = video.shape

        # convert to uint8
        video_u8 = (video.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        video_u8 = video_u8.cpu().numpy()

        joints3d_np = joints3d.cpu().numpy().astype(np.float32)
        joints2d_np = joints2d.cpu().numpy().astype(np.float32)
        K_np = K.cpu().numpy().astype(np.float32) if torch.is_tensor(K) else K
        box_np = box.cpu().numpy().astype(np.float32) if (box is not None and torch.is_tensor(box)) else None

        for b in range(B):

            clip = ds.index[i]
            i += 1

            rel_dir = Path(f"S{clip.subject}") / clip.action / f"{clip.cam}"
            save_dir = out_root / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"clip_{clip.start}_{clip.end}_fs{args.frame_skip}_len{args.seq_len}.npz"

            np.savez_compressed(
                save_path,
                video_u8=video_u8[b],   # uint8
                joints3d=joints3d_np[b],
                joints2d=joints2d_np[b],
                K=K_np[b] if K_np.ndim >= 3 else K_np,
                box=box_np[b] if box_np is not None else None,
                subject=clip.subject,
                action=str(clip.action),
                cam=str(clip.cam),
                start=clip.start,
                end=clip.end,
            )

            # progress prints
            if i % 200 == 0 or i == total:

                elapsed = time.time() - t0
                clips_per_sec = i / elapsed

                pct = 100.0 * i / total

                remaining = (total - i) / clips_per_sec if clips_per_sec > 0 else 0

                print(
                    f"[{pct:6.2f}%] "
                    f"{i}/{total} clips | "
                    f"{clips_per_sec:.2f} clips/s | "
                    f"ETA {remaining/60:.1f} min"
                )

    total_time = time.time() - t0

    print("===================================")
    print("Export finished")
    print(f"Total clips : {i}")
    print(f"Time        : {total_time/60:.2f} min")
    print(f"Speed       : {i/total_time:.2f} clips/s")
    print(f"Saved to    : {out_root}")
    print("===================================")


if __name__ == "__main__":
    main()