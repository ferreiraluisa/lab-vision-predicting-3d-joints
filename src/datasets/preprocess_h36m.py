import os
import argparse
from pathlib import Path
import time
from queue import Queue
from threading import Thread

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import Human36MPreprocessedClips

"""
This script precomputes ResNet50 features for Human3.6M video clips and saves them to disk.
Each clip's features, along with 2D/3D joints and camera intrinsics, are stored in a structured directory format.
This allows for efficient loading during model training or evaluation. Avoiding redundant computation of ResNet features speeds up experiments significantly.

Coded by Luísa Ferreira, 2026

Used Claude to help optimize and refactor the code for better performance and readability.
(older version was taking too long(5 hours to process only 6k/200k clips), this version processes all ~20k clips in ~30 minutes on a single GPU).


ATTENTION: I ended up using extract_clips.py + tf_resnet_features.py to extract features using the original HMR ResNetV2-50 TF1 checkpoint, since it gave better results than the pytorch resnet pretrained on imagenet. So this script is not used in the final project, but I keep it here for reference since it was my first attempt at preprocessing the dataset.
"""


class AsyncFileWriter:
    # asynchronous file writer using a background thread and a queue, to speed up saving features to disk without blocking the main processing loop
    # this helped a lot!!!!!!!!
    def __init__(self, max_queue_size=100):
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

def _meta_at(meta_batch, b: int):
    # meta_batch is a dict produced by default_collate: {key: list/tensor of length B}
    out = {}
    for k, v in meta_batch.items():
        if torch.is_tensor(v):
            # tensor with shape (B, ...) -> select b
            vb = v[b]
            out[k] = vb.item() if vb.numel() == 1 else vb
        else:
            # list/tuple -> select b
            out[k] = v[b]
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("OPTIMIZED: Precompute per-clip ResNet50 features for H36M")
    parser.add_argument("--root", type=str, required=True, help="H36M preprocessed root")
    parser.add_argument("--out", type=str, required=True, help="Output directory for cached features")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 5, 6, 7, 8, 9, 11])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-fp16", action="store_true", help="Store feats as float16")
    parser.add_argument("--augment", action="store_true", help="enable online data augmentation")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # to help performance, enable cudnn benchmark and allow TF32 on Ampere+ GPUs, which can speed up ResNet inference significantly with minimal impact on feature quality. Also set pin_memory=True in DataLoader for faster host to GPU transfers.
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPUs available: {torch.cuda.device_count()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        prefetch_factor=2,  
        # persistent_workers=True,  # keep workers alive for the entire epoch, faster than respawning each time
    )

    # ResNet50 backbone
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = backbone.to(device).eval()

    # -------------------------
    # multi-gpu with nn.DataParallel
    # -------------------------
    use_dp = device.startswith("cuda") and (torch.cuda.device_count() > 1)
    if use_dp:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        backbone = nn.DataParallel(backbone)

    # Compile for massive speedup (PyTorch 2.0+)
    use_compile = (not use_dp) and device.startswith("cuda")
    if use_compile:
        try:
            backbone = torch.compile(backbone, mode="max-autotune")
            print("Using torch.compile(backbone)")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing without compilation")

    # async writer
    writer = AsyncFileWriter() 

    global_i = 0
    t_all = time.time()
    t_last = time.time()
    
    print("Warming up compiled model...")
    warmup_batch = next(iter(loader))
    warmup_video = warmup_batch[0].to(device, non_blocking=True)
    B, T, C, H, W = warmup_video.shape
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", 
                        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32):
        _ = backbone(warmup_video.view(B * T, C, H, W).contiguous())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print("✓ Warmup complete")

    print(f"\nProcessing {len(ds)} clips...")
    print("-" * 60)

    for it, batch in enumerate(loader):
        video, joints3d, joints2d, K, box, meta = batch
        B, T, C, H, W = video.shape

        video = video.to(device, non_blocking=True)

        # forward pass with torch.autocast for mixed precision
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            enabled=device.startswith("cuda"),
        ):
            x = video.view(B * T, C, H, W).contiguous()  # contiguous for better perf
            feats = backbone(x).flatten(1)
            feats = feats.view(B, T, -1)

        feats_to_save = feats.half() if args.save_fp16 else feats.float()

        for b in range(B):
            clip = ds.index[global_i]
            global_i += 1

            rel_dir = Path(f"S{clip.subject}") / clip.action / f"{clip.cam}"
            if rel_dir.exists():
                print(f"Warning: Directory {rel_dir} already exists, skipping clip {clip} to avoid overwriting.")
                continue
            save_dir = out_root / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"clip_{clip.start}_{clip.end}_fs{args.frame_skip}_len{args.seq_len}.pt"

            payload = {
                "feats": feats_to_save[b].cpu(),
                "joints3d": joints3d[b].cpu(),
                "joints2d": joints2d[b].cpu(),
                "K": K[b].cpu() if K.ndim >= 3 else K.cpu(),
            }
            save_meta = {
                "subject": clip.subject,
                "action": clip.action,
                "cam": clip.cam,
                "start": clip.start,
                "end": clip.end,
                "box": box[b].cpu() if box is not None else None,
                "frame_skip": args.frame_skip,  
                "seq_len": args.seq_len,
            }
            save_meta.update(_meta_at(meta, b))  # adds aug flags etc

            payload = {**payload, "meta": save_meta}
            
            if writer:
                writer.save(payload, save_path)
            else:
                torch.save(payload, save_path, _use_new_zipfile_serialization=False)

        # Progress reporting
        if global_i % 100 == 0 or global_i == len(ds):
            dt = time.time() - t_last
            clips_per_sec = 100 / dt if dt > 0 else 0
            t_last = time.time()
            progress = 100 * global_i / len(ds)
            eta = (len(ds) - global_i) / clips_per_sec if clips_per_sec > 0 else 0
            
            print(f"[{progress:5.1f}%] {global_i:5d}/{len(ds)} clips | "
                  f"{clips_per_sec:6.1f} clips/s | ETA: {eta:6.1f}s")

    if writer:
        print("\nWaiting for file writes to complete...")
        writer.wait()
        writer.stop()

    total_time = time.time() - t_all
    print("-" * 60)
    print(f"✓ Done! Saved {global_i} clips to: {out_root}")
    print(f"✓ Total time: {total_time:.1f}s ({len(ds)/total_time:.1f} clips/s)")
    print(f"✓ Average time per clip: {1000*total_time/len(ds):.1f}ms")


if __name__ == "__main__":
    main()