import os
import glob
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.io import VideoReader

"""
This module defines a PyTorch Dataset for loading fixed-length clips from the Human3.6M dataset.
It handles video reading, 2D/3D joint loading, cropping, resizing, and camera parameter adjustment.
Input will be: 
video: (T,3,224,224) float32 in [0,1] cropped and centered on the subject
joints3d: (T,17,3) ; T: number of frames in the clip
joints2d: (T,17,2) in pixel coords of the (cropped/resized) frames
K: (3,3) camera intrinsics adjusted for the cropped/resized frames
box: (4,) [top, left, height, width] of the crop applied to the original frames

Coded by Luisa Ferreira, 2026.
"""
@dataclass # used for creating simple classes to hold clip metadata, like video path, gt path, subject, action, cam, start/end frames, etc, without needing to write boilerplate code for init, repr, etc.
class ClipIndex:
    video_path: str
    gt_path: str
    subject: str
    action: str
    cam: str
    cam_params: dict
    start: int
    end: int  # exclusive
    video_idx: int = 0


def _load_poses(gt_path): 
    with open(gt_path, "rb") as f: 
        data = pickle.load(f) 
    j3d = torch.from_numpy(np.asarray(data["3d"])).float() 
    j2d = torch.from_numpy(np.asarray(data["2d"])).float() 
    return j3d, j2d


def _load_camera_params(cam_path):
    with open(cam_path, "rb") as f:
        data = pickle.load(f)
    return data


def _compute_square_crop_from_2d(joints2d, img_h, img_w, scale=1.6):
    pts = joints2d.reshape(-1, 2)

    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    w = (x_max - x_min).clamp(min=1.0)
    h = (y_max - y_min).clamp(min=1.0)

    side = scale * torch.max(w, h)

    left = cx - 0.5 * side
    top = cy - 0.5 * side

    max_left = torch.tensor(float(img_w), device=left.device) - side
    max_top  = torch.tensor(float(img_h), device=top.device) - side
    left = left.clamp(0.0, max_left.item())
    top  = top.clamp(0.0, max_top.item())

    left_i = int(torch.round(left).item())
    top_i = int(torch.round(top).item())
    side_i = int(torch.round(side).item())

    side_i = max(1, min(side_i, img_w - left_i, img_h - top_i))
    return torch.tensor([top_i, left_i, side_i, side_i], dtype=torch.int64)


def _adjust_joints2d_after_crop_and_resize(joints2d, box, out_size=224):
    top, left, hh, ww = box.tolist()
    scale_x = out_size / float(ww)
    scale_y = out_size / float(hh)

    joints2d_cropped = joints2d.clone()
    joints2d_cropped[..., 0] = (joints2d[..., 0] - float(left)) * scale_x
    joints2d_cropped[..., 1] = (joints2d[..., 1] - float(top)) * scale_y
    return joints2d_cropped


def _adjust_camera_after_crop_and_resize(cam_params, box, out_size=224):
    top, left, hh, ww = box.tolist()
    sx = out_size / float(ww)
    sy = out_size / float(hh)

    new_cam = dict(cam_params)

    f = np.asarray(new_cam["f"], dtype=np.float32).reshape(2)
    c = np.asarray(new_cam["c"], dtype=np.float32).reshape(2)

    c_new = np.array([(c[0] - float(left)) * sx, (c[1] - float(top)) * sy], dtype=np.float32)
    f_new = np.array([f[0] * sx, f[1] * sy], dtype=np.float32)

    K = np.array(
        [[f_new[0], 0.0,     c_new[0]],
         [0.0,     f_new[1], c_new[1]],
         [0.0,     0.0,      1.0]],
        dtype=np.float32
    )
    K = torch.from_numpy(K)
    return K


def _crop_and_resize_video_uint8(frames_uint8, box, out_size=224):
    top, left, hh, ww = box.tolist()

    frames = frames_uint8.permute(0, 3, 1, 2)
    frames = frames[:, :, top:top + hh, left:left + ww]

    frames = F.resize(frames, [out_size, out_size], antialias=False)

    frames = frames.to(torch.float32) / 255.0
    return frames


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------
# In one of my experiments I tried adding these augmentations. But end it up doing the augmentation within the features to avoid overdata.
# def _aug_hflip(video, joints3d, joints2d, K):
#     # horizontal flip + fix joints and K
#     # if torch.rand(1).item() > p:
#     #     return video, joints2d, joints3d, K

#     W = video.shape[-1]  # 224

#     # --- video ---
#     video = torch.flip(video, dims=[-1])

#     # --- joints2d: mirror x ---
#     joints2d = joints2d.clone()
#     joints2d[..., 0] = W - joints2d[..., 0]

#     # --- joints3d: negate x (camera space convention) ---
#     joints3d = joints3d.clone()
#     joints3d[..., 0] = -joints3d[..., 0]

#     # --- swap left/right joint pairs ---
#     for l_idx, r_idx in H36M_FLIP_PAIRS:
#         joints2d[:, [l_idx, r_idx]] = joints2d[:, [r_idx, l_idx]]
#         joints3d[:, [l_idx, r_idx]] = joints3d[:, [r_idx, l_idx]]

#     # --- K: principal point cx mirrors to W - cx ---
#     K = K.clone()
#     K[0, 2] = W - K[0, 2]

#     return video, joints3d, joints2d, K


# def _aug_color_jitter(video):
#     # Color jitter can help the model be more robust to variations in lighting and appearance. It randomly changes the brightness, contrast, saturation, and hue of the video frames, which can help prevent overfitting to specific lighting conditions or colors in the training data. This augmentation encourages the model to learn more general features that are invariant to such changes, improving its ability to generalize to new videos with different lighting and color characteristics.

#     jitter = T.ColorJitter(
#         brightness=0.3,
#         contrast=0.3,
#         saturation=0.2,
#         hue=0.05,
#     )
#     # torchvision v2 handles (T, C, H, W) batches natively
#     return jitter(video)


# def _aug_temporal_reverse(video, joints3d, joints2d):
#     # Reverse the temporal order of the clip, which can help the model learn to predict in both forward and backward time directions, improving its robustness and generalization to different motion patterns. When we reverse the video frames, we also need to reverse the order of the joints2d and joints3d sequences so that they still correspond correctly to each frame.

#     video    = torch.flip(video,    dims=[0])
#     joints2d = torch.flip(joints2d, dims=[0])
#     joints3d = torch.flip(joints3d, dims=[0])
#     return video, joints3d, joints2d

class Human36MPreprocessedClips(Dataset):
    def __init__(
        self,
        root: str,
        subjects: List[int],
        seq_len: int = 40,
        stride: int = 10,
        frame_skip: int = 2,
        cams: Optional[List[int]] = None,
        resize: int = 224,
        crop_scale: float = 1.6,
        max_clips: Optional[int] = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        trev_prob: float = 0.5,
    ):
        super().__init__()
        self.root = root
        self.subjects = subjects
        self.seq_len = seq_len
        self.stride = stride
        self.frame_skip = frame_skip
        self.resize = resize
        self.crop_scale = crop_scale
        self.augment = augment
        # normalization for resnet imagenet
        # self.frame_tf = T.Compose([
        #     T.Normalize(mean=(0.485, 0.456, 0.406),
        #                 std=(0.229, 0.224, 0.225)),
        # ])

        self.index: List[ClipIndex] = []
        self._gt_cache = {}
        self._cam_cache = {}
        
        # NEW: Track which clips belong to which video for potential batching
        self._video_to_clips = defaultdict(list)
        video_counter = 0

        for s in subjects:
            subj_dir = os.path.join(root, f"S{s}")
            actions = sorted(
                a for a in os.listdir(subj_dir)
                if os.path.isdir(os.path.join(subj_dir, a))
            )

            for action in actions:
                action_dir = os.path.join(subj_dir, action)
                cam_dirs = sorted(glob.glob(os.path.join(action_dir, "cam_*")))

                for cam_dir in cam_dirs:
                    cam_name = os.path.basename(cam_dir)
                    cam_id = int(cam_name.replace("cam_", ""))
                    if cams is not None and cam_id not in cams:
                        continue

                    mp4s = glob.glob(os.path.join(cam_dir, "*.mp4"))
                    gt_path = os.path.join(cam_dir, "gt_poses.pkl")
                    cam_path = os.path.join(cam_dir, "camera_wext.pkl")
                    if not mp4s or not os.path.isfile(gt_path) or not os.path.isfile(cam_path):
                        continue

                    video_path = mp4s[0]

                    if gt_path not in self._gt_cache:
                        self._gt_cache[gt_path] = _load_poses(gt_path)
                    joints3d_all, _ = self._gt_cache[gt_path]
                    n_frames = int(joints3d_all.shape[0])

                    n_frames_sub = (n_frames + self.frame_skip - 1) // self.frame_skip

                    if cam_path not in self._cam_cache:
                        self._cam_cache[cam_path] = _load_camera_params(cam_path)
                    cam_params = self._cam_cache[cam_path]

                    for start in range(0, n_frames_sub - seq_len + 1, stride):
                        clip_idx = ClipIndex(
                            video_path=video_path,
                            gt_path=gt_path,
                            subject=s,
                            action=action,
                            cam=cam_name,
                            cam_params=cam_params,
                            start=start,
                            end=start + seq_len,
                            video_idx=video_counter,
                        )
                        self.index.append(clip_idx)
                        self._video_to_clips[video_path].append(len(self.index) - 1)
                        
                        if max_clips is not None and len(self.index) >= max_clips:
                            break
                    
                    video_counter += 1
                    
                    if max_clips is not None and len(self.index) >= max_clips:
                        break
                if max_clips is not None and len(self.index) >= max_clips:
                    break
            if max_clips is not None and len(self.index) >= max_clips:
                break

        if len(self.index) == 0:
            raise RuntimeError(f"No clips found under root={root}. Check your folder structure and files.")

    def __len__(self) -> int:
        return len(self.index)

    def _read_video_uint8_clip_fast(self, video_path, start, end):
        try:
            # nn.VideoReader is much faster than torchvision.io.read_video, but can be less robust on some files.
            reader = VideoReader(video_path, "video")
            metadata = reader.get_metadata()
            fps = metadata['video']['fps'][0]
            
            # calculate the timestamp to seek to, based on the start frame and frame_skip
            start_time = (start * self.frame_skip) / fps
            
            reader.seek(start_time)
            
            frames = []
            target_frames = end - start
            frame_idx = 0
            
            # read frames with skip
            for frame in reader:
                if frame_idx % self.frame_skip == 0:
                    frame_data = frame['data']        # (C,H,W)
                    frame_data = frame_data.permute(1, 2, 0)  # (H,W,C) ← FIXED!
                    frames.append(frame_data)    
                    if len(frames) >= target_frames:
                        break
                frame_idx += 1
                
                # Safety: don't read too far
                if frame_idx > target_frames * self.frame_skip * 2:
                    break
            
            if len(frames) < target_frames:
                # fallback to old method if we couldn't read enough frames
                return self._read_video_uint8_clip(video_path, start, end)
            
            return torch.stack(frames[:target_frames])
            
        except Exception as e:
            # fallback to old method torchvision.io.read_video which is more robust but slower
            print("VideoReader failed for {}, falling back to legacy method. Error: {}".format(video_path, e))
            return self._read_video_uint8_clip(video_path, start, end)

    def _read_video_uint8_clip(self, video_path, start, end):
        # read video using torchvision, which is more robust but much slower than nn.VideoReader
        frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        frames = frames[::self.frame_skip]
        frames = frames[start:end]
        
        if frames.shape[0] != (end - start):
            raise RuntimeError(
                f"Frame count mismatch reading {video_path}: "
                f"got {frames.shape[0]}, expected {end-start} for slice [{start}:{end}]."
            )
        return frames

    def __getitem__(self, idx):
        ci = self.index[idx]

        frames_uint8 = self._read_video_uint8_clip_fast(ci.video_path, ci.start, ci.end)
        
        Tt, H, W, C = frames_uint8.shape
        assert C == 3

        joints3d_all, joints2d_all = self._gt_cache[ci.gt_path]
        orig_idx = torch.arange(ci.start, ci.end, dtype=torch.long) * self.frame_skip

        if int(orig_idx[-1]) >= joints3d_all.shape[0]:
            raise RuntimeError(
                f"Joint index out of range for {ci.gt_path}: "
                f"max orig_idx={int(orig_idx[-1])}, n_frames={joints3d_all.shape[0]}"
            )

        joints3d = joints3d_all[orig_idx]
        joints2d = joints2d_all[orig_idx]
        
        assert frames_uint8.shape[0] == joints3d.shape[0], (
            f"Mismatch T: video {frames_uint8.shape[0]} vs joints {joints3d.shape[0]}"
        )
    
        # compute box that tightly crops the person in the clip, so the image can be centered on the subject 
        box = _compute_square_crop_from_2d(
            joints2d=joints2d,
            img_h=H,
            img_w=W,
            scale=self.crop_scale,
        )

        # compute crop on video frames, adjust joints2d and camera intrinsics accordingly
        video = _crop_and_resize_video_uint8(frames_uint8, box, out_size=self.resize)
        joints2d = _adjust_joints2d_after_crop_and_resize(joints2d=joints2d, box=box, out_size=self.resize)
        K = _adjust_camera_after_crop_and_resize(ci.cam_params, box=box, out_size=self.resize)


        # # normalize video for ResNet imagenet pretrained weights
        # video = self.frame_tf(video)

        return video, joints3d, joints2d, K, box