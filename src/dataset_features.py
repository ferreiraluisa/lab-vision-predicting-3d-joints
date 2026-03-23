import os
import glob
from typing import List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

"""
This defines a PyTorch Dataset for loading the pre-extracted ResNet features from the Human3.6M dataset. Each item is a fixed-length clip of features, along with the corresponding 2D/3D joints and camera parameters.

Coded by Luísa Ferreira.
"""


class Human36MFeatureClips(Dataset):
    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        max_clips: Optional[int] = None,
        test_set: bool = False,
        augment: bool = False,
        feat_noise_std: float = 0.01,
        frame_drop_prob: float = 0.05,
        time_reverse_prob: float = 0.0,
    ):
        self.root = root
        self.test_set = test_set

        # augmentation flags
        self.augment = augment and (not test_set)
        self.feat_noise_std = feat_noise_std
        self.frame_drop_prob = frame_drop_prob
        self.time_reverse_prob = time_reverse_prob

        # pattern = os.path.join(root, "S*", "*", "cam_*", "clip_*.pt")
        pattern = os.path.join(root, "S*", "*", "cam_*", "clip_*.npz")
        files = sorted(glob.glob(pattern))

        if subjects is not None:
            keep = []
            subj_set = set(subjects)
            for p in files:
                parts = p.split(os.sep)
                s_part = [x for x in parts if x.startswith("S")]
                if len(s_part) == 0:
                    continue
                s = int(s_part[0].replace("S", ""))
                if s in subj_set:
                    keep.append(p)
            files = keep

        if max_clips is not None:
            files = files[:max_clips]

        if len(files) == 0:
            raise RuntimeError(f"No cached clips found under {root}")

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):

        d = np.load(self.files[idx])

        feats = torch.from_numpy(d["feats"]).float()
        joints3d = torch.from_numpy(d["joints3d"]).float() / 1000.0
        joints2d = torch.from_numpy(d["joints2d"]).float()
        K = torch.from_numpy(d["K"]).float()
        # d = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        # feats = d["feats"]
        # joints3d = d["joints3d"] / 1000.0  # convert from mm to m
        # joints2d = d["joints2d"]

        # K = d["K"]

        if self.augment:
            # 1) small gaussian noise in feature space
            if self.feat_noise_std > 0:
                feats = feats + torch.randn_like(feats) * self.feat_noise_std

            # 2) randomly drop entire timesteps
            if self.frame_drop_prob > 0:
                keep_mask = (torch.rand(feats.shape[0], 1) > self.frame_drop_prob).float()
                feats = feats * keep_mask

            # 3) optional time reverse
            if self.time_reverse_prob > 0 and torch.rand(1).item() < self.time_reverse_prob:
                feats = torch.flip(feats, dims=[0])
                joints3d_norm = torch.flip(joints3d_norm, dims=[0])
                joints2d = torch.flip(joints2d, dims=[0])

        root = joints3d[:, 0:1, :]
        joints3d_norm = joints3d - root

        meta = {
            "box": torch.from_numpy(d["box"]).float(),
            "subject": int(d["subject"].item()),
            "action": str(d["action"].item()),
            "cam": str(d["cam"].item()),
            "start": int(d["start"].item()),
            "end": int(d["end"].item()),
        }
        # meta = d.get("meta", None)

        if self.test_set:
            return feats, joints3d_norm, joints2d, K, meta

        return feats, joints3d_norm, joints2d, K