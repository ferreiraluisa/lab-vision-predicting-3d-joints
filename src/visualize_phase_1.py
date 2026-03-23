import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.io import VideoReader
import torchvision.transforms.functional as F
from utils import _as_numpy, to_uint8_rgb, load_videos_from_meta


""""
Script to visualize the 3D pose estimation results from a .npz file containing:
- joints3d: (B,T,J,3) GT 3D joint positions
- predicted3djoints: (B,T,J,3) estimated 3D joint positions
- meta: list of dicts with metadata for each sample, including video paths and frame indices        

Plot videos with the GT vs estimated 3D joints overlaid, saving each frame as an image in a directory.(then use ffmpeg to make a video if desired)

Coded by Luisa Ferreira, 2026.
"""
H36M_EDGES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]



# --------------------------------------------------
# visualization
# --------------------------------------------------
def save_sample_3d_overlay(
    video, joints3d, pred3d,
    sample_idx=0, fps=10, point_size=18, line_width=2,
    save_dir="frames"
):
    """
    Saves each frame as an image.

    Layout:
      1) video frame
      2) single 3D plot with GT 3D and predicted 3D overlaid

    video:    (B,T,3,H,W)
    joints3d: (B,T,J,3)
    pred3d:   (B,T,J,3)
    """
    os.makedirs(save_dir, exist_ok=True)

    video    = _as_numpy(video)
    joints3d = _as_numpy(joints3d)
    pred3d   = _as_numpy(pred3d)

    vid = video[sample_idx]
    js3 = joints3d[sample_idx]
    pr3 = pred3d[sample_idx]

    T = vid.shape[0]
    _, _, H, W = vid.shape

    frames = [to_uint8_rgb(vid[tt]) for tt in range(T)]

    all3 = np.concatenate([js3, pr3], axis=0)
    xs, ys, zs = all3[..., 0], all3[..., 1], all3[..., 2]

    def pad(a, b, p=0.05):
        r = (b - a) if (b > a) else 1.0
        return a - p * r, b + p * r

    xlim = pad(xs.min(), xs.max())
    ylim = pad(ys.min(), ys.max())
    zlim = pad(zs.min(), zs.max())

    for tt in range(T):
        fig = plt.figure(figsize=(11, 4))
        ax_2d = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

        # 2D frame
        ax_2d.set_title(f"Frame {tt}")
        ax_2d.imshow(frames[tt])
        ax_2d.set_xlim(0, W - 1)
        ax_2d.set_ylim(H - 1, 0)
        ax_2d.axis("off")

        # 3D GT + prediction
        ax_3d.set_title("GT 3D and Estimated 3D")
        ax_3d.set_xlim(*xlim)
        ax_3d.set_ylim(*ylim)
        ax_3d.set_zlim(*zlim)
        ax_3d.view_init(elev=-80, azim=-90)

        ax_3d.scatter(
            js3[tt, :, 0], js3[tt, :, 1], js3[tt, :, 2],
            s=point_size, c="steelblue", label="GT"
        )
        for a, b in H36M_EDGES:
            ax_3d.plot(
                [js3[tt, a, 0], js3[tt, b, 0]],
                [js3[tt, a, 1], js3[tt, b, 1]],
                [js3[tt, a, 2], js3[tt, b, 2]],
                linewidth=line_width, c="steelblue"
            )

        ax_3d.scatter(
            pr3[tt, :, 0], pr3[tt, :, 1], pr3[tt, :, 2],
            s=point_size, c="orange", label="Estimated"
        )
        for a, b in H36M_EDGES:
            ax_3d.plot(
                [pr3[tt, a, 0], pr3[tt, b, 0]],
                [pr3[tt, a, 1], pr3[tt, b, 1]],
                [pr3[tt, a, 2], pr3[tt, b, 2]],
                linewidth=line_width, c="orange"
            )

        ax_3d.legend(loc="upper right")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"frame_{tt:04d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    print(f"Saved {T} frames to '{save_dir}'")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    root_dir = "../"  

    data = np.load("result_hmr.npz", allow_pickle=True)

    j3d = data["joints3d"]       
    pred = data["predicted3djoints"]     
    metas = data["meta"] 

    print("Loaded result_hmr.npz")
    print("  j3d:", getattr(j3d, "shape", None))
    print("  pred:", getattr(pred, "shape", None))
    print("  metas:", len(metas))

    video, kept_indices, not_found = load_videos_from_meta(
        metas,
        root_dir=root_dir,
        out_size=224
    )

    j3d = j3d[kept_indices]
    pred = pred[kept_indices]

    print("video shape:", video.shape)
    print("j3d shape:", j3d.shape)
    print("pred shape:", pred.shape)
    print("not_found:", not_found)

    save_sample_3d_overlay(
        video,
        j3d,
        pred,
        sample_idx=15,
        save_dir="frames"
    )


if __name__ == "__main__":
    main()