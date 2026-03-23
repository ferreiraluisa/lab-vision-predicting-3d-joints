import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from utils import _as_numpy, to_uint8_rgb, load_videos_from_meta, mpjpe_per_frame_mm, pa_mpjpe_per_frame_mm, dtw_cost_matrix, dtw_path, compute_similarity_transform

H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


# --------------------------------------------------
# plotting helpers
# --------------------------------------------------
def add_border(ax, color, lw=6):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)


def compute_axis_limits(gt_seq, pred_seq, pad_ratio=0.05):
    all3 = np.concatenate([gt_seq, pred_seq], axis=0)
    xs, ys, zs = all3[..., 0], all3[..., 1], all3[..., 2]

    def pad(a, b, p=pad_ratio):
        r = (b - a) if (b > a) else 1.0
        return a - p * r, b + p * r

    return pad(xs.min(), xs.max()), pad(ys.min(), ys.max()), pad(zs.min(), zs.max())


def fig_to_rgb_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    return buf[..., :3]

def align_pred_to_gt_timeline_with_dtw(pred_future, gt_future):
    """
    pred_future: (F_pred, J, 3)
    gt_future:   (F_gt,   J, 3)

    Returns:
        pred_future_aligned_to_gt: (F_gt, J, 3)
        path: list of (i_pred, j_gt)
    """
    cost = dtw_cost_matrix(pred_future, gt_future)
    path = dtw_path(cost)

    F_gt = gt_future.shape[0]
    aligned = np.zeros_like(gt_future, dtype=np.float64)

    matches_per_gt = [[] for _ in range(F_gt)]
    for i, j in path:
        matches_per_gt[j].append(i)

    for j in range(F_gt):
        idxs = matches_per_gt[j]
        if len(idxs) == 0:
            aligned[j] = pred_future[min(j, pred_future.shape[0] - 1)]
        else:
            aligned[j] = pred_future[idxs].mean(axis=0)

    return aligned.astype(pred_future.dtype, copy=False), path


def procrustes_align_sequence(pred_seq, gt_seq):
    """
    pred_seq, gt_seq: (T, J, 3)
    Returns pred aligned frame-by-frame to gt.
    """
    out = np.empty_like(pred_seq, dtype=np.float64)
    for t in range(pred_seq.shape[0]):
        out[t] = compute_similarity_transform(pred_seq[t], gt_seq[t])
    return out.astype(pred_seq.dtype, copy=False)


def prepare_sequence_for_plot(
    pred_seq,
    gt_seq,
    condition_len,
    use_dtw=False,
    use_pa=False,
    pa_only_future=False,
):
    """
    pred_seq, gt_seq: (T, J, 3)

    Returns:
        pred_out: aligned prediction for visualization
        dtw_path_pairs: DTW path or None
    """
    pred_seq = np.asarray(pred_seq)
    gt_seq = np.asarray(gt_seq)

    pred_out = pred_seq.copy()
    dtw_path_pairs = None

    pred_future = pred_seq[condition_len:]
    gt_future = gt_seq[condition_len:]

    if use_dtw:
        dtw_cost = dtw_cost_matrix(pred_future, gt_future)
        pred_future_aligned, dtw_path_pairs = align_pred_to_gt_timeline_with_dtw(
            pred_future, gt_future
        )
    else:
        pred_future_aligned = pred_future
        dtw_cost = None

    if use_pa and pa_only_future:
        pred_future_aligned = procrustes_align_sequence(pred_future_aligned, gt_future)

    pred_out[condition_len:] = pred_future_aligned

    if use_pa and not pa_only_future:
        pred_out = procrustes_align_sequence(pred_out, gt_seq)

    return pred_out, dtw_path_pairs, dtw_cost


# --------------------------------------------------
# GIF
# --------------------------------------------------
def save_phase2_gif(video, gt_seq, pred_seq, condition_len, horizons=None, sample_idx=0, gif_path="phase2.gif", fps=3, point_size=18, line_width=2, elev=-80, azim=-90, title_pred="Prediction", meta_item=None):
    """
    video:       (B,T,3,H,W)
    gt_seq:      (B,T,J,3)
    pred_seq:    (B,T,J,3)
    condition_len: int
    horizons: optional list/array of future horizons, e.g. [1,5,10,20,25]
    """
    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)

    video = _as_numpy(video)
    gt_seq = _as_numpy(gt_seq)
    pred_seq = _as_numpy(pred_seq)

    vid = video[sample_idx]
    gt = gt_seq[sample_idx]
    pr = pred_seq[sample_idx]

    T = vid.shape[0]

    if condition_len < 0 or condition_len > T:
        raise ValueError(f"condition_len={condition_len} is invalid for sequence length T={T}")

    frames_rgb = [to_uint8_rgb(vid[t]) for t in range(T)]
    xlim, ylim, zlim = compute_axis_limits(gt, pr)

    horizon_set = set()
    if horizons is not None:
        horizon_set = {condition_len + int(h) - 1 for h in np.asarray(horizons).tolist()}

    gif_frames = []

    action_text = ""
    if meta_item is not None:
        subj = meta_item.get("subject", "?")
        act = meta_item.get("action", "?")
        cam = meta_item.get("cam", "?")
        action_text = f"S{subj} | {act} | {cam}"

    for t in range(T):
        in_conditioning = t < condition_len
        is_boundary = (t == condition_len - 1)

        fig = plt.figure(figsize=(12, 5))
        # fig.patch.set_facecolor("#FAFAF9")
        ax_vid = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        # ax_3d.set_facecolor("#FAFAF9")

        # left: video
        ax_vid.imshow(frames_rgb[t])
        ax_vid.set_xticks([])
        ax_vid.set_yticks([])

        if in_conditioning:
            phase_text = "Conditioning / Phase 1 estimate"
            border_color = "limegreen"
        else:
            phase_text = "Prediction"
            border_color = "crimson"

        ax_vid.set_title(f"Frame {t} | {phase_text}", fontsize=11)
        add_border(ax_vid, border_color, lw=5)

        # right: 3D overlay
        ax_3d.set_title("GT vs prediction", fontsize=11)
        ax_3d.set_xlim(*xlim)
        ax_3d.set_ylim(*ylim)
        ax_3d.set_zlim(*zlim)
        ax_3d.view_init(elev=elev, azim=azim)

        # GT
        ax_3d.scatter(
            gt[t, :, 0], gt[t, :, 1], gt[t, :, 2],
            s=point_size, c="steelblue", label="GT"
        )
        for a, b in H36M_EDGES:
            ax_3d.plot(
                [gt[t, a, 0], gt[t, b, 0]],
                [gt[t, a, 1], gt[t, b, 1]],
                [gt[t, a, 2], gt[t, b, 2]],
                linewidth=line_width, c="steelblue"
            )

        # Prediction
        pred_color = "orange" if in_conditioning else "crimson"
        ax_3d.scatter(
            pr[t, :, 0], pr[t, :, 1], pr[t, :, 2],
            s=point_size, c=pred_color, label=title_pred
        )
        for a, b in H36M_EDGES:
            ax_3d.plot(
                [pr[t, a, 0], pr[t, b, 0]],
                [pr[t, a, 1], pr[t, b, 1]],
                [pr[t, a, 2], pr[t, b, 2]],
                linewidth=line_width, c=pred_color
            )

        if in_conditioning:
            status_text = "conditioning frame"
            status_color = "limegreen"
        else:
            status_text = "future predicted frame"
            status_color = "crimson"

        ax_3d.text2D(
            0.03, 0.97,
            status_text,
            transform=ax_3d.transAxes,
            color=status_color,
            fontsize=10,
            fontweight="bold",
            va="top"
        )

        if is_boundary:
            ax_3d.text2D(
                0.03, 0.90,
                "last frame before AR rollout",
                transform=ax_3d.transAxes,
                color="limegreen",
                fontsize=9,
                fontweight="bold",
                va="top"
            )

        if t in horizon_set:
            future_h = t - condition_len + 1
            ax_3d.text2D(
                0.03, 0.83,
                f"horizon = +{future_h}",
                transform=ax_3d.transAxes,
                color="black",
                fontsize=9,
                fontweight="bold",
                va="top"
            )

        ax_3d.legend(loc="upper right")

        fig.suptitle(
            f"{action_text}    |    condition_len={condition_len}",
            fontsize=10,
            fontweight="bold"
        )
        plt.tight_layout()

        gif_frames.append(fig_to_rgb_array(fig))
        plt.close(fig)

    imageio.mimsave(gif_path, gif_frames, fps=fps, loop=0)
    print(f"[OK] Saved GIF to: {gif_path}")


def main():
    root_dir = "../"
    npz_path = "debug_prediction_phase2.npz"
    raw_gif_path = "p2_ex6_raw.gif"
    aligned_gif_path = "results/p2_ex3.gif"

    sample_idx = 0
    fps = 4

    use_dtw_vis = True
    use_pa_vis = True
    pa_only_future = False

    save_raw_gif = True
    save_aligned_gif = True

    data = np.load(npz_path, allow_pickle=True)

    print("Loaded:", npz_path)
    print("Keys:", data.files)

    gt_seq = data["gt_seq"]
    condition_len = int(np.asarray(data["condition_len"]).item())
    horizons = data["horizons"] if "horizons" in data.files else None

    boxes = data["meta_box"] if "meta_box" in data.files else None
    subjects = data["meta_subject"] if "meta_subject" in data.files else None
    actions = data["meta_action"] if "meta_action" in data.files else None
    cams = data["meta_cam"] if "meta_cam" in data.files else None
    starts = data["meta_start"] if "meta_start" in data.files else None
    ends = data["meta_end"] if "meta_end" in data.files else None
    # for i, action in enumerate(actions):
    #     # print(i, action, cams[i])
    #     if action == "WakingDog_0" and cams[i] == "cam_1":
    #         print(f"Found sample index {i} for action 'Waiting_1' and cam 2")
    # return


    metas = []
    B = len(subjects)

    for i in range(B):
        item = {
            "box": boxes[i] if boxes is not None else None,
            "subject": int(subjects[i]) if subjects is not None else None,
            "action": str(actions[i]) if actions is not None else None,
            "cam": str(cams[i]) if cams is not None else None,
            "start": int(starts[i]) if starts is not None else None,
            "end": int(ends[i]) if ends is not None else None,
        }
        metas.append(item)

    pred_seq_phi = data["pred_seq_phi"] if "pred_seq_phi" in data.files else None
    pred_seq_constant = data["pred_seq_constant"] if "pred_seq_constant" in data.files else None

    print("gt_seq shape:", getattr(gt_seq, "shape", None))
    print("condition_len:", condition_len)
    print("horizons:", horizons)
    if pred_seq_phi is not None:
        print("pred_seq_phi shape:", pred_seq_phi.shape)
    if pred_seq_constant is not None:
        print("pred_seq_constant shape:", pred_seq_constant.shape)

    video, kept_indices, not_found = load_videos_from_meta(
        metas,
        root_dir=root_dir,
        out_size=224
    )

    gt_seq = gt_seq[kept_indices]
    metas_kept = [metas[i] for i in kept_indices]

    if pred_seq_phi is not None:
        pred_seq_phi = pred_seq_phi[kept_indices]

    if pred_seq_constant is not None:
        pred_seq_constant = pred_seq_constant[kept_indices]

    print("video shape:", video.shape)
    print("filtered gt_seq shape:", gt_seq.shape)
    print("not_found:", not_found)

    if sample_idx >= len(metas_kept):
        raise IndexError(f"sample_idx={sample_idx} is out of range for {len(metas_kept)} kept samples")

    if pred_seq_phi is not None:
        # print metrics for selected sample before/after alignment
        pred_raw_sample = pred_seq_phi[sample_idx]
        gt_sample = gt_seq[sample_idx]

        raw_mpjpe = mpjpe_per_frame_mm(pred_raw_sample, gt_sample)
        raw_pa = pa_mpjpe_per_frame_mm(pred_raw_sample, gt_sample)

        print("\n===== Selected sample metrics (RAW) =====")
        print("Sample:", sample_idx)
        print("Mean MPJPE (mm):", float(raw_mpjpe.mean()))
        print("Mean PA-MPJPE (mm):", float(raw_pa.mean()))

        pred_aligned_sample, dtw_pairs, cost_matrix = prepare_sequence_for_plot(
            pred_seq=pred_raw_sample,
            gt_seq=gt_sample,
            condition_len=condition_len,
            use_dtw=use_dtw_vis,
            use_pa=use_pa_vis,
            pa_only_future=pa_only_future,
        )
        if cost_matrix is not None and dtw_pairs is not None:
            path_arr = np.asarray(dtw_pairs, dtype=np.int64)  # (N,2) -> (i_pred, j_gt)

            plt.figure(figsize=(6, 5))
            plt.imshow(cost_matrix, origin="lower", aspect="auto")
            plt.plot(
                path_arr[:, 1],  # eixo X = GT (j)
                path_arr[:, 0],  # eixo Y = Pred (i)
                color="red",
                linewidth=1.5,
                label="DTW path"
            )

            plt.colorbar(label="Mean joint distance")
            plt.xlabel("GT future frame (j)")
            plt.ylabel("Pred future frame (i)")
            plt.title(f"DTW cost matrix + path | sample {sample_idx}")

            plt.legend()
            plt.tight_layout()

            plt.savefig(
                f"dtw_cost_path_sample{sample_idx}.png",
                dpi=200,
                bbox_inches="tight"
            )
            plt.close()

            print(f"[OK] Saved DTW cost + path PNG")
        aligned_mpjpe = mpjpe_per_frame_mm(pred_aligned_sample, gt_sample)
        aligned_pa = pa_mpjpe_per_frame_mm(pred_aligned_sample, gt_sample)

        print("\n===== Selected sample metrics (VIS-ALIGNED) =====")
        print("use_dtw_vis:", use_dtw_vis)
        print("use_pa_vis:", use_pa_vis)
        print("pa_only_future:", pa_only_future)
        print("Mean MPJPE (mm):", float(aligned_mpjpe.mean()))
        print("Mean PA-MPJPE (mm):", float(aligned_pa.mean()))
        if dtw_pairs is not None:
            print("DTW path length:", len(dtw_pairs))

        if save_raw_gif:
            save_phase2_gif(
                video=video,
                gt_seq=gt_seq,
                pred_seq=pred_seq_phi,
                condition_len=condition_len,
                horizons=horizons,
                sample_idx=sample_idx,
                gif_path=raw_gif_path,
                fps=fps,
                title_pred="Pred (raw)",
                meta_item=metas_kept[sample_idx],
            )

        if save_aligned_gif:
            pred_vis = pred_seq_phi.copy()
            pred_vis[sample_idx] = pred_aligned_sample

            title_parts = ["Pred"]
            if use_dtw_vis:
                title_parts.append("DTW")
            if use_pa_vis:
                title_parts.append("PA")
            title_pred = " (".join([title_parts[0], "+".join(title_parts[1:])]) + ")" if len(title_parts) > 1 else "Pred"

            save_phase2_gif(
                video=video,
                gt_seq=gt_seq,
                pred_seq=pred_vis,
                condition_len=condition_len,
                horizons=horizons,
                sample_idx=sample_idx,
                gif_path=aligned_gif_path,
                fps=fps,
                title_pred=title_pred,
                meta_item=metas_kept[sample_idx],
            )

    # constant baseline if needed
    # if pred_seq_constant is not None:
    #     save_phase2_gif(
    #         video=video,
    #         gt_seq=gt_seq,
    #         pred_seq=pred_seq_constant,
    #         condition_len=condition_len,
    #         horizons=horizons,
    #         sample_idx=sample_idx,
    #         gif_path="phase2_constant.gif",
    #         fps=fps,
    #         title_pred="Pred (constant)",
    #         meta_item=metas_kept[sample_idx],
    #     )


if __name__ == "__main__":
    main()