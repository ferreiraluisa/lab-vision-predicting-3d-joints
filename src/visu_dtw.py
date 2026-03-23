import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

PRED_COLOR = "tab:orange"
GT_COLOR = "deepskyblue"


def mpjpe_frame(a, b):
    return np.linalg.norm(a - b, axis=-1).mean()


def dtw_cost_matrix(pred, gt):
    t_pred, t_gt = pred.shape[0], gt.shape[0]
    cost = np.zeros((t_pred, t_gt), dtype=np.float64)
    for i in range(t_pred):
        for j in range(t_gt):
            cost[i, j] = mpjpe_frame(pred[i], gt[j])
    return cost


def dtw_path(cost):
    t_pred, t_gt = cost.shape
    dp = np.full((t_pred + 1, t_gt + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, t_pred + 1):
        for j in range(1, t_gt + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    i, j = t_pred, t_gt
    path = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
            prevs = [
                (dp[i - 1, j], i - 1, j),
                (dp[i, j - 1], i, j - 1),
                (dp[i - 1, j - 1], i - 1, j - 1),
            ]
            _, i, j = min(prevs, key=lambda x: x[0])
        elif i > 0:
            path.append((i - 1, 0))
            i -= 1
        else:
            path.append((0, j - 1))
            j -= 1

    path.reverse()
    return path


def plot_cost_matrix(cost, path, save_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor("#FAFAF9")
    ax.set_facecolor("#FAFAF9")

    im = ax.imshow(
        cost.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    px = [i for i, j in path]
    py = [j for i, j in path]
    ax.plot(px, py, linewidth=2.5, color="red")

    ax.set_title("DTW Cost Matrix (future frames)", fontsize=18, pad=14)
    ax.set_xlabel("Pred", fontsize=14)
    ax.set_ylabel("GT", fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MPJPE", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved cost matrix to {save_path}")
    plt.close(fig)


def set_equal_3d_axes(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_skeleton(ax, joints, color, label=None):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=18, label=label)
    for a, b in H36M_EDGES:
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color=color,
            linewidth=2,
        )


def setup_axis(ax, all_pts):
    ax.cla()
    set_equal_3d_axes(ax, all_pts)
    ax.set_facecolor("#FAFAF9")
    ax.view_init(elev=-80, azim=-90)


def animate(pred, gt, dtw_pairs, save=None, fps=3):
    n = len(dtw_pairs)
    all_pts = np.concatenate([pred.reshape(-1, 3), gt.reshape(-1, 3)], axis=0)

    fig = plt.figure(figsize=(10.5, 5.2))
    fig.patch.set_facecolor("#FAFAF9")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    def update(t):
        i_dtw, j = dtw_pairs[t]
        i_raw = min(j, pred.shape[0] - 1)

        setup_axis(ax1, all_pts)
        draw_skeleton(ax1, pred[i_raw], PRED_COLOR, "Pred")
        draw_skeleton(ax1, gt[j], GT_COLOR, "GT")
        ax1.set_title(f"No DTW\npred={i_raw} gt={j}", fontsize=13)

        setup_axis(ax2, all_pts)
        draw_skeleton(ax2, pred[i_dtw], PRED_COLOR, "Pred")
        draw_skeleton(ax2, gt[j], GT_COLOR, "GT")
        ax2.set_title(f"With DTW\npred={i_dtw} gt={j}", fontsize=13)

        handles = [
            plt.Line2D([0], [0], color=PRED_COLOR, lw=2, label="Pred"),
            plt.Line2D([0], [0], color=GT_COLOR, lw=2, label="GT"),
        ]
        ax1.legend(handles=handles, loc="upper right", fontsize=10)
        ax2.legend(handles=handles, loc="upper right", fontsize=10)

    anim = FuncAnimation(fig, update, frames=n, interval=1000 // fps, repeat=True)

    if save:
        if save.endswith(".gif"):
            anim.save(save, writer=PillowWriter(fps=fps))
        else:
            anim.save(save, fps=fps)
        print(f"Saved animation to {save}")

    plt.show()
    plt.close(fig)


def select_sequences(data, sample_idx, future_len):
    if "pred_seq_phi" in data and "gt_seq" in data:
        pred = data["pred_seq_phi"][sample_idx][:future_len]
        gt = data["gt_seq"][sample_idx][:future_len]
        return pred, gt

    if "pred_seq" in data and "gt_seq" in data:
        pred_seq = data["pred_seq"][sample_idx]
        gt_seq = data["gt_seq"][sample_idx]

        if "condition_len" not in data:
            raise KeyError("condition_len not found in npz, cannot slice future from pred_seq/gt_seq")

        condition_len = int(np.array(data["condition_len"]).item())
        pred = pred_seq[condition_len:condition_len + future_len]
        gt = gt_seq[condition_len:condition_len + future_len]
        return pred, gt

    raise KeyError("Could not find pred_future/gt_future or pred_seq/gt_seq in npz")


def build_gt_index_to_pred_index(path, t_gt):
    best = {}
    for i_pred, j_gt in path:
        c = abs(i_pred - j_gt)
        if j_gt not in best or c < best[j_gt][0]:
            best[j_gt] = (c, i_pred)

    pairs = []
    for j in range(t_gt):
        if j in best:
            pairs.append((best[j][1], j))
        else:
            pairs.append((min(j, t_gt - 1), j))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="debug_prediction_phase2.npz")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--future_len", type=int, default=25)
    parser.add_argument("--save_anim", default="")
    parser.add_argument("--save_cost", default="dtw_cost_future.png")
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    pred, gt = select_sequences(data, args.sample_idx, args.future_len)

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[-1] != 3 or gt.shape[-1] != 3:
        raise ValueError(f"Expected (T, J, 3), got pred={pred.shape}, gt={gt.shape}")

    t = min(len(pred), len(gt))
    pred = pred[:t]
    gt = gt[:t]

    print("Using ONLY future frames")
    print("Pred shape:", pred.shape)
    print("GT shape:", gt.shape)

    cost = dtw_cost_matrix(pred, gt)
    path = dtw_path(cost)
    dtw_pairs = build_gt_index_to_pred_index(path, t)

    plot_cost_matrix(cost, path, args.save_cost)

    raw_pairs = [(j, j) for j in range(t)]

    no_dtw_err = np.mean([mpjpe_frame(pred[i], gt[j]) for i, j in raw_pairs])
    dtw_err = np.mean([mpjpe_frame(pred[i], gt[j]) for i, j in dtw_pairs])

    print(f"Mean error (no DTW): {no_dtw_err:.4f}")
    print(f"Mean error (DTW): {dtw_err:.4f}")

    animate(pred, gt, dtw_pairs, save=args.save_anim, fps=args.fps)


if __name__ == "__main__":
    main()