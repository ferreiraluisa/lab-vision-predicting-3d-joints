import argparse
import os
from pathlib import Path
import time
import numpy as np

"""
This script loads the .npz clips extracted by extract_clips.py, runs the video frames through a TensorFlow ResNet-50 (with the HMR pre-trained weights).

This was the way I found to make the training faster, instead of feeding the video frames directly into the model, I feed the features, since in both training phases the feature extractor(ResNet) is frozen and doesn't need to be backpropagated through. 

Output .npz files will contain:
    - feats: The ResNet-50 features for each frame (shape: [T, 2048])
    - joints3d: The 3D joint positions (shape: [T, num_joints, 3])
    - joints2d: The 2D joint positions (shape: [T, num_joints, 2])
    - K: The camera intrinsics (shape: [3, 3] or [T, 3, 3])
    - box: The bounding box (if available, shape: [T, 4] or [4])
    - metadata: subject, action, cam, start frame, end frame

Coded by Luisa Ferreira, 2026 with assistance of ChatGPT 5.2 (OpenAI).
"""


def to_tf_resnetv2_input(video01_tchw: np.ndarray) -> np.ndarray:
    """
    video01_tchw: (T,3,224,224) float in [0,1]
    returns: (T,224,224,3) float32 in [-1,1] NHWC
    """
    x = np.clip(video01_tchw, 0.0, 1.0)
    x = x * 2.0 - 1.0
    x = np.transpose(x, (0, 2, 3, 1))  # T,H,W,3
    return x.astype(np.float32)


def build_tf_graph(height=224, width=224, weight_decay=0.0):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    slim = tf.contrib.slim
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2

    x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, height, width, 3], name="input_nhwc")

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        net, _ = resnet_v2.resnet_v2_50(
            x_ph,
            num_classes=None,
            is_training=False,
            reuse=False,
            scope="resnet_v2_50",
        )
        feat_op = tf.squeeze(net, axis=[1, 2], name="feat_2048")  # (N,2048)

    resnet_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v2_50")
    saver = tf.compat.v1.train.Saver(var_list=resnet_vars)
    return x_ph, feat_op, saver, resnet_vars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="root of exported .npz clips")
    ap.add_argument("--out", required=True, help="output root for .npz with feats or .npy")
    ap.add_argument("--ckpt", required=True, help="TF1 checkpoint prefix (no .index/.data)")
    ap.add_argument("--tf-batch", type=int, default=256)
    ap.add_argument("--subject", nargs="+", default=None, help="optional list of subjects to process (e.g. S1 S5)")
    args = ap.parse_args()

    in_root = Path(args.inp)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    x_ph, feat_op, saver, resnet_vars = build_tf_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, args.ckpt)
    print(f"✓ Restored resnet_v2_50 vars: {len(resnet_vars)} from {args.ckpt}")
    print("Starting feature extraction...")
    if args.subject is not None:
        subject_root = in_root / f"S{args.subject[0]}"
        if not subject_root.exists():
            raise ValueError(f"Subject folder not found: {subject_root}")
        files = sorted(subject_root.rglob("*.npz"))
    else:
        files = sorted(in_root.rglob("*.npz"))
    print(f"Found {len(files)} clips")

    t0 = time.time()
    for idx, fpath in enumerate(files, 1):

        # save next to mirrored structure
        rel = fpath.relative_to(in_root)
        out_path = (out_root / rel).with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"[{idx}/{len(files)}] Skipping existing {out_path}")
            continue

        d = np.load(fpath, allow_pickle=True)

        if "video01" in d:
            video01 = d["video01"].astype(np.float32)
        elif "video_u8" in d:
            video01 = d["video_u8"].astype(np.float32) / 255.0
        else:
            raise KeyError(f"{fpath} does not have any video.")

        frames = to_tf_resnetv2_input(video01)

        feats = sess.run(feat_op, feed_dict={x_ph: frames})

        np.savez(
            out_path,
            feats=feats.astype(np.float32),
            joints3d=d["joints3d"],
            joints2d=d["joints2d"],
            K=d["K"],
            box=d["box"] if "box" in d else None,
            subject=d["subject"],
            action=d["action"],
            cam=d["cam"],
            start=d["start"],
            end=d["end"],
        )

        if idx % 200 == 0:
            dt = time.time() - t0
            print(f"[{idx}/{len(files)}] {idx/dt:.2f} clips/s")


    sess.close()
    dt = time.time() - t0
    print(f"Done: {len(files)} clips in {dt:.1f}s ({len(files)/dt:.2f} clips/s)")
    print(f"Saved to {out_root}")


if __name__ == "__main__":
    main()