#!/bin/bash
#SBATCH --partition=A40devel    # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/xlips.out
#SBATCH --error=logs/xlips.err

source ~/.bashrc
conda activate h36m
which python
python -V

python src/datasets/extract_clips.py \
  --root /home/s26ldeso/Human3.6M_preprocessed \
  --out /home/s26ldeso/tmp_clips_npz_ \
  --seq-len 40 --frame-skip 2 --stride 5 \
  --batch-size 16 --num-workers 8 \
  --subjects 9

conda deactivate
conda activate tf
which python
python -V

python src/datasets/tf_resnet_features.py \
  --in /home/s26ldeso/tmp_clips_npz_ \
  --out /home/s26ldeso/h36m_feats_from_hmr_resnet \
  --ckpt models/model.ckpt-667589 \
  --tf-batch 256 \
  --subject 9
