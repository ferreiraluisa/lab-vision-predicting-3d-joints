#!/bin/bash
#SBATCH --partition=A40short    # GPU partition
#SBATCH --time=8:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/results.out
#SBATCH --error=logs/results.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 1800 &

python src/eval_phase2.py \
  --root /home/s26ldeso/h36m_feats_from_hmr_resnet \
  --ckpt checkpoints_phase2/best_phase2.pt \
  --subjects 9 \
  --condition_len 15 \
  --horizons 1 5 10 20 25 \
  --stride_clips 25 \
  --batch_size 32 \
  --num_workers 8 \
  --amp \
  --use_dtw \
  --csv_out eval_phase2_metrics.csv \
  --save_debug_npz debug_prediction_phase2.npz