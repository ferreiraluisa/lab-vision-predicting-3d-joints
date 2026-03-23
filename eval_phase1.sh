#!/bin/bash
#SBATCH --partition=A40devel    # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/results.out
#SBATCH --error=logs/results.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 1800 &

python -u src/eval_phase1.py \
  --features_root /home/s26ldeso/h36m_feats_from_hmr_resnet \
  --preprocessed_root /home/s26ldeso/Human3.6M_preprocessed \
  --model_path /home/s26ldeso/implementation-phd-lab-vision/runs/phase1_bone/best.pt \
  --out outputs/result_hmr.npz