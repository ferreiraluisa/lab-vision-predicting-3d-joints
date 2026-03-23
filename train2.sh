#!/bin/bash
#SBATCH --partition=A100short    # GPU partition
#SBATCH --time=8:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=4                # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/train%j.out
#SBATCH --error=logs/train%j.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 3600 &

python -u src/train_phase2.py \
  --root /home/s26ldeso/h36m_feats_from_hmr_resnet \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --phase1_ckpt runs/phase1_bone/best.pt \