#!/bin/bash
#SBATCH --partition=A40short    # GPU partition
#SBATCH --time=8:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=2                # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/train%j.out
#SBATCH --error=logs/train%j.err

source ~/.bashrc
conda activate h36m

which python
python -V

nvidia-smi -l 3600 &

python -u src/train_phase1.py \
  --root /home/s26ldeso/Human3.6M_resnet_data \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 8 \
  --lr 1e-4 \
  --outdir runs2d/phase1/