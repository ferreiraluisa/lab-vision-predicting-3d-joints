#!/bin/bash
#SBATCH --partition=A40devel     # GPU partition
#SBATCH --time=1:00:00           # max runtime (HH:MM:SS)
#SBATCH --gpus=1                 # number of GPUs
#SBATCH --ntasks=1               # number of CPU tasks
#SBATCH --output=logs/run_read_h36m_%j.out
#SBATCH --error=logs/run_read_h36m_%j.err


source ~/.bashrc
conda activate h36m

which python
python -V

python /home/s26ldeso/implementation-phd-lab-vision/src/datasets/read_human_36m.py \
    --source_dir /home/s26ldeso/Human3.6 \
    --out_dir /home/s26ldeso/Human3.6M_preprocessed 
