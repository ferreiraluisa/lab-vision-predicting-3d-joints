# config.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# params for dataset and training
H36M_ROOT = "/home/s26ldeso/Human3.6M_preprocessed"

# data params
H36M_ROOT = "/home/s26ldeso/Human3.6M_preprocessed_resnet_features"  # Path to Human3.6M dataset
FRAME_SKIP = 2           # Frame subsampling rate during video loading
SEQ_LEN = 40               # total frames per sequence
INPUT_LEN = 15             # Frames used to warm up the history
PRED_LEN = 25              # Frames to predict (Paper predicts 25)

# model params
LATENT_DIM = 2048          # Dimension of "Movie Strip" (same as ResNet feature)
JOINTS_NUM = 17            # Standard Human3.6M joints (or 32 depending on your processing)

# training params
BATCH_SIZE = 8             # Batch size for single GPU (adjust based on your GPU memory)
LR = 1e-4
EPOCHS = 50
CURRICULUM_STEPS = 25      # Slowly increase autoregressive steps from 1 to 25