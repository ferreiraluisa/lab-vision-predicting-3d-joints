import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
"""
Attempted implementation of adapted PHD model for 3D joint prediction(in PyTorch).

Implemented by Luisa Ferreira (2025)
"""


# ============================================================
# RESIDUAL BLOCK
# ============================================================
# from the original paper, we have that both the causal temporal encoder fmovie and the autoregressive predictor fAR have the same architecture, consisting of 3 residual blocks. Each block consists of GroupNorm, ReLU, 1D Convolution, GroupNorm, ReLU, 1D Convolution, and each 1D Convolution uses a kernel size of 3 and a filter size of 2048.

# implementing causal convolution because nn.Conv1d does not support causality natively, output sees both past and future frames.
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,):
        super().__init__()
        self.left_pad = (kernel_size - 1) # to ensure causality, depend only on previous frames
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0
        )

    def forward(self, x):
        # x: (B, C, T)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0), mode="replicate") # pad only on the left, replicating first values
        return self.conv(x)

class ResidualBlock(nn.Module):
    # group norm + relu + causal conv1d + group norm + relu + causal conv1d + skip connection
    def __init__(self, hidden=512, kernel=3, dropout=0.2):
        super().__init__()
        self.gn1 = nn.GroupNorm(32 if hidden >= 32 else 1, hidden)
        self.gn2 = nn.GroupNorm(32 if hidden >= 32 else 1, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = CausalConv1d(hidden, hidden, kernel_size=kernel)
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=kernel)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # (B,H,T)
        y = self.gn1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.drop(y)
        y = self.gn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x +  y



# ============================================================
# CAUSAL TEMPORAL ENCODER(fmovie) and AUTOREGRESSIVE PREDICTOR(fAR)
# ============================================================
# causal temporal encoder => encode 3D Human Dynamics, generating latent movie strips that captures temporal context of 3D motion until time t.
# autoregressive predictor => predict the next latent movie strip given the previous ones.
# from paper: "Predicting 3D Human Dynamics from Video":
# 3 residual blocks, each with kernel size 3, resulting in a receptive field of 13 frames.
class CausalTemporalNet(nn.Module):
    def __init__(self, dim=2048, hidden=512, num_blocks=3, kernel=3, dropout=0.2):
        super().__init__()
        self.pre_ln = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden=hidden, kernel=kernel, dropout=dropout) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden, dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  

    def forward(self, feats): 
        x = self.in_proj(self.pre_ln(feats))     # (B,T,H)
        x = x.permute(0, 2, 1)       # (B,H,T)
        for b in self.blocks:
            x = b(x)
        x = x.permute(0, 2, 1)       # (B,T,H)
        delta = self.out_proj(x)     # (B,T,dim)
        return feats + delta
    

# ============================================================
# 3D REGRESSOR (adapted from HMR's)
# ============================================================
# from the original HMR paper, the iterative 3D regressor consists of two fully-connected layers with 1024 neurons each with a dropout layer in between, followed by a final layer of 85D neurons.
#
# I've adapted it to output 17 joints * 3D = 51D. (instead of SMPL parameters Θ = {θ,β,R,t,s})
class JointRegressor(nn.Module): 
    def __init__(self, latent_dim=2048, joints_num=17, hidden=1024, dropout=0.2):
        super().__init__()
        self.joints_num = joints_num
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, joints_num * 3)

    def forward(self, phi):
        B, T, D = phi.shape
        x = torch.relu(self.fc1(phi))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        y = self.out(x)
        return y.view(B, T, self.joints_num, 3)


# ============================================================
# PHD MODEL
# ============================================================
# overall architecture(inference pipeline ):
# first, a ResNet-50 backbone is used to extract image features from each frame in the input video.
# second, these features are passed through the causal temporal encoder fmovie to obtain latent movie strips.
# third, the autoregressive predictor fAR predicts the next latent movie strip based on previous ones
# finally, the joint regressor f3D takes both the latent movie strips and the predicted ones to output the 3D joint positions.
class PHDFor3DJoints(nn.Module):
    def __init__(self, latent_dim=2048, joints_num=17, dropout=0.0):
        super().__init__()

        # ----------------------------------------------------
        # PER-FRAME FEATURE EXTRACTOR
        # ----------------------------------------------------
        # pretrained ResNet-50 + average pooling of last layer as 2048D feature extractor
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # commented out bc we don't want to extract features here, only in preprocess_resnet_features.py

        # if freeze_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False

        self.latent_dim = latent_dim

        self.f_movie = CausalTemporalNet(latent_dim, dropout=dropout)
        self.f_AR = CausalTemporalNet(latent_dim, dropout=dropout)
        self.f_3D = JointRegressor(latent_dim, joints_num, dropout=dropout)

    # @torch.no_grad() # resnet must not be trained
    # def extract_features(self, video):
    #     # video: (B, T, C, H, W) batch_size, frames, channels, height, width
    #     B, T, C, H, W = video.shape
    #     x = video.view(B * T, C, H, W)
    #     feats = self.backbone(x)          
    #     feats = feats.flatten(1)         
    #     return feats.view(B, T, -1)

    # def forward(self, video, predict_future=False):
    def forward(self, feats, predict_future=False):
        # feats = self.extract_features(video) # preprocessed features input in preprocess_resnet_features.py
        # feats: (B, T, 2048)
        phi = self.f_movie(feats) # temporal causal encoder f_movie
        joints_phi = self.f_3D(phi) # 3D regressor f_3D 
        if not predict_future:
            return phi, None, joints_phi, None

        ar_out = self.f_AR(phi) # autoregressive predictor f_AR
        phi_hat = torch.zeros_like(ar_out)
        phi_hat[:, 1:, :] = ar_out[:, :-1, :]

        joints_hat = self.f_3D(phi_hat)

        # phi : teacher movie strips (B,T,D) batch_size, time, latent_dim
        # phi_hat : predicted movie strips (B,T,D) batch_size, time, latent_dim
        # joints_phi : joints from phi
        # joints_hat : joints from phi_hat (optional)
        return phi, phi_hat, joints_phi, joints_hat	
