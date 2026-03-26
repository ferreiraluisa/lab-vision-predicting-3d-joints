# Lab Vision: Predicting 3D Human Joints from Video

This repository contains an adapted implementation of the paper **[Predicting 3D Human Dynamics from Video](https://arxiv.org/pdf/1908.04781)** developed for the course *MA-INF 2307: Lab Vision* at the University of Bonn during my master's studies.

![Example](results/p2_ex1.gif)

The implementation trains an autoregressive network on the Human3.6M dataset to predict 3D human joints from video sequences. In this work, the original PHD framework is adapted to predict 3D joint coordinates instead of SMPL body parameters.

## Getting Started

The original paper uses a pretrained ResNet from the repository [End-to-end Recovery of Human Shape and Pose](https://github.com/akanazawa/hmr). The pretrained model and weights are provided in **TensorFlow**, so a separate
environment is required for the feature preprocessing step.

You can create the TensorFlow environment with:

```bash
conda env create -f tf.yaml
```

Then download the pretrained weights, as described in the original repository:

```bash
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz
tar -xf models.tar.gz
```
This environment is used to extract features using the pretrained HMR ResNet.

The rest of the pipeline is implemented in PyTorch, as required by the lab
course. You can install the PyTorch environment with:

```bash
conda env create -f h36m.yaml
```
This environment is used for training and evaluating the model.

## Pre-processing Data

You can obtain the **Human3.6M dataset** from the [official website](http://vision.imar.ro/human3.6m/description.php). To organize the dataset, I used this [pre-existing script](https://github.com/akanazawa/human_dynamics/blob/master/src/datasets/h36/read_human36m.py#L449) from the original PHD implementation, this script helps separate the 2D keypoints, 3D joint positions, and camera parameters for each of the four cameras available in Human3.6M. You can run using run_read_h36.sh file.

After organizing the data, I extract image features using ResNet-50 with the pretrained HMR weights. The features are extracted **offline** and saved to disk. This makes training much faster because the model receives precomputed features instead of raw video frames. This approach is also convenient because:

- The feature extractor (ResNet) is frozen during both training phases.
- The original HMR model is implemented in **TensorFlow**, while the rest of this project is implemented in **PyTorch**.

The preprocessing pipeline is implemented in:

```
tf_feature_extraction.sh
```

You can run the preprocessing step with:

```bash
sbatch tf_feature_extraction.sh
```

## Training

### Phase 1

In the first training phase, the autoregressive module is frozen.  We train the temporal encoder `f_movie` together with the 3D regressor `f_3d` so that the model learns to estimate 3D human joints from the extracted features. You can run the first training phase with:

```bash
sbatch train1.sh
```



### Phase 2

In the second training phase, the 3D regressor `f_3d` is frozen. We then train the temporal encoder `f_movie` together with the autoregressive module `f_ar` to predict future poses.

Training uses a **curriculum learning strategy**. At the beginning, the model is fed with ground-truth poses (teacher forcing) as input for the next prediction step. As training progresses, the model gradually starts using its own predictions as input, increasing the number of consecutive autoregressive steps. You can run the second training phase with:
```bash
sbatch train2.sh
```

## Evaluation
### Phase 1
For phase, the model is evaluated using **MPJPE (Mean Per Joint Position Error)**, following the protocol used in the original paper. You can run the evaluation with:
```bash
sbatch eval_phase1.sh
```

The evaluation was performed on the test set using Subject 9. The results obtained are:

| Method | MPJPE (mm) |
|------|------|
| PHD (paper) | 83.7 |
| Mine | **79.9** |

Our implementation achieves a slightly lower MPJPE compared to the value reported in the original paper for this setup.

### Phase 2
For phase 2, the model is evaluated using **Reconstruction Error**, i.e., MPJPE after Procrustes alignment (PA-MPJPE), following the protocol used in the original paper. You can run the evaluation with:
```bash
sbatch eval_phase2.sh
```

The evaluation was performed on the test set using Subject 9.  The Constant Baseline corresponds to simply repeating the last observed frame for all future time steps.

The results obtained are in reconstruction error(mm):

| Method | 1 | 5 | 10 | 20 | 25 |
|------|------|------|------|------|------|
| PHD (paper) | **57.7** | **59.5** | 61.1 | 61.9 | 65.3 |
| Mine |  59.0 | 59.6 | **60.3** | **61.3** |  **63.0** |
| Constant Baseline | 95.2 | 95.9 | 96.7 | 99.1 | 103.2 |

## Visualization

We provide visualization scripts (`src/visualize_phase_1.py`, `src/visualize_phase_2.py`, and `src/visu_dtw.py`) used to generate the qualitative results presented in the Lab Vision presentation (24.03.2026).  

These tools enable inspection of predicted 3D human motion, as well as comparisons with ground truth sequences and DTW-aligned results, following the evaluation protocol described in the original [PHD](https://arxiv.org/pdf/1908.04781) work.  

Sample visualizations are available in the `results/` directory.

## Author

| [Luísa Ferreira](https://ferreiraluisa.github.io/luisaferreira/) |
| :------------------------------------------------------------: | 
| Msc. Student¹                                                 | 
| <s26ldeso@uni-bonn.de>                                       

¹University of Bonn \
Bonn, North Rhine-Westphalia, Germany

## Acknowledgements

Thanks to **Fatemeh Jabbari** for advising me throughout the semester :)


---

### Enjoy it! :smiley:
