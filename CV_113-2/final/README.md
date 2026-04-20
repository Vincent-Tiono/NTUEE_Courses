# 3D Reconstruction on 7SCENES

## Project Overview

This project implements 3D reconstruction techniques using the 7SCENES dataset. We propose two different methods to achieve accurate 3D scene reconstruction:

1. **Pose Estimation**: A traditional approach for dense 3D reconstruction using camera pose estimation
2. **Fast3R**: An pre-trained model for 3D reconstruction

## Dataset

You can download the 7SCENES dataset from:
[Download 7SCENES Dataset](https://drive.google.com/file/d/1r172cIGZKBc3b7_b1-cscPnVFj8bl8HF/view)

## Quick Start

### Pose Estimation Method

```bash
cd pose_estimation
conda create --name math python=3.11
conda activate math
pip install opencv-python pillow open3d

# Run reconstruction
python calculate.py --dataset <your_7SCENES_dataset_path>
python seq2ply.py --dataset <your_7SCENES_dataset_path>
```

### Fast3R Method

```bash
cd fast3r
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r
conda install pytorch torchvision torchaudio pytorch-cuda=12.2 nvidia/label/cuda-12.2.0::cuda-toolkit -c pytorch -c nvidia
pip install -r requirements.txt

# Run the full pipeline
bash pipeline.sh <base_dir> <dataset> <sequence>
```

## Detailed Documentation

For detailed instructions on how to use each method, please refer to the README.md in each method's directory:

- [Pose Estimation Documentation](./pose_estimation/README.md)
- [Fast3R Documentation](./fast3r/README.md)

## Results

The final output of both methods is a 3D point cloud (PLY file) representing the reconstructed scene.