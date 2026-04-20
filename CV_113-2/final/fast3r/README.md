# Fast3R Pipeline

This project provides a streamlined pipeline for camera pose estimation and point cloud generation using Fast3R.

## Setup Instructions

First, set up the Fast3R environment:

```bash
cd fast3r

# Create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# Install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.2 nvidia/label/cuda-12.2.0::cuda-toolkit -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install fast3r as a package
pip install -e .

cd ..
```

## Running the Pipeline

The entire pipeline can be executed with a single bash script:

```bash
bash pipeline.sh [BASE_DIR] [DATASET] [SEQUENCE]
```

Example:
```bash
bash pipeline.sh /tmp2/vincentchang chess seq-03
```

Parameters:
- `BASE_DIR`: Base directory path where datasets are located (default: /tmp2/vincentchang)
- `DATASET`: Dataset name (default: chess)
- `SEQUENCE`: Sequence number (default: seq-03)

## Pipeline Steps

The script automates the following process:

1. Creates necessary directory structure for the pipeline
2. Extracts a reference pose from the first frame (frame-000000) 
3. Runs Fast3R to predict camera poses for all frames
4. Aligns the predicted poses with the ground truth
5. Cleans existing pose files in the dataset directory
6. Copies aligned poses back to the dataset
7. Generates a PLY point cloud from the sequence with aligned poses

## Output

The final point cloud will be available at `./ply_output/[DATASET]-[SEQUENCE].ply`