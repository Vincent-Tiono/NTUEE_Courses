# Set dataset path
# DATASET=<your_7SCENES_dataset_path>
DATASET="../../7SCENES"

# Run dense dataset
python calculate.py --dataset $DATASET
python seq2ply.py --dataset $DATASET

# Run sparse dataset
python calculate_random_sample.py --dataset $DATASET
python seq2ply_sparse.py --dataset $DATASET