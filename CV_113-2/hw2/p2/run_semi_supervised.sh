#!/bin/bash

# Step 1: Set paths
DATA_DIR="../hw2_data/p2_data"
UNLABELED_DIR="../hw2_data/p2_unlabeled"  # Path to unlabeled data
CHECKPOINT_DIR="./checkpoint/improved_model"

# Step 2: Make sure checkpoint directory exists
mkdir -p $CHECKPOINT_DIR

# Step 3: Update config.py to use mynet
sed -i '' 's/model_type = .*/model_type = "mynet"/' ./config.py

# Step 4: Train the improved model first
echo "=== Training the improved MyNet model ==="
python p2_train.py --dataset_dir $DATA_DIR

# Step 5: Get the path to the best model
BEST_MODEL=$(find $CHECKPOINT_DIR -name "*_model_best.pth" | sort -r | head -n 1)
echo "Best model found at: $BEST_MODEL"

# Step 6: Run semi-supervised learning with pseudo-labeling
echo "=== Running semi-supervised learning ==="
python semi_supervised.py \
    --model_path $BEST_MODEL \
    --train_dir $DATA_DIR/train \
    --unlabeled_dir $UNLABELED_DIR \
    --val_dir $DATA_DIR/val \
    --threshold 0.85

echo "Semi-supervised learning completed!"

# Step 7: Evaluate the final model
SEMI_MODEL=$(find $CHECKPOINT_DIR -name "*_semi_supervised.pth" | sort -r | head -n 1)
echo "Evaluating semi-supervised model: $SEMI_MODEL"
python p2_eval.py --model_path $SEMI_MODEL --dataset_dir $DATA_DIR/val 