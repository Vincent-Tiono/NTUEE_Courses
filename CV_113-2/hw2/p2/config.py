# ============================================================================
# File: config.py
# Date: 2025-03-11
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'improved_mynet'  # name of experiment

# Model Options
model_type = 'mynet'  # 'mynet' or 'resnet18'

# Learning Options
epochs = 60                 # train more epochs
batch_size = 64             # larger batch size
use_adam = True             # use Adam optimizer
lr = 3e-4                   # better learning rate for Adam
milestones = [25, 40, 50]   # adjusted milestones