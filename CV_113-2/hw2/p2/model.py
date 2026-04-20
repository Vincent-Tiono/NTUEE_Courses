# ============================================================================
# File: model.py
# Date: 2025-03-11
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Add parent directory to sys.path to import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p2.config import batch_size

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        # First convolutional block with residual connection
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.downsample1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional block with residual connection
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )
        self.downsample2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional block with residual connection
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        self.downsample3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # First block with residual connection
        identity = self.downsample1(x)
        out = self.conv_block_1(x)
        out = out + identity  # Use + instead of += to avoid in-place operations
        out = self.relu1(out)
        out = self.pool1(out)
        
        # Second block with residual connection
        identity = self.downsample2(out)
        out = self.conv_block_2(out)
        out = out + identity  # Use + instead of += to avoid in-place operations
        out = self.relu2(out)
        out = self.pool2(out)
        
        # Third block with residual connection
        identity = self.downsample3(out)
        out = self.conv_block_3(out)
        out = out + identity  # Use + instead of += to avoid in-place operations
        out = self.relu3(out)
        out = self.pool3(out)
        
        # Classifier
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        # Use pretrained weights for better feature extraction
        self.resnet = models.resnet18(weights="DEFAULT")  # Python3.8 w/ torch 2.2.1
        # self.resnet = models.resnet18(pretrained=True)  # Python3.6 w/ torch 1.10.1
        
        # Modify the first conv layer for CIFAR-10's smaller image size
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        


    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    # Print ResNet18 Architecture and parameter count
    # resnet = ResNet18()
    # print("\nResNet18 Architecture:")
    # print(resnet)
    
    mynet = MyNet()
    print("\nMyNet Architecture:")
    print(mynet)
    
    # Calculate and print total parameters
    total_params = sum(p.numel() for p in mynet.parameters())
    trainable_params = sum(p.numel() for p in mynet.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
