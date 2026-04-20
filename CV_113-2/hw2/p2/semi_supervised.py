import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import json

import config as cfg
from model import MyNet, ResNet18
from dataset import CIFAR10Dataset, get_dataloader

# Enable anomaly detection in development to catch issues
# torch.autograd.set_detect_anomaly(True)

class UnlabeledDataset(Dataset):
    """Dataset class for unlabeled data in semi-supervised learning"""
    def __init__(self, dataset_dir, transform=None):
        super(UnlabeledDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Get all image files from the unlabeled dataset directory
        self.image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png') or f.endswith('.jpg')]
        print(f'Number of unlabeled images: {len(self.image_files)}')

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_dir, self.image_files[index])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {'images': image, 'filename': self.image_files[index]}

def get_unlabeled_dataloader(dataset_dir, batch_size):
    """Create dataloader for unlabeled dataset"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = UnlabeledDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader

def generate_pseudo_labels(model, unlabeled_loader, threshold_k, device):
    """
    Generate pseudo-labels for unlabeled data using the trained model
    Args:
        model: trained model
        unlabeled_loader: dataloader for unlabeled dataset
        threshold_k: confidence threshold for pseudo-labeling
        device: device to run the model on
    Returns:
        pseudo_labels: dictionary containing filenames and corresponding pseudo-labels
        confidence_scores: dictionary containing filenames and corresponding confidence scores
    """
    model.eval()
    pseudo_labels = {}
    confidence_scores = {}
    
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, desc="Generating pseudo-labels"):
            images = batch['images'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            
            # Get predicted class and confidence scores
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_classes = torch.max(probabilities, dim=1)
            
            # Store pseudo-labels and confidence scores for each image
            for i, filename in enumerate(filenames):
                pseudo_labels[filename] = predicted_classes[i].item()
                confidence_scores[filename] = confidence[i].item()
    
    return pseudo_labels, confidence_scores

def filter_pseudo_labels(pseudo_labels, confidence_scores, threshold_k):
    """
    Filter pseudo-labels based on confidence threshold
    Args:
        pseudo_labels: dictionary containing filenames and corresponding pseudo-labels
        confidence_scores: dictionary containing filenames and corresponding confidence scores
        threshold_k: confidence threshold for pseudo-labeling
    Returns:
        filtered_pseudo_labels: dictionary containing filenames and corresponding pseudo-labels
                               for images with confidence score > threshold_k
    """
    filtered_pseudo_labels = {}
    
    for filename, score in confidence_scores.items():
        if score > threshold_k:
            filtered_pseudo_labels[filename] = pseudo_labels[filename]
    
    print(f"Number of images with confidence > {threshold_k}: {len(filtered_pseudo_labels)}")
    print(f"Original unlabeled set size: {len(pseudo_labels)}")
    return filtered_pseudo_labels

class PseudoLabelDataset(Dataset):
    """Dataset class for pseudo-labeled data"""
    def __init__(self, dataset_dir, pseudo_labels, transform=None):
        super(PseudoLabelDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.pseudo_labels = pseudo_labels
        
        # Use only the filenames that have pseudo-labels
        self.image_files = list(pseudo_labels.keys())
        print(f'Number of pseudo-labeled images: {len(self.image_files)}')

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        filename = self.image_files[index]
        img_path = os.path.join(self.dataset_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get the pseudo-label for this image
        label = torch.tensor(self.pseudo_labels[filename], dtype=torch.long)
        
        return {'images': image, 'labels': label}

def get_pseudo_labeled_dataloader(dataset_dir, pseudo_labels, batch_size):
    """Create dataloader for pseudo-labeled dataset"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PseudoLabelDataset(dataset_dir, pseudo_labels, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def semi_supervised_training(model_path, train_dir, unlabeled_dir, val_dir, threshold_k=0.8):
    """
    Implement semi-supervised learning with pseudo-labeling
    Args:
        model_path: path to the trained model
        train_dir: directory containing training data
        unlabeled_dir: directory containing unlabeled data
        val_dir: directory containing validation data
        threshold_k: confidence threshold for pseudo-labeling
    """
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the trained model
    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training with a fresh model instead...")
    
    model.to(device)
    
    # Get unlabeled dataloader
    unlabeled_loader = get_unlabeled_dataloader(unlabeled_dir, batch_size=cfg.batch_size)
    
    # Generate pseudo-labels for unlabeled data
    pseudo_labels, confidence_scores = generate_pseudo_labels(model, unlabeled_loader, threshold_k, device)
    
    # Filter pseudo-labels based on confidence threshold
    filtered_pseudo_labels = filter_pseudo_labels(pseudo_labels, confidence_scores, threshold_k)
    
    # Create augmented training set with pseudo-labeled data
    train_loader = get_dataloader(train_dir, batch_size=cfg.batch_size, split='train')
    pseudo_labeled_loader = get_pseudo_labeled_dataloader(unlabeled_dir, filtered_pseudo_labels, batch_size=cfg.batch_size)
    val_loader = get_dataloader(val_dir, batch_size=cfg.batch_size, split='val')
    
    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr / 10)  # Lower learning rate for fine-tuning
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr / 10, momentum=0.9, weight_decay=1e-6)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    
    # Training with pseudo-labeled data
    best_val_acc = 0.0
    model_save_dir = os.path.dirname(model_path)
    # Ensure directory exists
    os.makedirs(model_save_dir, exist_ok=True)
    semi_supervised_model_path = os.path.join(model_save_dir, f'{cfg.model_type}_semi_supervised.pth')
    
    print("Starting semi-supervised training with pseudo-labeled data...")
    
    for epoch in range(20):  # Train for fewer epochs when fine-tuning
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        # Train on original labeled data
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}: Training on labeled data")):
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        # Train on pseudo-labeled data (if available)
        if len(filtered_pseudo_labels) > 0:
            for batch_idx, batch in enumerate(tqdm(pseudo_labeled_loader, desc=f"Epoch {epoch+1}: Training on pseudo-labeled data")):
                images, labels = batch['images'].to(device), batch['labels'].to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}: Validation")):
                images, labels = batch['images'].to(device), batch['labels'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate metrics
        train_acc = train_correct / total_samples if total_samples > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Calculate average loss
        num_batches = len(train_loader) + (len(pseudo_labeled_loader) if len(filtered_pseudo_labels) > 0 else 0)
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1}/{20}, Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), semi_supervised_model_path)
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        scheduler.step()
    
    print(f"Semi-supervised training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {semi_supervised_model_path}")
    
    return semi_supervised_model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semi-supervised learning with pseudo-labeling")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--unlabeled_dir", type=str, required=True, help="Directory containing unlabeled data")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory containing validation data")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for pseudo-labeling")
    
    args = parser.parse_args()
    
    semi_supervised_training(
        model_path=args.model_path,
        train_dir=args.train_dir,
        unlabeled_dir=args.unlabeled_dir,
        val_dir=args.val_dir,
        threshold_k=args.threshold
    ) 