import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

import config

def train_model():
    """
    Downloads the EuroSAT dataset, fine-tunes a pre-trained MobileNetV2 model,
    evaluates it on a validation set, saves the weights, and generates a confusion matrix.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Data Transforms
    # We resize images to the TILE_SIZE (e.g., 64x64) defined in config,
    # convert to PyTorch tensors, and normalize using ImageNet stats.
    transform = transforms.Compose([
        transforms.Resize((config.TILE_SIZE, config.TILE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Load EuroSAT Dataset
    print("Downloading/Loading EuroSAT dataset...")
    # This automatically downloads the dataset if not present.
    dataset = datasets.EuroSAT(root='./data', download=True, transform=transform)
    
    # 4. Train/Validation Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Dataset split: {train_size} training images, {val_size} validation images.")

    # 5. Initialize Pre-trained MobileNetV2
    print("Initializing MobileNetV2 with ImageNet weights...")
    # We use MobileNetV2 because it's lightweight and fast, perfect for a simple MVP
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Replace the final classifier layer to output 10 classes (for EuroSAT)
    # The original MobileNetV2 outputs 1000 classes for ImageNet.
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # 6. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # We use Adam optimizer for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Training Loop
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    best_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # Validation Phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        # We also collect all predictions and true labels for the confusion matrix
        # during the last epoch.
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                if epoch == config.NUM_EPOCHS - 1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # 8. Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            # Ensure models directory exists
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print(f"Training complete. Best model saved to {config.MODEL_SAVE_PATH}")

    # 9. Generate and Save Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.EUROSAT_CLASSES, 
                yticklabels=config.EUROSAT_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - MobileNetV2 on EuroSAT')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'.")

if __name__ == "__main__":
    train_model()
