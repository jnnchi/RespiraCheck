"""
cnn_model.py: Defines a Convolutional Neural Network for mel-spectrogram classification.

This module includes the model architecture and any necessary helper functions for 
defining and modifying the CNN.

Classes:
    CNNModel: A PyTorch implementation of a basic CNN for image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from data_loader import train_dataloader

# Using a 70% training, 15% validation, and 15% testing data split

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        # NOTE: shape of each image is: (335, 840, 4) -> this informs our parameters for the layers

        # Convolutional layers
        # This one should get textures from the image
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1) # 4 input channels, 32 output chans
        # This one gets more complex patterns from the feature maps made by conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        # Flattens output of conv layers into 1d vectors
        self.fc1 = nn.Linear(64 * 83 * 210, 128) 
        # Output layer
        self.fc2 = nn.Linear(128, num_classes) # 128 is # of input features, which is # of outputs from fc1; numclasses = # of classifications

    def forward(self, x):
        # Apply layers and ReLU activation function
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        # Flatten the tensor into 2D tensor with shape [batch_size, num_features]
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model Initialization
def initialize_and_train(train_dataloader, num_classes, epochs=10):
    # Splitting the dataset
    dataset_size = len(train_dataloader.dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataloader.dataset, [train_size, val_size, test_size]
    )

    # Create new data loaders for the splits
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model, Loss, Optimizer
    model = CNNModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs)

    # Save the model
    torch.save(model.state_dict(), 'cnn_model.pth')

    return model, test_dataloader

# Training Pipeline
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, features, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.argmax(dim=1))  # One-hot to class index
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_dataloader)}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, features, labels in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels.argmax(dim=1))
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()
        print(f"Validation Loss: {val_loss / len(val_dataloader)}, Accuracy: {100 * correct / total:.2f}%")


# Evaluate the Model
def evaluate_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, features, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    num_classes = 10
    model, test_dataloader = initialize_and_train(train_dataloader, num_classes=num_classes, epochs=10)

    # Evaluate the model on the test set
    evaluate_model(model, test_dataloader)