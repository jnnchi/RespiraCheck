"""
cnn_model.py: Defines a Convolutional Neural Network for mel-spectrogram classification.

This module includes the model architecture and necessary helper functions for 
training, validating, and testing a CNN on audio spectrograms.

Classes:
    CNNModel: A PyTorch implementation of a basic CNN for image classification.
Functions:
    initialize_and_train: Initializes the model and orchestrates the training process.
    train_model: Handles the training and validation phases for the model.
    evaluate_model: Evaluates the model on a given test dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from data_loader import train_dataloader

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network for classifying mel-spectrogram images.

    Attributes:
        conv1: A convolutional layer for low-level feature extraction.
        conv2: A convolutional layer for extracting higher-level features.
        fc1: A fully connected layer to process flattened feature maps.
        fc2: The output layer producing class scores.
    """

    def __init__(self, num_classes):
        """
        Initializes the CNNModel.

        Args:
            num_classes: An integer representing the number of output classes.
        """
        super(CNNModel, self).__init__()

        # NOTE: shape of each image is: (335, 840, 4) -> this informs our parameters for the layers

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # 4 input channels, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 83 * 210, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Maps features to diagnoses

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x: A PyTorch tensor representing a batch of input images.

        Returns:
            A PyTorch tensor containing class scores for each image in the batch.
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def initialize_and_train(train_dataloader, num_classes, epochs=10):
    """
    Splits the dataset, initializes the CNN model, and trains it.

    Args:
        train_dataloader: A DataLoader object containing the dataset.
        num_classes: An integer representing the number of output classes.
        epochs: An integer representing the number of training epochs (default is 10).

    Returns:
        A tuple containing the trained model and the test dataloader.
    """
    # Splitting the dataset
    dataset_size = len(train_dataloader.dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
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


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10):
    """
    Handles the training and validation of the CNN model.

    Args:
        model: The CNNModel instance to be trained.
        train_dataloader: A DataLoader object for training data.
        val_dataloader: A DataLoader object for validation data.
        criterion: The loss function (e.g., CrossEntropyLoss).
        optimizer: The optimizer for updating model weights.
        epochs: An integer representing the number of training epochs (default is 10).

    Returns:
        No return value. Prints training and validation losses and accuracy during training.
    """
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


def evaluate_model(model, test_dataloader):
    """
    Evaluates the trained CNN model on a test dataset.

    Args:
        model: The trained CNNModel instance to be evaluated.
        test_dataloader: A DataLoader object for the test dataset.

    Returns:
        No return value. Prints the accuracy of the model on the test dataset.
    """
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
