"""Joseph's CNN Model Module.

This module provides the `CNNModel` class, which is a convolutional neural network 
model built on top of a ResNet architecture for processing spectrogram data.

Dependencies:
    - PyTorch (torch)
    - torchvision.models for ResNet architecture.

TODO:
    - Integrate the forward pass logic for processing spectrogram data.
    - Add support for loading custom ResNet configurations.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNModel(nn.Module):
    """A convolutional neural network model based on ResNet for spectrogram processing.

    Attributes:
        input_folder (str): Path to the input folder containing spectrograms.
        output_folder (str): Path to the folder where model outputs will be saved.
        resnet (torchvision.models.resnet.ResNet): The ResNet backbone used for feature extraction.
    """

    def __init__(self):
        """Initializes the CNNModel.
        """
        super(CNNModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        # Remove the last FC layer and replace it with a binary classifier
        num_features = self.resnet.fc.in_features  # Get input size of original FC layer
        self.resnet.fc = nn.Linear(num_features, 1)  # Output a single logit

        # Initialize weights and biases for the new FC layer
        nn.init.normal_(self.resnet.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.resnet.fc.bias)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the model.

        Args:
            spectrogram (torch.Tensor): Input tensor representing the spectrogram.

        Returns:
            torch.Tensor: The model's output after processing the spectrogram.
        """
        return self.resnet(spectrogram)
