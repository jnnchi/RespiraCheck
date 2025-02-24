"""CNN Model Module.

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

    def __init__(self, dropout: float = 0.0):
        """Initializes the CNNModel.
        """
        super(CNNModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.resnet.fc.in_features  # Get input size of original FC layer

        # Remove the last FC layer and replace it with a binary classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),  # Apply dropout before the final layer
            nn.Linear(num_features, 1)  # Binary classification output
        )

        # Freeze all the pre-trained layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        ### Selectively unfreeze some layers
        # for m in self.resnet.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.requires_grad = True  # Ensure BatchNorm is updating

        # for name, param in self.resnet.layer4[1].named_parameters():
        #     if '2' in name:  # Unfreeze conv2.weight, bn2.weight, bn2.bias
        #         param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Initialize weights and biases for the new FC layer
        self.resnet.fc[1].weight.data.normal_(mean=0.0, std=0.01)
        self.resnet.fc[1].bias.data.zero_()

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the model.

        Args:
            spectrogram (torch.Tensor): Input tensor representing the spectrogram.

        Returns:
            torch.Tensor: The model's output after processing the spectrogram.
        """
        return self.resnet(spectrogram)

    def check_frozen_layers(self):
        """Prints the frozen status of each layer in the model.
        requires_grad = True indicates that the layer is trainable.
        """
        for name, param in self.resnet.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
