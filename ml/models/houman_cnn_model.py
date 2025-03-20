import torch
import torch.nn as nn
import torchvision.models as models


class CNNModel(nn.Module):
    """A convolutional neural network model based on EfficientNet for spectrogram processing."""

    def __init__(self, dropout: float = 0.0):
        """Initializes the CNNModel using EfficientNet-B0 with an optional dropout layer.

        Args:
            dropout (float): Dropout probability before the final classification layer.
        """
        super(CNNModel, self).__init__()

        # Load EfficientNet-B0 with pre-trained weights
        self.efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Get the number of features from the last layer of EfficientNet
        num_features = self.efficientnet.classifier[1].in_features

        # Replace the classifier with a new sequence including Dropout and FC layer
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # Dropout before classification layer
            nn.Linear(num_features, 1)  # Binary classification output
        )

        # Initialize the new FC layer weights
        nn.init.normal_(self.efficientnet.classifier[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.efficientnet.classifier[1].bias)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for EfficientNet with dropout.

        Args:
            spectrogram (torch.Tensor): Input tensor representing the spectrogram.

        Returns:
            torch.Tensor: The model's output (logit for binary classification).
        """
        return self.efficientnet(spectrogram)
