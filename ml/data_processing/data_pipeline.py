"""Dataset Processing Module.

This module provides the `DataPipeline` class for handling dataset operations, 
including loading, processing, and splitting datasets for training and inference.

Dependencies:
    - pandas
    - sklearn

TODO: - Implement actual dataset processing logic.
      - Include error handling for file operations.
"""
from .audio_processor import AudioProcessor
from .image_processor import ImageProcessor

import pandas as pd
import os
from PIL import Image

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision import transforms

class DataPipeline:
    """Processes datasets, including loading, splitting, and preparing for inference.

    This class provides methods for loading datasets, processing them for training,
    and preparing single instances for inference.

    Attributes:
        data_path (str): Path to the dataset file.
        test_size (float): Proportion of the dataset to include in the test split.
        audio_processor: AudioProcessor instance for handling audio processing.
        spectrogram_processor: SpectrogramProcessor instance for handling spectrogram processing.
        metadata_df (pd.DataFrame): DataFrame containing metadata for the dataset.
        metadata_path (str): Path to the metadata file.
    """

    def __init__(self, test_size: float, val_size: float, audio_processor: AudioProcessor,  
                image_processor: ImageProcessor):
        """Initializes the DatasetProcessor.

        Args: 
            data_path (str): Path to the dataset file.
            test_size (float): Proportion of the dataset to include in the test split.
            audio_processor (AudioProcessor): Instance for handling audio processing.
            image_processor (ImageProcessor): Instance for handling spectrogram processing.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.audio_processor = audio_processor
        self.image_processor = image_processor

    def process_all(self) -> None:
        """Processes the entire dataset for training or analysis. 
        Creates folders of labeled audio and spectrograms
        """
        self.audio_processor.process_all_audio()
        self.image_processor.process_all_images()
        
    def load_dataset(self) -> TensorDataset:
        """Loads the dataset from the specified file path into a DataFrame."""
        tensors = []
        labels = []  

        for label_folder, label_value in zip(["positive", "negative"], [1, 0]): 
            output_dir = os.path.join(self.image_processor.output_folder, label_folder)

            for image_name in os.listdir(output_dir):
                image_path = os.path.join(output_dir, image_name)
                image_tensor = self.image_to_tensor(image_path)
                
                tensors.append(image_tensor)
                labels.append(label_value)

        # Tensor of all features (N x D) - N is number of samples (377), D is feature dimension (3,224,224)
        X = torch.stack(tensors)  
        # Tensor of all labels (N x 1) - 377x1
        y = torch.tensor(labels, dtype=torch.long) 

        return TensorDataset(X, y)


    def image_to_tensor(self, image_path: str) -> torch.Tensor:
        """Converts a spectrogram image to a PyTorch tensor.

        Args:
            image_path (str): Path to the spectrogram image file.

        Returns:
            torch.Tensor: The PyTorch tensor representation of the image.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to ResNet18 input size
            transforms.ToTensor(),  # Convert image to tensor
        ])

        image = Image.open(image_path).convert("RGB") # Convert from RGBA to RGB
        tensor_image = transform(image)

        return tensor_image  # shape will be 3, 224, 224

    def process_single_for_inference(self, instance) -> torch.Tensor:
        """Processes a single instance for inference.

        Args:
            instance: A single data instance to be processed.
        """
        pass


    def create_dataloaders(self, batch_size) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Splits the dataset into training and test sets.

        Returns:
            tuple: (train_df, test_df) - The training and testing DataFrames.
        """
        dataset = self.load_dataset()

        # Calculate sizes
        test_size = round(self.test_size * len(dataset))
        val_size = round(self.val_size * len(dataset))
        train_size = round(len(dataset) - test_size - val_size)  # Remaining for training

        # Perform split
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
