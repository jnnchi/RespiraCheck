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
from .spectrogram_processor import SpectrogramProcessor

import pandas as pd
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader, TensorDataset
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

    def __init__(self, test_size: float, audio_processor: AudioProcessor,  
                spectrogram_processor: SpectrogramProcessor, metadata_df: pd.DataFrame, 
                metadata_path: str, input_path="data/cough_data/original_data", output_path="data/cough_data"):
        """Initializes the DatasetProcessor.

        Args: 
            data_path (str): Path to the dataset file.
            test_size (float): Proportion of the dataset to include in the test split.
            audio_processor (AudioProcessor): Instance for handling audio processing.
            spectrogram_processor (SpectrogramProcessor): Instance for handling spectrogram processing.
            metadata_df (pd.DataFrame): DataFrame containing metadata for the dataset.
            metadata_path (str): Path to the metadata file.
        """
        self.input_path = input_path
        self.audio_output_path = f"{output_path}/processed_audio"
        self.spectrogram_output_path = f"{output_path}/spectrograms"
        self.test_size = test_size
        self.audio_processor = audio_processor
        self.spectrogram_processor = spectrogram_processor
        self.metadata_df = metadata_df
        self.metadata_path = metadata_path

    def process_all(self) -> None:
        """Processes the entire dataset for training or analysis. 
        Creates folders of labeled audio and spectrograms
        """
        self.audio_processor.process_all_audio()
        self.spectrogram_processor.process_all_spectrograms()
        
    def load_dataset(self) -> TensorDataset:
        """Loads the dataset from the specified file path into a DataFrame."""
        tensors = []
        labels = []  

        for label_folder, label_value in zip(["positive", "negative"], [1, 0]): 
            spectrogram_dir = os.path.join(self.spectrograms_folder, label_folder)

            for image_name in os.listdir(spectrogram_dir):
                image_path = os.path.join(spectrogram_dir, image_name)
                spectrogram_tensor = self.spectrogram_to_tensor(image_path)
                
                tensors.append(spectrogram_tensor)
                labels.append(label_value)

        # Tensor of all features (N x D) - N is number of samples, D is feature dimension
        X = torch.stack(tensors)  
        # Tensor of all labels (N x 1)
        y = torch.tensor(labels, dtype=torch.long) 

        return TensorDataset(X, y)


    def spectrogram_to_tensor(self, image_path: str) -> torch.Tensor:
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


    def load_dataset(self) -> TensorDataset:
        """Loads the dataset from the specified file path into a DataFrame."""
        pass

    def split_dataset(self) -> tuple[DataLoader, DataLoader]:
        """Splits the dataset into training and test sets.

        Returns:
            tuple: (train_df, test_df) - The training and testing DataFrames.
        """
        pass
