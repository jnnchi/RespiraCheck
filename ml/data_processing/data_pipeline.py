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
import json
import shutil

import torch
from torch.utils.data import DataLoader, TensorDataset

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
        """Processes the entire dataset for training or analysis."""
        
        pass


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

    def sort_wav_files(self, source_folder, destination_folder):
        """
        Reads .wav files and their corresponding .json files from the source folder.
        Copies them into a 'positive' or 'negative' subfolder inside the destination folder
        based on the 'status' field in the JSON file. The copied JSON file contains only the status field.

        Args:
            source_folder (str): Path to the folder containing .wav and .json files.
            destination_folder (str): Path to the output directory where sorted files will be stored.
        """
        positive_folder = os.path.join(destination_folder, "positive")
        negative_folder = os.path.join(destination_folder, "negative")
        os.makedirs(positive_folder, exist_ok=True)
        os.makedirs(negative_folder, exist_ok=True)
        
        for file in os.listdir(source_folder):
            if file.endswith(".wav"):
                wav_path = os.path.join(source_folder, file)
                json_path = os.path.join(source_folder, file.replace(".wav", ".json"))
                
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        try:
                            data = json.load(f)
                            status = data.get("status", "").lower()
                            
                            if status == "covid-19":
                                target_folder = positive_folder
                            elif status == "healthy":
                                target_folder = negative_folder
                            else:
                                continue
                            
                            shutil.copy(wav_path, target_folder)
                            
                        except json.JSONDecodeError:
                            print(f"Error reading JSON file: {json_path}")
                            continue