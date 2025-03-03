"""Dataset Processing Module.

This module provides the `DataPipeline` class for handling dataset operations, 
including loading, processing, and splitting datasets for training and inference.

"""
from .audio_processor import AudioProcessor
from .image_processor import ImageProcessor

import os
from PIL import Image

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from pydub import AudioSegment
import io


class DataPipeline:
    """Processes datasets, including loading, splitting, and preparing for inference.

    This class provides methods for loading datasets, processing them for training,
    and preparing single instances for inference.

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include for validation.
        audio_processor: AudioProcessor instance for handling audio processing.
        image_processor: ImageProcessor instance for handling spectrogram or extracted features processing.
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
        
    def load_and_save_dataset(self, dataset_path: str) -> TensorDataset:
        """
        Loads the dataset from the specified file path and saves it, returns TensorDataset.
        dataset_path (str): path to save the TensorDataset
        """
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
        
        dataset = TensorDataset(X, y)
        if dataset_path:
            torch.save(dataset, dataset_path)
            print(f"Dataset saved at {dataset_path}")

        return dataset


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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as per ResNet18
        ])

        image = Image.open(image_path).convert("RGB") # Convert from RGBA to RGB
        tensor_image = transform(image)

        return tensor_image  # shape will be 3, 224, 224

    def process_single_for_inference(self, audio: AudioSegment) -> torch.Tensor:
        """Processes a single instance for inference.

        Args:
            audio: assume we receive audio in bytes
        """
        # Convert AudioSegment to WAV format
        audio = audio.set_frame_rate(48000).set_channels(1)

        audio = self.audio_processor.process_single_audio_for_inference(audio)

        # just spectrograms for now
        image_array = self.image_processor.process_single_image_for_inference(audio)

        # Convert spectrogram (NumPy array) to PIL Image (needed for torchvision transforms)
        image = Image.fromarray(image_array)

        # Same transformations used in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
        ])

        # Apply transformations
        image_tensor = transform(image)  # Shape: (1, 3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension → (1, 3, 224, 224)
        
        return image_tensor


    def create_dataloaders(self, batch_size, dataset_path = None, upsample = True) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Splits the dataset into training and test sets.

        Args:
            batch_size (int): The batch size for the DataLoader.
            dataset_path (str | None): Path to the TensorDataset file.

        Returns:
            tuple: (train_df, test_df) - The training and testing DataFrames.
        """
        if dataset_path:
            print(f"Loading dataset from {dataset_path}")
            dataset = torch.load(dataset_path, weights_only=False)
        else:
            print("Processing and loading dataset")
            dataset = self.load_and_save_dataset(dataset_path)

        # Calculate sizes
        test_size = round(self.test_size * len(dataset))
        val_size = round(self.val_size * len(dataset))
        train_size = round(len(dataset) - test_size - val_size)  # Remaining for training

        # Perform split
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Upsample positive class
        if upsample:
            print("Upsampling data")
            labels = [label.item() for _, label in train_dataset]
            train_counts = {}
            for label in labels:
                train_counts[label] = train_counts.get(label, 0) + 1
            # print(train_counts)

            weights = torch.where(torch.tensor(labels) == 0, 1 / train_counts[0], 1 / train_counts[1])
            # print(labels[:5], weights[:5])

            wr_sampler = WeightedRandomSampler(weights, int(len(train_dataset) * 1.5))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=wr_sampler)
        
        else:
            print("No upsampling")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create DataLoaders
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Count labels in train_loader
        train_counts = {}
        for _, labels in train_loader:
            for label in labels:
                train_counts[label.item()] = train_counts.get(label.item(), 0) + 1

        # print(train_counts)

        # Reduce memory footprint
        dataset, train_dataset, val_dataset, test_dataset = None, None, None, None

        return train_loader, val_loader, test_loader
