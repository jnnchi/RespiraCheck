"""Dataset Processing Module.

This module provides the `DataPipeline` class for handling dataset operations,
including loading, processing, and splitting datasets for training and inference.

"""

from PIL import Image
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment
from .audio_processor import AudioProcessor
from .spectrogram_processor import SpectrogramProcessor
from .image_processor import ImageProcessor

import torch
from torch.utils.data import (
    random_split,
    DataLoader,
    WeightedRandomSampler,
)
from torchvision import transforms

class DataPipeline:
    """Processes datasets, including loading, splitting, and preparing for inference.

    This class provides methods for loading datasets, processing them for training,
    and preparing single instances for inference.

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include for validation.
    """

    def __init__(self, test_size: float, val_size: float, audio_processor: AudioProcessor, image_processor: ImageProcessor):
        """Initializes the DatasetProcessor.

        Args:
            data_path (str): Path to the dataset file.
            test_size (float): Proportion of the dataset to include in the test split.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.audio_processor = audio_processor
        self.image_processor = image_processor

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

    def create_dataloaders(self, batch_size, dataset_path = None, upsample = False) -> tuple[DataLoader, DataLoader, DataLoader]:
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
            dataset = self.load_dataset()

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

        print(train_counts)

        # Reduce memory footprint
        dataset, train_dataset, val_dataset, test_dataset = None, None, None, None

        return train_loader, val_loader, test_loader
    
    def process_single_for_inference(self, audio: AudioSegment) -> torch.Tensor:
        """Processes a single instance for inference.

        Args:
            audio: assume we receive audio in bytes
        Returns:
            torch.Tensor: The processed instance for inference.
            BytesIO: The in-memory spectrogram image (for visualization).
        """
        # Convert AudioSegment to WAV format with desired sample rate/channels
        audio = audio.set_frame_rate(
            self.audio_processor.target_sample_rate
        ).set_channels(1)
        audio = self.audio_processor.process_single_audio_for_inference(audio)
        if not audio:
            return None, None

        # Create a spectrogram from the audio
        spectrogram_array = self.image_processor.process_single_image_for_inference(
            audio
        )

        # Create and format the matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(spectrogram_array, aspect="auto", origin="lower", cmap="inferno")
        ax.axis("off")

        # Save the figure to an in-memory bytes buffer instead of a file
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Convert spectrogram array to a PIL Image (for transformations)
        image = Image.fromarray(spectrogram_array)
        image = image.convert("RGB")

        # Same transformations used in training
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize to ResNet18 input size
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize as per ResNet18
            ]
        )

        # Apply transformations
        image_tensor = transform(image)  # Expected shape: (C, H, W)
        print("Processed image into image tensor.")
        # Only add a batch dimension if necessary
        return image_tensor, buf
