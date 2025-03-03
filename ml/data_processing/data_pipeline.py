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
        
    def load_and_save_dataset(self, image_dir_path: str, tensor_path: str) -> TensorDataset:
        """
        Loads the dataset from the specified file path and saves it, returns TensorDataset.
        image_dir_path (str): path to the folder containing the images
        tensor_path (str): path to save the TensorDataset
        """
        tensors = []
        labels = []

        for label_folder, label_value in zip(["positive", "negative"], [1, 0]):
            output_dir = os.path.join(image_dir_path, label_folder)
            print(f"Processing folder: {output_dir}")
            
            if not os.path.exists(output_dir):
                print(f"Folder not found: {output_dir}. Skipping...")
                continue
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
        if tensor_path:
            torch.save(dataset, tensor_path)
            print(f"Dataset saved at {tensor_path}")

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
        image_array = self.image_processor.process_single_image_for_inference()

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


    def create_dataloaders(self,
                           batch_size,
                           dataset_path,
                           spectro_dir_path = None,
                           upsample = True,
                           aug_spectro_dir_path = None,
                           aug_dataset_path = None) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Splits the dataset into training and test sets.

        Usage:
        
        The first time you train on the dataset, this function will process the images into tensors
        and save them to the location `dataset_path` specifically for the tensor dataset.
        In subsequent runs, this function will detect that the augmented tensor data has already been created
        and will skip image processing.

        The first time you are training on augmented data, provide the path to the augmented spectrograms
        and the path you would like to save the augmented tensor data to.
        This function will convert the spectrograms into tensors and save them to the indicated location.
        In subsequent runs, this function will detect that the augmented tensor data has already been created
        and will skip image processing.

        Args:
            batch_size (int): The batch size for the DataLoader.
            spectro_dir_path (str | None): Path to the spectrogram directory.
            dataset_path (str | None): Path to the TensorDataset file (created by this function)
            upsample (bool): Whether to upsample the positive class.
            aug_spectro_dir_path (str | None): Path to the augmented spectrogram directory.
            aug_dataset_path (str | None): Path to the augmented tensor dataset file (created by this function).

        Returns:
            tuple: (train_df, test_df) - The training and testing DataFrames.
        """
        if not spectro_dir_path:
            spectro_dir_path = self.image_processor.output_dir

        if os.path.exists(dataset_path):  # Tensor dataset created already
            print(f"Loading dataset from {dataset_path}")
            dataset = torch.load(dataset_path, weights_only=False)
        else:  # Folder is empty or does not exist
            print(f"Processing and loading dataset from {spectro_dir_path}")
            dataset = self.load_and_save_dataset(image_dir_path=spectro_dir_path,
                                                 tensor_path=dataset_path)

        if aug_dataset_path is not None:
            if os.path.exists(aug_dataset_path):  # Augmented tensor dataset created already
                print(f"Loading augmented dataset from {aug_dataset_path}")
                aug_dataset = torch.load(aug_dataset_path, weights_only=False)
            elif aug_spectro_dir_path:
                print(f"Adding augmented spectrogram data to training dataset from {aug_spectro_dir_path}")
                aug_dataset = self.load_and_save_dataset(image_dir_path=aug_spectro_dir_path,
                                                         tensor_path=aug_dataset_path)
            else:
                print("No augmented spectrogram data found. Proceeding without augmented data...")
                aug_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
        else:
            print("No augmented dataset path provided. Proceeding without augmented data...")
            aug_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))

        # Calculate sizes
        test_size = round(self.test_size * len(dataset))
        val_size = round(self.val_size * len(dataset))
        train_size = round(len(dataset) - test_size - val_size)  # Remaining for training

        # Perform split
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Augmented dataset size: {len(aug_dataset)}")
        # Combine train and augmented datasets
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_dataset])
        print(f"Train + augmented dataset size: {len(train_dataset)}")

        # Upsample positive class
        if upsample:
            print("Upsampling data")
            labels = [label.item() for _, label in train_dataset]
            train_counts = {0: 0, 1: 0}
            for label in labels:
                train_counts[label] = train_counts[label] + 1
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

        # Reduce memory footprint (only matters on JupyterNotebook)
        dataset, train_dataset, val_dataset, test_dataset = None, None, None, None

        print("Done.")

        return train_loader, val_loader, test_loader
