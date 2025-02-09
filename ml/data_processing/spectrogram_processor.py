"""
Spectrogram Processing Module.

This module provides the `SpectrogramProcessor` class for converting audio files 
into spectrograms, normalizing them, and extracting features.

Dependencies:
    - librosa
    - numpy

TODO:
    - Implement all methods to handle creation of spectrograms.
"""

import numpy as np
import librosa
import os

class SpectrogramProcessor:
    """Processes and extracts features from audio spectrograms.

    This class provides methods for converting audio files to spectrograms, 
    normalizing spectrograms, and extracting features for further analysis.

    Attributes:
        features_filepath (str): Path to the directory where extracted features will be saved.
        extracted_features (dict): Dictionary mapping filenames to their extracted features (as numpy arrays).
    """

    def __init__(self, features_filepath: str):
        """Initializes the SpectrogramProcessor.

        Args:
            features_filepath (str): Path to the directory where extracted features will be saved.
        """
        self.features_filepath = features_filepath
        self.extracted_features = {}

    def process_all_spectrograms(self) -> None:
        """Processes all spectrograms in the given directory."""
        pass

    def process_single_spectrogram(self, audio_path: str) -> np.ndarray:
        """Processes a single audio file to generate its spectrogram.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: The generated spectrogram as a numpy array.
        """
        pass

    def conv_to_spectrogram(self, audio_path: str) -> np.ndarray:
        """Converts an audio file to its spectrogram representation.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: The spectrogram of the audio file.
        """
        # Load the audio file with original sample rate 
        # (assumes sample rate already standardized by AudioProcessor)
        y, sr = librosa.load(audio_path, sr=None)

        # Convert to log scaled mel spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        # Convert to decibels
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        return spectrogram_db

    def normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range using Min-Max Scaling

        Args:
            spectrogram (np.ndarray): The spectrogram to normalize.

        Returns:
            np.ndarray: The normalized spectrogram.
        """
        spectrogram_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        return spectrogram_norm

    def extract_features(self, audio_path: str, extracted_features: dict) -> None:
        """Extracts features from a given audio file's spectrogram.

        Args:
            audio_path (str): Path to the audio file.
            extracted_features (dict): A dictionary mapping filename to features that were extracted.
        """
        pass