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
        audio-dir (str): Path to directory containing processed audio files
        features_filepath (str): Path to the directory where extracted features will be saved.
        extracted_features (dict): Dictionary mapping filenames to their extracted features (as numpy arrays).
        extracted_spectograms (dict): Dictionary mapping filenames to extracted spectograms (as numpy arrays).
    """

    def __init__(self, features_filepath: str, audio_dir: str):
        """Initializes the SpectrogramProcessor.

        Args:
            features_filepath (str): Path to the directory where extracted features will be saved.
            audio_dir (str): Path to the directory with processed audio files in wav
        """
        self.audio_dir = audio_dir # directory containing processed audio files
        self.features_filepath = features_filepath # path to store extracted features from spectograms
        self.extracted_spectograms = {} # dictionary to store processed spectograms
        self.extracted_features = {} # store extracted features from spectograms
        os.makedirs(self.features_filepath, exist_ok=True) # create directory in case it does not exist

    def process_all_spectrograms(self) -> None:
        """Processes all spectrograms in the given directory."""
        # Loop through all audio files
        for filename in os.listdir(self.audio_dir):
          if filename.endswith(".wav"):
            # Get path for audio file
            audio_path = os.path.join(self.audio_dir, filename)

            # Process file to generate spectogram using helper function
            spectogram = self.process_single_spectrogram(audio_path)

            # Store spectogram into dictionary
            self.extracted_spectograms[filename] = spectogram

            # # Store extracted features 
            # self.extract_features(audio_path, self.extracted_features)

    def process_single_spectrogram(self, audio_path: str) -> np.ndarray:
        """Processes a single audio file to generate its spectrogram.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: The generated spectrogram as a numpy array.
        """
        # Create spectogram using helper function
        spectogram = self.conv_to_spectrogram(audio_path)

        # Normalize generated spectogram using helper function
        spectogram_norm = self.normalize_spectrogram(spectogram)

        return spectogram_norm

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