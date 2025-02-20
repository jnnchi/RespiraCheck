"""
Spectrogram Processing Module.

This module provides the `SpectrogramProcessor` class for converting audio files 
into spectrograms, normalizing them, and extracting features.

Dependencies:
    - librosa
    - numpy
    - ffmpeg
    - matplotlib.pyplot

TODO:
    - Implement all methods to handle creation of spectrograms.
"""

import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

class SpectrogramProcessor:
    """Processes and extracts features from audio spectrograms.

    This class provides methods for converting audio files to spectrograms, 
    normalizing spectrograms, and extracting features for further analysis.

    Attributes:
        audio_folder (str): Path to directory containing processed audio files
        features_filepath (str): Path to the directory where extracted features will be saved.
        extracted_features (dict): Dictionary mapping filenames to their extracted features (as numpy arrays).
        extracted_spectrograms (dict): Dictionary mapping filenames to their extracted spectrograms (as numpy arrays).
    """

    def __init__(self, audio_folder="ml/data/cough_data/processed_audio", spectrograms_folder="ml/data/cough_data/spectrograms"):
        """Initializes the SpectrogramProcessor.

        Args:
            audio_folder (str): Path to the directory with processed audio files in wav
            features_filepath (str): Path to the directory where extracted features will be saved.
        """
        self.audio_folder = audio_folder # directory containing processed audio files
        self.spectrograms_folder = spectrograms_folder 

    def process_all_spectrograms(self) -> None:
        """Processes all spectrograms in the given directory and saves them as images.

        Note: This assumes that all processed audio files are in WAV format
                and saved in one folder(directory) whose path is in self.audio_folder.
        """
        for label in ["positive", "negative"]: 
            audio_dir = os.path.join(self.audio_folder, label)  # Full path to the labeled folder
            spectrogram_dir = os.path.join(self.spectrograms_folder, label)  # Path to save spectrogram

            # Make spectrogram folder if it doesn't exist
            os.makedirs(spectrogram_dir, exist_ok=True)

            for filename in os.listdir(audio_dir):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(audio_dir, filename)

                    # Process file to generate spectrogram
                    spectrogram = self.process_single_spectrogram(audio_path)

                    # Save spectrogram image to the spectrogram folder
                    spectrogram_path = os.path.join(spectrogram_dir, filename.replace(".wav", ".png"))
                    self.save_spectrogram_image(spectrogram, spectrogram_path)

    def save_spectrogram_image(self, spectrogram: np.ndarray, save_path: str) -> None:
        """Saves a spectrogram array as an image.

        Args:
            spectrogram (np.ndarray): The spectrogram array.
            save_path (str): Path where the spectrogram image should be saved.
        """

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')
        plt.colorbar(label='Amplitude (dB)')
        plt.tight_layout()
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()


    def process_single_spectrogram(self, audio_path: str) -> np.ndarray:
        """Processes a single audio file to generate its spectrogram.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: The generated spectrogram as a numpy array.
        """
        # Create spectrogram using helper function
        spectrogram = self.conv_to_spectrogram(audio_path)

        # Normalize generated spectrogram using helper function
        spectrogram_norm = self.normalize_spectrogram(spectrogram)

        return spectrogram_norm

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

    def apply_stft(self, audio_path: str) -> np.ndarray:
      """Applies STFT Filter with a Hanning Window to a Audio File

      Args:
        audio_path (str): Path to the audio file

      Returns:
        np.ndarray: Features exxtracted by STFT on Audio   
      """
      # Reference Code: https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056
      # Load the audio file with original sample rate
      y, sr = librosa.load(audio_path, sr=None)

      # Compute STFT (the values below are default except hop length)
      n_fft = 2048  # FFT window size
      hop_length = 512  # Hop length for STFT
      audio_stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann"))

      # Converting the amplitude to decibels
      log_spectro = librosa.amplitude_to_db(audio_stft)  

      return log_spectro
    
    def plot_spectrogram(self, filename, spectrogram) -> None:
        """Plots the spectrogram using Matplotlib

        Args:
            filename: File name of the audio
            spectrogram: Spectrogram data extracted from the audio
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')
        plt.colorbar(label='Amplitude (dB)')
        plt.title(f"Spectrogram: {filename}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()