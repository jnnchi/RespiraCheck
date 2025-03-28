"""
Spectrogram Processing Module.

This module provides the `SpectrogramProcessor` class for converting audio files
into spectrograms, normalizing them, and extracting features.
"""

import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from .image_processor import ImageProcessor
from pydub import AudioSegment
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


class SpectrogramProcessor(ImageProcessor):
    """Processes and extracts features from audio spectrograms.

    This class provides methods for converting audio files to spectrograms,
    normalizing spectrograms, and extracting features for further analysis.

    Attributes:
        audio_folder (str): Path to directory containing processed audio files
        features_filepath (str): Path to the directory where extracted features will be saved.
        extracted_features (dict): Dictionary mapping filenames to their extracted features (as numpy arrays).
        extracted_spectrograms (dict): Dictionary mapping filenames to their extracted spectrograms (as numpy arrays).
    """

    def __init__(
        self,
        stft=False,
        audio_folder="ml/data/cough_data/processed_audio",
        output_folder="ml/data/cough_data/spectrograms",
    ):
        """Initializes the SpectrogramProcessor.

        Args:
            audio_folder (str): Path to the directory with processed audio files in wav
            features_filepath (str): Path to the directory where extracted features will be saved.
        """
        super().__init__(audio_folder, output_folder)
        self.stft = stft

    def process_all_images(self) -> None:
        """Processes all spectrograms in the given directory and saves them as images.

        Note: This assumes that all processed audio files are in WAV format
            and saved in one folder (directory) whose path is in self.audio_folder.
        """
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as p:
            for label in ["positive", "negative"]:
                audio_dir = os.path.join(
                    self.audio_folder, label
                )  # Full path to the labeled folder
                spectrogram_dir = os.path.join(
                    self.output_folder, label
                )  # Path to save spectrogram
                os.makedirs(spectrogram_dir, exist_ok=True)
                for filename in os.listdir(audio_dir):
                    if filename.endswith(".wav"):
                        audio_path = os.path.join(audio_dir, filename)
                        spectrogram_path = os.path.join(
                            spectrogram_dir, filename.replace(".wav", ".png")
                        )
                        p.submit(
                            self.process_and_save_spectrogram,
                            audio_path,
                            spectrogram_path,
                        )

    def process_and_save_spectrogram(
        self, audio_path: str, spectrogram_path: str
    ) -> None:
        """Processes spectrogram for use in multiprocessing"""
        spectrogram = self.process_single_spectrogram(audio_path)
        self.save_spectrogram_image(spectrogram, spectrogram_path)
        print(f"Processed and saved spectrogram: {spectrogram_path}")

    def save_spectrogram_image(self, spectrogram: np.ndarray, save_path: str) -> None:
        """Saves a spectrogram array as an image.

        Args:
            spectrogram (np.ndarray): The spectrogram array.
            save_path (str): Path where the spectrogram image should be saved.
        """

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="inferno")
        plt.axis("off")

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def process_single_spectrogram(self, audio_path: str) -> np.ndarray:
        """Processes a single audio file to generate its spectrogram.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: The generated spectrogram as a numpy array.
        """

        y, sr = librosa.load(audio_path, sr=None)

        # Create spectrogram using helper function
        if self.stft:
            spectrogram = self.apply_stft(y)
        else:
            spectrogram = self.conv_to_spectrogram(y, sr)

        # Normalize generated spectrogram using helper function
        spectrogram_norm = self.normalize_spectrogram(spectrogram)

        return spectrogram_norm

    def process_single_image_for_inference(self, audio: AudioSegment) -> np.ndarray:
        """Converts an AudioSegment object to a mel spectrogram.
        NOTE SHOULD MAKE THE REST OF THE CODE WORK ON AUDIOSEGMENT INSTEAD OF FILEPATH
        THIS IS TEMPORARY FOR TESTING STAGES

        Args:
            audio (AudioSegment): The input audio segment.

        Returns:
            np.ndarray: Log-scaled mel spectrogram.
        """
        # Convert AudioSegment to raw samples (NumPy array)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize to range [-1, 1] (librosa expects this format)
        samples /= np.max(np.abs(samples))

        # Get sample rate
        sr = audio.frame_rate  # librosa needs sample rate information

        if self.stft:
            # Compute STFT (the values below are default except hop length)
            n_fft = 2048  # FFT window size
            hop_length = 512  # Hop length for STFT
            audio_stft = np.abs(
                librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, window="hann")
            )

            # Converting the amplitude to decibels
            spectrogram_db = librosa.amplitude_to_db(audio_stft)
        else:
            # Convert to log-scaled mel spectrogram
            spectrogram = librosa.feature.melspectrogram(
                y=samples, sr=sr, n_mels=128, fmax=8000
            )

            # Convert to decibels
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        return spectrogram_db

    def conv_to_spectrogram(
        self, audio_clip: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Converts an audio file to its spectrogram representation.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: The spectrogram of the audio file.
        """

        # Convert to log scaled mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=audio_clip, sr=sample_rate, n_mels=128, fmax=8000
        )

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
        # spectrogram_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # CURRENTLY SKIPPING THIS FUNCTION TO TEST USING TRANSFORMS WITHIN TENSOR TO
        # ADHERE TO IMAGE NET TRAINING SPECS

        return spectrogram

    def apply_stft(self, audio_clip: np.ndarray) -> np.ndarray:
        """Applies STFT Filter with a Hanning Window to a Audio File

        Args:
          audio_path (str): Path to the audio file

        Returns:
          np.ndarray: Features extracted by STFT on Audio
        """
        # Reference Code: https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056
        # Load the audio file with original sample rate

        # Compute STFT (the values below are default except hop length)
        n_fft = 2048  # FFT window size
        hop_length = 512  # Hop length for STFT
        audio_stft = np.abs(
            librosa.stft(audio_clip, n_fft=n_fft, hop_length=hop_length, window="hann")
        )

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
        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="inferno")
        plt.colorbar(label="Amplitude (dB)")
        plt.title(f"Spectrogram: {filename}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
