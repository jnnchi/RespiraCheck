"""
Data Augmentation Module.

This module provides "DataAugmentProcessor" Class for creating new audio
and mel spectrogram classes using data augmentation techniques.
"""

import librosa
import numpy as np
import os
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from spectrogram_processor import SpectrogramProcessor


class DataAugmentProcessor:
    """Creates new audio and spectrogram samples through augmentation techniques

    The class provides methods for increasing/decreasing volume,
    time shifting, pitch shifing on audio samples and time/frequency masking
    for mel spectrograms. It does the necessary augmentations and returns the
    mel spectrograms.
    NOTE: It also assumes all data has been processed previously.

    Attributes:
      audio_path: Stores the path to audio file to be augmented
    """

    def __init__(self, audio_path):
        # Store path to audio file
        self.audio_path = audio_path

    def augment_all_audio(
        self,
        augmentations_to_perform: list[str],
        percent: float,
        vol_shift: int,
        time_shift: float,
        pitch_shift: int,
        freq_mask: int,
        time_mask: int,
        input_folder: str = "ml/data/cough_data/processed_audio",
        output_folder: str = "ml/data/cough_data/augmented_spectrograms",
    ) -> None:
        """Reads in all audio from the processed_audio folder converts to
        spectrograms, and augments the dataset by "percent" percentage.

        Args:
        augmentations_to_perform: A list of strings, each string is the name of an augmentation to perform.
          ex: ["CV", "TS", "PS", "FM", "TM"]
        percent: The percent of the dataset that should be augmented (eg. 0.5 => two original spectrograms for every augmented spectrogram)
        extracted_features: A boolean, False if you want to generate spectrograms, True if you want to generate FBANK or MFCC images.
        output_folder: The folder we are saving augmented spectrograms to. Defaults to cough_data/augmented_spectrograms.
        """
        # Process orignial dataset:
        processor = SpectrogramProcessor(
            audio_folder=input_folder, output_folder=output_folder
        )
        processor.process_all_images()

        # Augment dataset and generate images:
        for label in "positive", "negative":
            dir = os.listdir(os.path.join(input_folder, label))
            save_directory = os.path.join(
                    output_folder + "_" + "_".join([augmentation for augmentation in augmentations_to_perform]),
                    label
                )
            os.makedirs(save_directory, exist_ok=True)
            for filepath in dir[0 : round(len(dir) * percent)]:

                path_in = os.path.join(input_folder, label, filepath)
                audio, sr = librosa.load(path_in, sr=None)

                # Augmentation Audio
                if "CV" in augmentations_to_perform:
                    audio = self.change_volume(audio, sr, vol_shift)
                if "TS" in augmentations_to_perform:
                    audio = self.time_shift(audio, sr, time_shift)
                if "PS" in augmentations_to_perform:
                    audio = self.pitch_shift(audio, sr, pitch_shift)

                # Generate spectrogram
                spectrogram = processor.normalize_spectrogram(
                    processor.conv_to_spectrogram(
                        audio_clip=audio,
                        sample_rate=sr,
                    )
                )

                # Augment Spectrogram
                if "FM" in augmentations_to_perform:
                    spectrogram = self.freq_mask(spectrogram, freq_mask)
                if "TM" in augmentations_to_perform:
                    spectrogram = self.time_mask(spectrogram, time_mask)

                # Save image file
                save_path = os.path.join(save_directory, filepath[:-4] + "_aug.png")
                print(save_path)

                processor.save_spectrogram_image(
                    spectrogram=spectrogram, save_path=save_path
                )

    def change_volume(
        self, audio: np.ndarray, sr_in: int, db_change: int
    ) -> np.ndarray:
        """Returns the mel spectrogram after increasing/decreasing volume

        Args:
        db_change: Change in Volume in Decibels (Default kept small to 5)
        Returns: Mel Spectrogram after increasing/decreasing volume"""
        # Add the Decibel change
        augmented_audio = audio + db_change
        return augmented_audio

    def time_shift(self, audio: np.ndarray, sr_in: int, shift_max: float) -> np.ndarray:
        """Returns Mel Spectrogram by time shifting the audio

        Args: shift_max: Max % of length of audio sample to be shifted (Default
        kept small to only 0.1 or 10%)
        Returns: Mel Spectrogram after shifting time"""
        # Random (Uniform Dist.) shift up to 10% of total audio length
        shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))

        # Calculate time shift
        shifted_audio = np.roll(audio, shift)

        # Return Mel Spectrogram
        return shifted_audio

    def pitch_shift(self, audio: np.ndarray, sr_in: int, n_steps: int) -> np.ndarray:
        """Returns the Mel Spectrogram after pitch shifting

        Args: n_steps: Number of semitones for frequency to be shifted (default
        used 4 as found in research papers)
        Returns: Mel Spectrogram after shifting pitch"""

        # Pitch shift
        pitch_shifted_audio = librosa.effects.pitch_shift(
            y=audio, sr=sr_in, n_steps=n_steps
        )

        # Return Mel Spectrogram
        return pitch_shifted_audio

    def freq_mask(self, spectrogram: np.ndarray, freq_mask: int) -> np.ndarray:
        """Returns mel spectrogram after frequency masking

        Args:
        spectrogram: orginal spectrogram data, passed in as a np.ndarray
        freq_mask: Max no. of frequency bands that can be masked (used default value from research paper = 30).
        Returns: Mel Spectrogram after freq masking"""

        # Time masking
        freq_masking = T.FrequencyMasking(freq_mask)
        augmented_melspec = np.array(freq_masking(torch.tensor(spectrogram)))

        # Return Augmented Mel Spectrogram
        return augmented_melspec

    def time_mask(self, spectrogram: np.ndarray, time_mask: int) -> np.ndarray:
        """Returns mel spectrogram after time masking

        Args:
        spectrogram: orginal spectrogram data, passed in as a np.ndarray
        time_mask_param: No. of time stamps to mask (used default value from research paper = 30)

        Returns: Mel Spectrogram with Time Masking"""

        # Time masking
        time_masking = T.TimeMasking(time_mask)
        augmented_melspec = np.array(time_masking(torch.tensor(spectrogram)))

        # Return Augmented Mel Spectrogram
        return augmented_melspec

    ###### Visualization Functions: ######

    def plot_spectrogram(self, name, spectrogram) -> None:
        """Plots the spectrogram using Matplotlib

        Args:
            name: Augmentation Type
            spectrogram: Spectrogram data extracted from the augmented audio
        """
        # Plot spectrogram using matplotlib
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="inferno")
        plt.colorbar(label="Amplitude (dB)")
        plt.title(f"Spectrogram: {name}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_compare(
        self,
        audio_path: str,
        vol_shift: int,
        time_shift: float,
        pitch_shift: int,
        freq_mask: int,
        time_mask: int,
    ) -> None:
        """Plots spectrogram across all augmentations for comparison"""

        processor = SpectrogramProcessor()

        audio, sr = librosa.load(audio_path, sr=None)

        spectrogram = processor.normalize_spectrogram(
            processor.conv_to_spectrogram(
                audio_path=None, path_input=False, audio_clip=audio, sample_rate=sr
            )
        )
        self.plot_spectrogram("Original", spectrogram)

        audio1 = self.change_volume(audio, sr, vol_shift)
        spectrogram = processor.normalize_spectrogram(
            processor.conv_to_spectrogram(
                audio_path=None, path_input=False, audio_clip=audio1, sample_rate=sr
            )
        )
        self.plot_spectrogram("Volume Increased", spectrogram)

        audio2 = self.time_shift(audio, sr, time_shift)
        spectrogram = processor.normalize_spectrogram(
            processor.conv_to_spectrogram(
                audio_path=None, path_input=False, audio_clip=audio2, sample_rate=sr
            )
        )
        self.plot_spectrogram("Time Shifted", spectrogram)

        audio3 = self.pitch_shift(audio, sr, pitch_shift)
        spectrogram = processor.normalize_spectrogram(
            processor.conv_to_spectrogram(
                audio_path=None, path_input=False, audio_clip=audio3, sample_rate=sr
            )
        )
        self.plot_spectrogram("Pitch Shifted", spectrogram)

        # Generate clean spectrogram
        spectrogram = processor.normalize_spectrogram(
            processor.conv_to_spectrogram(
                audio_path=None, path_input=False, audio_clip=audio, sample_rate=sr
            )
        )

        # Augment Spectrogram
        spectro1 = self.freq_mask(spectrogram, freq_mask)
        self.plot_spectrogram("Frequency Masked", spectro1)

        spectro2 = self.time_mask(spectrogram, time_mask)
        self.plot_spectrogram("Time Masked", spectro2)

        # Finally, do full augmentation

        aug_audio = self.time_shift(audio1, sr, time_shift)
        aug_audio = self.pitch_shift(aug_audio, sr, pitch_shift)
        aug_spectrogram = processor.normalize_spectrogram(
            processor.conv_to_spectrogram(
                audio_path=None, path_input=False, audio_clip=aug_audio, sample_rate=sr
            )
        )
        aug_spectrogram = self.freq_mask(aug_spectrogram, freq_mask)
        aug_spectrogram = self.time_mask(aug_spectrogram, time_mask)


if __name__ == "__main__":

    # Load in all processed audio from ml/data/cough_data/processed_audio

    augment_proc = DataAugmentProcessor(audio_path=None)

    # Parameters
    percent = 0.5
    vol_shift = 5
    time_shift = 2
    pitch_shift = 25
    freq_mask = 30
    time_mask = 30

    augment_proc.augment_all_audio(
        ["CV", "TS", "PS", "FM", "TM"],
        percent,
        vol_shift,
        time_shift,
        pitch_shift,
        freq_mask,
        time_mask,
    )
    print("Augmentation Complete! Check your folders my friend.")

    augment_proc.plot_compare(
        "ml/data/cough_data/processed_audio/positive/0b1a540a-b6e7-4a2f-8796-28bd04554a36.wav",
        vol_shift,
        time_shift,
        pitch_shift,
        freq_mask,
        time_mask,
    )
