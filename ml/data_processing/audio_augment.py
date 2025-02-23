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

  def get_mel_spectrogram(self, y, sr) -> np.ndarray:
    """Helper function similar to one in SpectrogramProcessor 
    to return Mel Spectrogram
    
    Args: y, sr = Extracted Audio from libros.load
    Returns: Mel Spectrogram"""
    # Convert to log scaled mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to decibels
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    #  Normalize Spectrogram
    spectrogram_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())

    # Return normalized Mel Spectrogram
    return spectrogram_norm

  def change_volume(self, db_change=5) -> np.ndarray:
     """Returns the mel spectrogram after increasing/decreasing volume
     
     Args: db_change: Change in Volume in Decibels (Default kept small to 5)
     Returns: Mel Spectrogram after increasing/decreasing volume """
     # Load the audio
     y, sr = librosa.load(self.audio_path)
     
     # Add the Decibel change
     augmented_audio = y + db_change
     
     # Return Mel Spectrogram
     return self.get_mel_spectrogram(augmented_audio, sr)

  def time_shift(self, shift_max=0.1) -> np.ndarray:
    """Returns Mel Spectrogram by time shifting the audio
    
    Args: shift_max: Max % of length of audio sample to be shifted (Default
    kept small to only 0.1 or 10%)
    Returns: Mel Spectrogram after shifting time"""
    # Load the audio
    y, sr = librosa.load(self.audio_path)

    # Random (Uniform Dist.) shift upto 10% of total audio length
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))

    # Calculate time shift
    shifted_audio = np.roll(y, shift)

    # Return Mel Spectrogram
    return self.get_mel_spectrogram(shifted_audio, sr)
  
  def pitch_shift(self, n_steps=4) -> np.ndarray:
    """Returns the Mel Spectrogram after pitch shifting
    
    Args: n_steps: Number of semitones for frequency to be shifted (default
    used 4 as found in research papers)
    Returns: Mel Spectrogram after shifting pitch"""
    # Load audio
    y, sr = librosa.load(self.audio_path)

    # Pitch shift
    pitch_shifted_audio = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    # Return Mel Spectrogram
    return self.get_mel_spectrogram(pitch_shifted_audio, sr)
  
  def freq_mask(self, freq_mask_param=30) -> np.ndarray:
    """Returns mel spectrogram after frequency masking
    
    Args: freq_mask_param: Max No of frequency bands that can be masked (used 
    default value from research paper)
    Returns: Mel Spectrogram after freq masking"""
    # Load the audio file with original sample rate 
    y, sr = librosa.load(self.audio_path, sr=None)

    # Convert to log scaled mel spectrogram
    spectrogram_norm = self.get_mel_spectrogram(y, sr)

    # Time masking
    freq_masking = T.FrequencyMasking(freq_mask_param)
    augmented_melspec = np.array(freq_masking(torch.tensor(spectrogram_norm)))

    # Return Augmented Mel Spectrogram
    return augmented_melspec
  
  def time_mask(self, time_mask_param=30) -> np.ndarray:
    """Returns mel spectrogram after time masking
    
    Args: time_mask_param: No of time stamps to mask (used default
    value from research paper)
    
    Returns: Mel Spectrogram with Time Masking"""
    # Load the audio file with original sample rate 
    y, sr = librosa.load(self.audio_path, sr=None)

    # Convert to log scaled mel spectrogram
    spectrogram_norm = self.get_mel_spectrogram(y, sr)

    # Time masking
    time_masking = T.TimeMasking(time_mask_param)
    augmented_melspec = np.array(time_masking(torch.tensor(spectrogram_norm)))

    # Return Augmented Mel Spectrogram
    return augmented_melspec

  def plot_spectrogram(self, name, spectrogram) -> None:
    """Plots the spectrogram using Matplotlib

    Args:
        name: Augmentation Type
        spectrogram: Spectrogram data extracted from the augmented audio
    """
    # Plot spectrogram using matplotlib
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Amplitude (dB)')
    plt.title(f"Spectrogram: {name}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()

  def plot_compare(self) -> None:
    """Plots spectrogram across all augmentations for comparison
    
    Args: None
    Returns: None"""
    # Load audio file
    y, sr = librosa.load(self.audio_path, sr=None)

    # Do Augmentations and Plot accordingly
    spectrogram = self.get_mel_spectrogram(y, sr) 
    self.plot_spectrogram("Normal", spectrogram)

    pitch_spectrogram = self.pitch_shift()
    self.plot_spectrogram("Pitch Shifted", pitch_spectrogram)

    timeshift_spectrogram = self.time_shift()
    self.plot_spectrogram("Time Shifted", timeshift_spectrogram)

    vol_spectrogram = self.change_volume()
    self.plot_spectrogram("Volume Increased", vol_spectrogram)

    freq_spectrogram = self.freq_mask()
    self.plot_spectrogram("Frequency Masked", freq_spectrogram)

    time_spectrogram = self.time_mask()
    self.plot_spectrogram("Time Masked", time_spectrogram)
  