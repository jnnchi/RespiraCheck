"""Audio Processing Module.

This module provides the `AudioProcessor` class for processing audio files,
including noise reduction, silence removal, and format conversion.

Dependencies:
    - pandas
    - pydub

TODO: - Implement audio processing logic.
      - Include error handling for file operations.
"""

import os
import pandas as pd
import numpy as np
from pydub import AudioSegment, silence
import noisereduce as nr
from pydub.silence import detect_nonsilent
import librosa
import librosa.display
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import json
import shutil


class AudioProcessor:
    """Processes audio files, including noise reduction and silence removal.

    This class provides methods for processing multiple audio files,
    converting audio formats, and applying preprocessing techniques.

    Attributes:
        input_folder (str): Path to the folder containing input audio files.
        target_sample_rate (float): Desired sample rate for audio processing.
        target_duration (float): Target duration (in seconds) for each audio file.
        metadata_df (pd.DataFrame): DataFrame containing metadata for the audio files.
    """

    def __init__(
        self,
        target_sample_rate: float,
        target_duration: float,
        input_folder="ml/data/cough_data/original_data",
        output_folder="ml/data/cough_data/processed_audio",
    ):
        """Initializes the AudioProcessor.

        Args:
            input_folder (str): Path to the folder containing input audio files.
            target_sample_rate (float): Desired sample rate for audio processing.
            target_duration (float): Target duration (in seconds) for each audio file.
            metadata_df (pd.DataFrame): DataFrame containing metadata for the audio files.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration

        # create metadata dataframe
        self.metadata_df = pd.DataFrame(
            columns=[
                "filename",
                "duration",
                "sample_rate",
                "channels",
                "fbank_features",
            ]
        )


    def process_all_audio(self) -> None:
        """Processes all audio files in a given directory."""
        for label in ["positive", "negative"]: 
            labeled_input_dir = os.path.join(self.input_folder, label) 
            labeled_output_dir = os.path.join(self.output_folder, label)

            # Create labeled output folder if it doesn't exist
            os.makedirs(labeled_output_dir, exist_ok=True)

            for filename in os.listdir(labeled_input_dir)[:100]:
                if filename.endswith((".wav", ".mp3")):
                    audio_path = os.path.join(labeled_input_dir, filename)
                    output_audio_path = os.path.join(self.output_folder, label, filename)
            
                    print(f"Processing: {audio_path}")
                    self.process_single_audio(audio_path, output_audio_path)


        # Save metadata to a CSV file
        metadata_path = os.path.join(self.output_folder, "metadata.csv")
        self.metadata_df.to_csv(metadata_path, index=False)


    def process_single_audio(self, input_audio_path, output_audio_path, fbank=False) -> None:
        """Processes a single audio file."""

        filename = os.path.splitext(os.path.basename(input_audio_path))[0]

        # convert to wav if it isn't already
        if input_audio_path.endswith(".mp3"):
            self.conv_to_wav(input_audio_path, input_audio_path, "mp3")
        elif input_audio_path.endswith(".webm"):
            self.conv_to_wav(input_audio_path, input_audio_path, "webm")

        audio = AudioSegment.from_file(input_audio_path)

        # remove sections of no coughs
        audio = self.remove_no_cough(audio)
        if not audio:
            print("No cough detected. Skipping.")
            return 1
        
        # remove silences (may pass in non_silent_chunks into remove_silences)
        audio = self.remove_silences(audio)
        if not audio:
            print("Clip is silent. Skipping.")
            return 1

        # reduce noise
        audio = self.reduce_noise(audio)

        # overwrite the original file with the cleaned version
        audio.export(output_audio_path, format="wav")

        # save metadata
        self.save_metadata(input_audio_path, filename, fbank)
    
        return 0

    def save_metadata(self, input_audio_path, filename, fbank) -> None:
        """
        Saves metadata for the processed audio file.
        """
        # Save metadata for this processed audio
        y, sr = librosa.load(input_audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        channels = 1
        fbank_features = self.fbank(input_audio_path) if fbank else None
        new_row = pd.DataFrame(
            [[filename, duration, sr, channels, fbank_features]],
            columns=[
                "filename",
                "duration",
                "sample_rate",
                "channels",
                "fbank_features",
            ],
        )
        self.metadata_df = pd.concat([self.metadata_df, new_row], ignore_index=True)


    def get_labeled_path(self, filename: str) -> str:
        """
        Sorts audio files into positive or negative depending on their json annotation. 

        Args:
            filename (str): Name of the audio file (ex: 49f7f1de-5199-4291-b906-f058a8dc74d9)
       
        Returns:
            str: Path to the output folder (data/cough_data/processed_audio/positive or negative)
        """

        positive_folder = os.path.join(self.output_folder, "positive")
        negative_folder = os.path.join(self.output_folder, "negative")
        os.makedirs(positive_folder, exist_ok=True)
        os.makedirs(negative_folder, exist_ok=True)

        json_path = f"{self.input_folder}/{filename}.json"
        
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                    status = data.get("status", "").lower()
                    
                    if status == "covid-19":
                        return positive_folder
                    elif status == "healthy":
                        return negative_folder
                    else: return "none"
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {json_path}")
                    return "none"
        else:
            return "none"
        

    def conv_to_wav(self, audio_path: str, wav_path: str, file_type: str) -> None:
        """Converts an audio file to WAV format.

        Args:
            audio_path (str): Path to the audio file.
        """
        if file_type == "mp3":
            audio = AudioSegment.from_file(audio_path)
        elif file_type == "webm":
            # Ensure ffmpeg is correctly installed and set up
            AudioSegment.converter = "ffmpeg"  

            # Load the audio file
            audio = AudioSegment.from_file(audio_path, format="webm")
        
        # save wav file to output folder
        audio.export(wav_path, format="wav")


    def remove_silences(self, audio: AudioSegment) -> AudioSegment | None:
        """Removes silences from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            0 if non-silent chunks are found, 1 otherwise. When it returns 1, we should skip rest of data processing
        """
        non_silent_chunks = silence.split_on_silence(
            audio, min_silence_len=800, silence_thresh=-40
        )

        if non_silent_chunks:
            processed_audio = sum(non_silent_chunks)
            return processed_audio
        else:
            return None


    def reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        """Reduces background noise in an audio file.

        Args:
            input_audio_path (str): Path to the audio file.
        """

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
        # Normalize samples to range [-1.0, 1.0] (standardizes input for noise reduction)
        max_val = np.max(np.abs(samples))
        if max_val != 0:
            samples /= max_val
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(
            y=samples, sr=audio.frame_rate, stationary=False, prop_decrease=0.8
        )
        
        # Normalize the noise-reduced audio to restore amplitude
        max_reduced = np.max(np.abs(reduced_noise))
        if max_reduced != 0:
            normalized_reduced_noise = reduced_noise / max_reduced
        else:
            normalized_reduced_noise = reduced_noise

        # Need to convert back to int format (required by wav) cuz normalization turns into float
        # audio.sample_width == 4 -> scale to int32 range
        int_samples = (normalized_reduced_noise * 2147483647).astype(np.int32)
        
        
        # Create a new AudioSegment with the processed audio data
        processed_audio = AudioSegment(
            int_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels,
        )
        
        return processed_audio


    def remove_no_cough(self, audio: AudioSegment) -> AudioSegment | None:
        """Removes non-cough segments from an audio file.

        Args:
            audio_path (str): Path to the audio file.
        """
        min_silence_len = 500
        silence_thresh = -30

        non_silent_chunks = detect_nonsilent(audio, min_silence_len, silence_thresh)

        if not non_silent_chunks: # all chunks are nonsilent, no cough detected
            return None
        else:
            return audio


    def fbank(
        audio_path,
        samplerate=16000,
        winlen=0.025,
        winstep=0.01,
        nfilt=40,
        nfft=512,
        lowfreq=0,
        highfreq=None,
        preemph=0.97,
        wintype="hamming",
        grayscale=False,
        save_image=False,
        image_path="fbank_image.png",
    ):
        """Compute Mel-filterbank energy features and optionally convert to a grayscale image.

        :param audio_path: Path to the audio file.
        :param samplerate: Sample rate of the signal.
        :param winlen: Window length in seconds.
        :param winstep: Step size between windows in seconds.
        :param nfilt: Number of Mel filters.
        :param nfft: FFT size.
        :param lowfreq: Lowest frequency in Mel filters.
        :param highfreq: Highest frequency in Mel filters.
        :param preemph: Pre-emphasis factor.
        :param wintype: Window function type.
        :param grayscale: Whether to convert the filterbank to a grayscale image.
        :param save_image: Whether to save the grayscale image.
        :param image_path: File path to save the image.
        :return: Filterbank features (2D numpy array).
        """
        signal, samplerate = librosa.load(audio_path, sr=samplerate)

        highfreq = highfreq or samplerate / 2

        signal = np.append(signal[0], signal[1:] - preemph * signal[:-1])

        frame_length = int(winlen * samplerate)
        frame_step = int(winstep * samplerate)
        frames = librosa.util.frame(
            signal, frame_length=frame_length, hop_length=frame_step
        ).T

        if wintype == "hamming":
            window = np.hamming(frame_length)
        elif wintype == "hann":
            window = np.hanning(frame_length)
        else:
            window = np.ones(frame_length)
        frames *= window

        mag_frames = np.abs(np.fft.rfft(frames, n=nfft))
        pow_frames = (1.0 / nfft) * (mag_frames**2)

        mel_filters = librosa.filters.mel(
            sr=samplerate, n_fft=nfft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq
        )
        fbank_features = np.dot(pow_frames, mel_filters.T)
        fbank_features = np.where(
            fbank_features == 0, np.finfo(float).eps, fbank_features
        )

        if grayscale or save_image:
            plt.figure(figsize=(4, 4))
            plt.imshow(fbank_features.T, cmap="gray", origin="lower", aspect="auto")
            plt.axis("off")
            if save_image:
                plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close()

        return fbank_features
