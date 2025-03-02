"""Audio Processing Module.

This module provides the `AudioProcessor` class for processing audio files,
including noise reduction, silence removal, and format conversion.

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
import soundfile as sf
from scipy.signal import butter, lfilter
import json
import librosa.feature
from tqdm import tqdm


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
        target_sample_rate=48000,
        target_duration=5, # in seconds
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
            ]
        )


    def process_all_audio(self) -> None:
        """Processes all audio files in a given directory."""
        for label in ["positive", "negative"]:
            labeled_input_dir = os.path.join(self.input_folder, label)
            labeled_output_dir = os.path.join(self.output_folder, label)

            # Create labeled output folder if it doesn't exist
            os.makedirs(labeled_output_dir, exist_ok=True)

            for filename in tqdm(os.listdir(labeled_input_dir)):
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
        audio = self.standardize_duration(audio)

        # overwrite the original file with the cleaned version
        audio.export(output_audio_path, format="wav")

        # save metadata
        self.save_metadata(output_audio_path, filename)

        return 0

    def process_single_audio_for_inference(self, audio: AudioSegment) -> AudioSegment:
        """Processes a single audio file."""

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

        audio = self.standardize_duration(audio)

        return audio


    def save_metadata(self, audio_path, filename) -> None:
        """
        Saves metadata for the processed audio file.
        """
        # Save metadata for this processed audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        channels = 1
        new_row = pd.DataFrame(
            [[filename, duration, sr, channels]],
            columns=[
                "filename",
                "duration",
                "sample_rate",
                "channels",
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

    def standardize_duration(self, audio: AudioSegment) -> AudioSegment:
        """Standardizes the duration of an audio file to exactly 10 seconds.

        - If the audio is longer than 10 seconds, it will be trimmed.
        - If the audio is shorter than 10 seconds, it will be padded with silence.

        Args:
            audio (AudioSegment): The input audio file.

        Returns:
            AudioSegment: The standardized 10-second audio.
        """
        target_duration = self.target_duration * 1000  # seconds to milliseconds
        current_duration = len(audio)  # Current duration in milliseconds

        if current_duration > target_duration:
            # Trim the audio if it's too long
            standardized_audio = audio[:target_duration]
        elif current_duration < target_duration:
            # Pad with silence if it's too short
            silence = AudioSegment.silent(duration=target_duration - current_duration)
            standardized_audio = audio + silence
        else:
            # Already 10 seconds, return as is
            standardized_audio = audio

        return standardized_audio

    def bandpass_filter(self, y, sr, lowcut=300, highcut=4000, order=6):
        """
        Apply a bandpass filter to isolate frequencies between lowcut and highcut.
        Used to reduce background noise and focus on cough sounds.
        (specifically used for the minimize_speech method)

        Returns:
            np.ndarray: Filtered audio signal.
        """
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a, c = butter(order, [low, high], btype="band")
        return lfilter(b, a, y)

    def is_muffled(self, audio_path, threshold=0.5):
        """
        Determines whether an audio file is muffled based on spectral flatness.

        Spectral flatness measures how noise-like or tonal an audio signal is.
        A lower spectral flatness value indicates a more tonal sound, while a higher value suggests a noise-like sound.
        Muffled audio typically has low high-frequency energy, resulting in lower spectral flatness.

        Args:
            audio_path (str): Path to the audio file to be analyzed.
            threshold (float, optional): The spectral flatness threshold above which audio is considered muffled.
                                         Default is 0.5.

        Returns:
            bool: True if the audio is considered muffled, False otherwise.
        """
        y, sr = librosa.load(audio_path, sr=None)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        return flatness > threshold

    def highpass_filter(self, y, sr, cutoff=800):
        """
            Applies a high-pass Butterworth filter to remove low-frequency noise from the audio signal.

            This filter helps enhance clarity by eliminating frequencies below the specified cutoff,
            which is useful for reducing muffling and preserving important speech or cough frequencies.

            Args:
                y (np.ndarray): The input audio signal.
                sr (int): The sampling rate of the audio.
                cutoff (float, optional): The cutoff frequency in Hz. Default is 800 Hz.

            Returns:
                np.ndarray: The high-pass filtered audio signal.
        """

        b, a, c = butter(6, cutoff / (0.5 * sr), btype='high')
        return lfilter(b, a, y)

    def demuffle_audio(self, y, sr):
        """
           Reduces muffling in an audio signal by applying a bandpass filter
           and a high-pass filter.

           This method helps enhance clarity by removing low-frequency noise
           and preserving cough frequencies.

           Args:
               y (np.ndarray): The input audio signal.
               sr (int): The sampling rate of the audio.

           Returns:
               np.ndarray: The filtered audio signal with reduced muffling.
        """

        y_filtered, sr = self.bandpass_filter(y, sr, lowcut=800)
        y_filtered = self.highpass_filter(y_filtered, sr, cutoff=800)

        return y_filtered

    def minimize_speech(self ,audio_path, output_path="isolated_cough.wav (subject to change)", duration=None):
        """
        Minimizes speech while preserving cough sounds using HPSS, soft masking, and bandpass filtering.
        """
        y, sr = librosa.load(audio_path, sr=None, duration=duration)

        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1, 5))

        y_cough_filtered = self.bandpass_filter(y_percussive, sr)

        S_full, phase = librosa.magphase(librosa.stft(y_cough_filtered))

        S_filter = librosa.decompose.nn_filter(S_full,
                                            aggregate=np.median,
                                            metric='cosine',
                                            width=int(librosa.time_to_frames(2, sr=sr)))

        S_filter = np.minimum(S_full, S_filter)

        margin_speech, margin_cough = 2, 10
        power = 2

        mask_speech = librosa.util.softmask(S_filter,
                                            margin_speech * (S_full - S_filter),
                                            power=power)

        mask_cough = librosa.util.softmask(S_full - S_filter,
                                        margin_cough * S_filter,
                                        power=power)

        S_cough = mask_cough * S_full

        y_cough = librosa.istft(S_cough * phase)

        sf.write(output_path, y_cough, sr)
        print(f"✅ Isolated cough saved at: {output_path}")

        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 6))

        idx = slice(*librosa.time_to_frames([0, min(5, len(y) / sr)], sr=sr))

        img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                                    y_axis='log', x_axis='time', sr=sr, ax=ax[0])
        ax[0].set(title='Full Percussive Spectrum')
        ax[0].label_outer()

        librosa.display.specshow(librosa.amplitude_to_db(S_filter[:, idx], ref=np.max),
                                y_axis='log', x_axis='time', sr=sr, ax=ax[1])
        ax[1].set(title='Background Speech Estimate')
        ax[1].label_outer()

        librosa.display.specshow(librosa.amplitude_to_db(S_cough[:, idx], ref=np.max),
                                y_axis='log', x_axis='time', sr=sr, ax=ax[2])
        ax[2].set(title='Isolated Cough (Speech Reduced)')

        fig.colorbar(img, ax=ax)
        plt.show()

        return y_cough, sr
