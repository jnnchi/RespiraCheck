import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from .image_processor import ImageProcessor


class ExtractedFeaturesProcessor(ImageProcessor):
    def __init__(self, audio_folder="ml/data/cough_data/processed_audio", output_folder="ml/data/cough_data/", feature_type="fbank"):
        super().__init__(audio_folder, output_folder)
        self.feature_type = feature_type.lower()
        self.output_folder = os.path.join(output_folder, self.feature_type)

    def process_all_images(self):
        for label in ["positive", "negative"]: 
            audio_dir = os.path.join(self.audio_folder, label)  
            output_dir = os.path.join(self.output_folder, label)

            # Make output folder if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            for filename in os.listdir(audio_dir):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(audio_dir, filename)

                    # Process file to generate and save spectrogram
                    if self.feature_type == "fbank":
                        self.process_single_audio_to_fbank(audio_path, output_dir)
                    elif self.feature_type == "mfcc":
                        self.process_single_audio_to_mfcc()
    
    def process_single_audio_to_mfcc(self):
        pass

    def process_single_audio_to_fbank(self, audio_path: str, output_dir: str):
        """
        Extracts FBANK features from a single audio, saves to output_dir
        """
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        fbank_image_path = os.path.join(output_dir, filename)
        fbank_features = self.fbank(audio_path, wintype="hamming")
        
        # Save to image
        plt.figure(figsize=(4, 4))
        plt.imshow(fbank_features.T, cmap="gray", origin="lower", aspect="auto")
        plt.axis("off")
        plt.savefig(fbank_image_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def fbank(self,
        audio_path,
        samplerate=48000,
        winlen=0.025,
        winstep=0.01,
        nfilt=40,
        nfft=512,
        lowfreq=0,
        highfreq=None,
        preemph=0.97,
        wintype="hamming",
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
        frames = frames.copy() # make it writable

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

        return fbank_features

