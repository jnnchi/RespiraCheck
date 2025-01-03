import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(audio_file, output_file):
    """
    Function to convert audio to mel spectrogram and save as an image.
    """
    try:
        # Load audio file into:
        # y - 1d np.ndarray with amplitudes over time. length is duration of audio * sr
        # sr - int sampling rate of audio signal
        y, sr = librosa.load(audio_file, sr=None)
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000) # 2d array of mel scaled freq components of audio over time
        mel_spec_decibel = librosa.power_to_db(mel_spec, ref=np.max) # converted to decibels
        
        # Save spectrogram as an image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_decibel, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.axis('off')  # Remove axes for a cleaner image
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")


def create_all_spectrograms(input_folder, output_folder):
    """
    Function to iterate through all files and save as spectrogram image.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"): # is audio
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            save_mel_spectrogram(input_path, output_path)

if __name__=="__main__":
    input_folder = "data/split_audio"
    output_folder = "data/spectrograms"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    create_all_spectrograms(input_folder, output_folder)