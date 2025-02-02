"""
conv_to_spectro.py

Run this file to convert all audio .mp3 files within the data/split_audio/ folder
into spectrogram images and save the images to the data/spectrograms/ folder.

Functions:
- save_mel_spectrogram(): saves a single audio file as a mel spectrogram.
- parse_filename(): parses data labels, such as diagnosis, from audio file names.
- create_all_spectrograms(): iterates through all audio files to save all of 
them as spectrograms; creates the labels.csv file using the file names.
"""

import os
import csv
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
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        #print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")


def parse_filename(filename):
    """
    Function to parse the filename and extract labels.
    """
    parts = filename.split("_")
    parts[1] = parts[1].split(",")
    parts[1][0] = parts[1][0].lower().strip()
    
    return parts[1]

def create_all_spectrograms(input_folder, output_folder):
    """
    Function to iterate through all files, save as spectrogram image, and create labels CSV.
    """
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "labels.csv")

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(["filename", "diagnosis", "sound_type", "location", "age", "gender"])
        
        for filename in os.listdir(input_folder):
            if filename.endswith(".wav") or filename.endswith(".mp3"):  # is audio
                input_path = os.path.join(input_folder, filename)

                diagnosis, sound_type, location, age, gender = parse_filename(filename)

                # Construct output directory and path
                diagnosis_folder = os.path.join(output_folder, diagnosis)  # Relative path
                os.makedirs(diagnosis_folder, exist_ok=True)  # Ensure the directory exists

                output_path = os.path.join(diagnosis_folder, os.path.splitext(filename)[0] + ".png")

                # Save spectrogram
                save_mel_spectrogram(input_path, output_path)
                
                writer.writerow([filename, diagnosis, sound_type, location, age, gender])

if __name__=="__main__":
    input_folder = "data/stethoscope_data/split_audio"
    output_folder = "data/stethoscope_data/spectrograms"
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    create_all_spectrograms(input_folder, output_folder)