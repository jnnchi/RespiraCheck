import pandas as pd
from IPython.display import Audio
from pydub import AudioSegment
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from data_processing.audio_processor import AudioProcessor
from data_processing.spectrogram_processor import SpectrogramProcessor
from data_processing.data_pipeline import DataPipeline

# Generate output folder of all processed audio
audio_proc = AudioProcessor(target_sample_rate=0.5, target_duration=10)
audio_proc.process_all_audio()

spectroproc = SpectrogramProcessor()
#spectroproc.process_all_spectrograms()

#datapipe = DataPipeline(test_size=0.2, val_size=0.3, audio_processor=audio_proc, spectrogram_processor=spectroproc, metadata_df=None, metadata_path="data/cough_data/metadata.csv")

#train_loader, val_loader, test_loader = datapipe.create_dataloaders(batch_size=32)
