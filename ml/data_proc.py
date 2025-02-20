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
from data_processing.extracted_features_processor import ExtractedFeaturesProcessor
from data_processing.data_pipeline import DataPipeline

# Generate output folder of all processed audio
audio_proc = AudioProcessor()
#audio_proc.process_all_audio()

spectroproc = SpectrogramProcessor()
#spectroproc.process_all_images()

extractproc = ExtractedFeaturesProcessor()
#extractproc.process_all_images()

datapipe = DataPipeline(test_size=0.2, val_size=0.3, audio_processor=audio_proc, image_processor=extractproc)
datapipe.process_all()

train_loader, val_loader, test_loader = datapipe.create_dataloaders(batch_size=32)
