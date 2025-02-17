import pandas as pd
from IPython.display import Audio
from pydub import AudioSegment
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from data_processing.audio_processor import AudioProcessor

# Generate output folder of all processed audio
audio_proc = AudioProcessor(target_sample_rate=0.5, target_duration=10)
audio_proc.process_all_audio()

