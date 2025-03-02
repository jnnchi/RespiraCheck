"""
Testing file for data_pipeline.py.
Run this file to generate all output data folders on disk and load them into dataloaders in memory.
"""
from data_processing.audio_processor import AudioProcessor
from data_processing.spectrogram_processor import SpectrogramProcessor
from data_processing.extracted_features_processor import ExtractedFeaturesProcessor
from data_processing.audio_augment import DataAugmentProcessor
#from data_processing.data_pipeline import DataPipeline


# GENERATE OUTPUT DATA FOLDERS
audio_proc = AudioProcessor()
#audio_proc.process_all_audio()

spectroproc = SpectrogramProcessor(stft=False)
#spectroproc.process_all_images()

extractproc = ExtractedFeaturesProcessor(feature_type="fbank")
#extractproc.process_all_images()

augment_proc = DataAugmentProcessor(audio_path=None)

# Parameters
percent = 0.5
vol_shift = 0
time_shift = 0.4 # between 0.1-0.5 seconds since clip is only 5 secs long
pitch_shift = 3 # between 1 to 4 semitones
freq_mask = 10 # between 10-50
time_mask = 30 # between 10-50

augment_proc.augment_all_audio(
    ["TS", "PS", "FM", "TM"], # Augmentations to apply, apply only 3 or less at once
    ["positive"], # Labels to augment, can be ["positive"], ["negative"], or ["positive", "negative"]
    percent,
    vol_shift,
    time_shift,
    pitch_shift,
    freq_mask,
    time_mask,
)

#datapipe = DataPipeline(test_size=0.2, val_size=0.3, audio_processor=audio_proc, image_processor=spectroproc)
#datapipe.process_all()

#train_loader, val_loader, test_loader = datapipe.create_dataloaders(batch_size=32)
