"""
File that processes all data and saves to ourput folders
"""
from data_processing.audio_processor import AudioProcessor
from data_processing.spectrogram_processor import SpectrogramProcessor
from data_processing.extracted_features_processor import ExtractedFeaturesProcessor
from data_processing.audio_augment import DataAugmentProcessor
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    
    #### USE THIS FILE TO FILL YOUR DATA FOLDERS
    ####
    #### USAGE: 
    ####    - Comment out the functions you dont want to run. We reccomend 
    ####      you only run function one at a time, and comment out the rest.
    
    audio_proc = AudioProcessor()
    audio_proc.process_all_audio()

    spectroproc = SpectrogramProcessor(stft=False)
    spectroproc.process_all_images()

    # extractproc = ExtractedFeaturesProcessor(feature_type="fbank")
    #extractproc.process_all_images()


    # augment_proc = DataAugmentProcessor()

    # Parameters
    percent = 0.5
    vol_shift = 0
    time_shift = 0.4 # between 0.1-0.5 seconds since clip is only 5 secs long
    pitch_shift = 3 # between 1 to 4 semitones
    freq_mask = 10 # between 10-50
    time_mask = 30 # between 10-50

    # augment_proc.augment_all_audio(
    #     ["TS", "PS", "FM", "TM"], # Augmentations to apply, apply only 3 or less at once
    #     ["positive"], # Labels to augment, can be ["positive"], ["negative"], or ["positive", "negative"]
    #     percent,
    #     vol_shift,
    #     time_shift,
    #     pitch_shift,
    #     freq_mask,
    #     time_mask,
    # )

    ### Random Search for Augmentation Parameters - Uncomment to experiment with augmentation
    # percent = 0.5
    # time_shift = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # pitch_shift = [0, 1, 2, 3, 4]
    # freq_mask = [i for i in range(20, 41)]
    # time_mask = [i for i in range(10, 51)]

    # for i in range(10):
    #     ts = random.choice(time_shift)
    #     ps = random.choice(pitch_shift)
    #     freq = random.choice(freq_mask)
    #     time = random.choice(time_mask)
    #     augment_proc.augment_all_audio(
    #         ["TS", "PS", "FM", "TM"],
    #         percent,
    #         vol_shift=0,
    #         time_shift=ts,
    #         pitch_shift=ps,
    #         freq_mask=freq,
    #         time_mask=time,
    #         input_folder=processed_audio,
    #         output_folder=os.path.join(parent_dir, f"data/cough_data/aug_spec_{ts}_{ps}_{freq}_{time}")
    #     )
    #     print(f"Augmentation complete. TShift: {ts}, PShift: {ps}, FMask: {freq}, TMask: {time}")
