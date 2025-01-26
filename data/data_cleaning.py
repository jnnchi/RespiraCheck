"""Processing cough data, including trimming beginning of clips, splititing all clips in two, and removing background noise"""
import os
from pydub import AudioSegment

def trim_beginning(file_path: str) -> None:
    """
    Trims the beginning 0.5 seconds of an mp3 clip.

    Args:
        path: a string representing the path to the audio clip.
    
    Returns:
        No return value, given audio clip will be overwritten.
    """
    audio = AudioSegment.from_file(file_path)

    trim_size = 0.5*1000 # Milliseconds

    trimmed_audio = audio[trim_size:]

    trimmed_audio.export(file_path, "mp3")


def split_audio(input_file_path: str, output_folder_path: str) -> None:
    """
    Splits an mp3 audio clip into two equal parts.

    Args:
        file_path: a string representing the path to the audio clip.
        save_path: a string representing the new save path of the audio clip, with no name.
    
    Returns:
        No return value, given audio clip will be overwritten.
    """
    audio = AudioSegment.from_file(input_file_path)

    trim_size = (audio.duration_seconds * 1000) / 2

    split_1 = audio[:trim_size]
    split_2 = audio[trim_size:]

    original_file_name = input_file_path.split("/")[-1]
    
    split_1_path = output_folder_path + original_file_name[:-4] + "_split_1.mp3"
    split_2_path = output_folder_path + original_file_name[:-4] + "_split_2.mp3"

    split_1.export(split_1_path, "mp3")
    split_2.export(split_2_path, "mp3")


if __name__ == "__main__":
    # Ensure the output directory exists
    output_path = "data/split_audio/"
    os.makedirs(output_path, exist_ok=True)
    
    # Iterate through files in the original_audio folder
    for filename in os.listdir("data/original_audio"):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            try:
                input_path = f"data/original_audio/{filename}"

                # Trim beginning of the clip, continue to save in original_audio
                trim_beginning(input_path)

                # Split into two parts and save in split_audio folder
                split_audio(input_path, output_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                break