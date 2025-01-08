"""Processing cough data, including trimming beginning of clips, splititing all clips in two, and removing background noise"""

from pydub import AudioSegment

def trim_beginning(path: str) -> None:
    """
    Trims the beginning 0.5 seconds of an mp3 clip.

    Args:
        path: a string representing the path to the audio clip.
    
    Returns:
        No return value, given audio clip will be overwritten.
    """
    audio = AudioSegment.from_file(path)

    trim_size = 0.5*1000 # Milliseconds

    trimmed_audio = audio[trim_size:]

    trimmed_audio.export(path, "mp3")


def split_audio(file_path: str) -> None:
    """
    Splits an mp3 audio clip into two equal parts.

    Args:
        file_path: a string representing the path to the audio clip.
        save_path: a string representing the new save path of the audio clip, with no name.
    
    Returns:
        No return value, given audio clip will be overwritten.
    """
    audio = AudioSegment.from_file(file_path)

    trim_size = (audio.duration_seconds * 1000) / 2

    split_1 = audio[:trim_size]
    split_2 = audio[trim_size:]

    original_file_name = file_path.split("/")[-1]
    
    split_1_path = "split_audio/" + original_file_name[:-4] + "_split_1.mp3"
    split_2_path = "split_audio/" + original_file_name[:-4] + "_split_2.mp3"

    split_1.export(split_1_path, "mp3")
    split_2.export(split_2_path, "mp3")