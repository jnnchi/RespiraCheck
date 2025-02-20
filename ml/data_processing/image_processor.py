from abc import ABC, abstractmethod

# Define an abstract class
class ImageProcessor(ABC):

    def __init__(self, audio_folder: str, output_folder: str):
        self.audio_folder = audio_folder
        self.output_folder = output_folder

    @abstractmethod
    def process_all_images(self):
        pass  # This is an abstract method, no implementation here.
    