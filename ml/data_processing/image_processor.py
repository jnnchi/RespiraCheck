"""
Abstract ImageProcessor class, which is depended on by DataPipeline

This is the parent class of SpectrogramProcessor and ExtractedFeaturesProcessor
"""

from abc import ABC, abstractmethod

class ImageProcessor(ABC):

    def __init__(self, audio_folder: str, output_folder: str):
        self.audio_folder = audio_folder
        self.output_folder = output_folder

    @abstractmethod
    def process_all_images(self):
        pass  
    