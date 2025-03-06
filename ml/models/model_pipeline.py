"""Model Pipeline Module.

This module provides the `ModelPipeline` class for handling the complete 
machine learning pipeline, including data preparation, model training, 
saving, and inference.

Dependencies:
    - Custom `DataPipeline` and `ModelHandler` classes.

TODO: - Implement data pipeline integration.
      - Define model training and inference logic.
"""
import io

from pydub import AudioSegment

class ModelPipeline:
    """Handles the machine learning pipeline from training to inference.

    This class provides methods for training a model, saving it, and 
    making single inferences.

    Attributes:
        data_pipeline (DataPipeline): The data pipeline instance.
        model_handler (ModelHandler): The model handler instance.
        model_path (str): Path to save or load the model.
    """

    def __init__(self, data_pipeline, model_handler):
        """Initializes the ModelPipeline.

        Args:
            data_pipeline (DataPipeline): The data pipeline instance.
            model_handler (ModelHandler): The model handler instance.
        """
        self.data_pipeline = data_pipeline
        self.model_handler = model_handler

    def make_single_inference(self, audio_bytes: bytes, audio_format: str) -> int:
        """Performs inference on a single audio file.

        Args:
            webm_audio (str): Path to the webm audio file.

        Returns:
            int: 0 or 1 for the predicted class.
        """
        # Convert to WAV if needed
        if audio_format.lower() in ["webm", "mp3"]:
            print(audio_format)
            temp_wav = io.BytesIO()
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
            audio.export(temp_wav, format="wav") 
            temp_wav.seek(0)
            audio = AudioSegment.from_file(temp_wav, format="wav")
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)

        image_tensor, image = self.data_pipeline.process_single_for_inference(audio)
        prediction = 0#self.model_handler.predict(image_tensor)

        return prediction, image # pil image
