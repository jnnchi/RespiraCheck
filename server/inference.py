"""
single inference testing
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.models.model_pipeline import ModelPipeline
from ml.models.model_handler import ModelHandler
from ml.models.cnn_model import CNNModel
from ml.data_processing.data_pipeline import DataPipeline
from ml.data_processing.audio_processor import AudioProcessor
from ml.data_processing.spectrogram_processor import SpectrogramProcessor


if __name__=="__main__":
    audio_bytes = open("/Users/jennifer/IdeaProjects/RespiraCheck/ml/data/cough_data/processed_audio/negative/ecy0t93dh8TS3yPI4YJZDSuDFAi2_1.wav", "rb").read()

    # Call inference function
    data_pipeline = DataPipeline(test_size=0, val_size=0, audio_processor=AudioProcessor(), image_processor=SpectrogramProcessor())
    model_handler = ModelHandler(model=CNNModel(), 
                                 model_path="/Users/jennifer/IdeaProjects/RespiraCheck/ml/models/model_learning_rate_0.001_dropout_0.3_1739995033.834346.pth",
                                 optimizer=None,
                                 loss_function=None,
                                 lr_scheduler=None)
    model_pipeline = ModelPipeline(data_pipeline, model_handler)

    prediction, spectrogram_buf = model_pipeline.make_single_inference(audio_bytes, "wav")
    print(prediction)
