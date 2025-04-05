"""
Single inference for AWS Gateway API HTTP endpoint
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.models.model_pipeline import ModelPipeline
from ml.models.model_handler import ModelHandler
from ml.models.cnnLSTM_model import CNNLSTM as CNNModel
from ml.data_processing.data_pipeline import DataPipeline
from ml.data_processing.audio_processor import AudioProcessor
from ml.data_processing.spectrogram_processor import SpectrogramProcessor
import torch.optim as opt
import torch.nn as nn

def lambda_handler():
    audio_bytes = open(
        "src/cough-heavy.wav",
        "rb",
    ).read()

    # Static hyperparameters
    EPOCHS = 20

    # Learning rate scheduler
    STEPS_PER_LR_DECAY = 20
    LR_DECAY = 0.5

    cnn_model = CNNModel()
    # Training
    LOSS_FN = nn.BCEWithLogitsLoss()
    optimizer = opt.SGD(params=cnn_model.parameters())

    # Call inference function
    data_pipeline = DataPipeline(
        test_size=0,
        val_size=0,
        audio_processor=AudioProcessor(),
        image_processor=SpectrogramProcessor(),
    )
    model_handler = ModelHandler(
        model=cnn_model,
        model_path="ml/models/krish_TEMP_model.pth",
        optimizer=optimizer,
        loss_function=LOSS_FN,
        lr_scheduler=None,
    )
    model_pipeline = ModelPipeline(data_pipeline, model_handler)

    prediction, spectrogram_buf = model_pipeline.make_single_inference(
        audio_bytes, "wav"
    )
    return {
        'status code': 200,
        'body': json.dumps(prediction, spectrogram_buf)

    }

if __name__ == "__main__":
    # Test run
    print(lambda_handler())


