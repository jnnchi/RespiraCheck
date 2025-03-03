
"""
To run the server, you can use:
./venv/bin/uvicorn server.main:app --reload

View the app at: http://localhost:8000
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from pydub import AudioSegment
import io
from fastapi import FastAPI, File, UploadFile

from ml.models.model_pipeline import ModelPipeline
from ml.models.model_handler import ModelHandler
from ml.models.cnn_model import CNNModel
from ml.data_processing.data_pipeline import DataPipeline
from ml.data_processing.audio_processor import AudioProcessor
from ml.data_processing.spectrogram_processor import SpectrogramProcessor

app = FastAPI()

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Receives an audio file upload."""
    print("hi")

    file_format = file.filename.split(".")[-1]
    print(f"Received audio file: {file.filename}, Format: {file_format}")

    audio_bytes = await file.read()  # Read file bytes
    #audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_format)

    # Call inference function
    data_pipeline = DataPipeline(test_size=0, val_size=0, audio_processor=AudioProcessor(), image_processor=SpectrogramProcessor())
    model_handler = ModelHandler(model=CNNModel(), 
                                 model_path="/Users/jennifer/IdeaProjects/RespiraCheck/ml/models/model_learning_rate_0.001_dropout_0.3_1739995033.834346.pth",
                                 optimizer=None,
                                 loss_function=None,
                                 lr_scheduler=None)
    model_pipeline = ModelPipeline(data_pipeline, model_handler)
    prediction = model_pipeline.make_single_inference(audio_bytes, model_name="model_learning_rate_0.001_dropout_0.3_1739994880.920523.pth")

    # Post the inference to inference URL
    requests.post("http://localhost:8000/inference", json={"prediction": prediction})
     
    return {"prediction": prediction}

@app.post("/inference")
async def inference(prediction: int):
    """Receives the prediction from the model."""
    print(f"Received prediction: {prediction}")
    return {"message": "Prediction received - {prediction}"}

@app.get("/")
def read_root():
    return {"verification message": "This is respiracheck. No other endpoints currently available."}