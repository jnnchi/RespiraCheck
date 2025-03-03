
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

from fastapi import FastAPI, File, UploadFile
import io
from pydub import AudioSegment
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.models.model_pipeline import ModelPipeline
from ml.models.model_handler import ModelHandler
from ml.models.cnn_model import CNNModel
from ml.data_processing.data_pipeline import DataPipeline
from ml.data_processing.audio_processor import AudioProcessor
from ml.data_processing.spectrogram_processor import SpectrogramProcessor

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Receives an audio file, processes it, and returns a prediction."""
    
    file_format = file.filename.split(".")[-1]
    print(f"Received audio file: {file.filename}, Format: {file_format}")

    # Read file bytes
    audio_bytes = await file.read()
    
    # Convert UploadFile to AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_format)

    # Call inference function
    data_pipeline = DataPipeline(test_size=0, val_size=0, audio_processor=AudioProcessor(), image_processor=SpectrogramProcessor())
    model_handler = ModelHandler(model=CNNModel(), 
                                 model_path="/Users/jennifer/IdeaProjects/RespiraCheck/ml/models/model_learning_rate_0.001_dropout_0.3_1739995033.834346.pth",
                                 optimizer=None,
                                 loss_function=None,
                                 lr_scheduler=None)
    model_pipeline = ModelPipeline(data_pipeline, model_handler)

    prediction = 0  # Mock prediction for now
    # prediction = model_pipeline.make_single_inference(audio_bytes)  # Uncomment this when ready

    print(f"Prediction: {prediction}")

    # Return prediction directly to frontend
    return {"prediction": prediction}

@app.get("/")
def read_root():
    return {"verification message": "This is respiracheck. No other endpoints currently available."}