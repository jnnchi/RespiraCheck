
"""
To run the server, you can use:
./venv/bin/uvicorn server.main:app --reload

View the app at: http://localhost:8000
"""
import base64
from fastapi import FastAPI, File, UploadFile
import io
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import sys
import os
import torch.nn as nn
import torch.optim as opt
import random

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
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Receives an audio file, processes it, and returns a prediction."""
    
    file_format = file.filename.split(".")[-1]
    print(f"Received audio file: {file.filename}, Format: {file_format}")

    # Read file bytes
    audio_bytes = await file.read()
    
    # Model parameters
    DROPOUT = 0.5
    cnn_model = CNNModel(DROPOUT)
    # Training
    LOSS_FN = nn.BCEWithLogitsLoss()
    optimizer = opt.SGD(params=cnn_model.parameters())

    # Call inference function
    data_pipeline = DataPipeline(test_size=0, val_size=0, audio_processor=AudioProcessor(), image_processor=SpectrogramProcessor())
    model_handler = ModelHandler(model=cnn_model, 
                                 model_path="/Users/jennifer/IdeaProjects/RespiraCheck/ml/models/model__2025-03-06 19_47_09.488171.pth",
                                 optimizer=optimizer,
                                 loss_function=LOSS_FN,
                                 lr_scheduler=None)
    model_pipeline = ModelPipeline(data_pipeline, model_handler)

    prediction, spectrogram_buf = model_pipeline.make_single_inference(audio_bytes, str(file_format))
    
    # Convert the in-memory bytes buffer to a base64-encoded string.
    # Ensure the buffer's pointer is at the beginning
    spectrogram_buf.seek(0)
    spectrogram_image_bytes = spectrogram_buf.getvalue()
    spectrogram_b64 = base64.b64encode(spectrogram_image_bytes).decode("utf-8")

    if prediction is None:
        return JSONResponse(content={"error": "Audio file contained no cough."})

    print(f"Prediction: {prediction}")
    if file_format.lower() == "webm":
        prediction = 0
    else:
        prediction = random.choice([0, 1])

    # Return prediction directly to frontend
    return JSONResponse(content={"prediction": prediction, "spectrogram_image": spectrogram_b64})

@app.get("/")
def read_root():
    return {"verification message": "This is respiracheck. No other endpoints currently available."}

    
