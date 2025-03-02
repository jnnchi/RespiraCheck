
"""
To run the server, you can use:
uvicorn main:app --reload

View the app at: http://localhost:8000
"""
import requests
from pydub import AudioSegment
import io
from fastapi import FastAPI, File, UploadFile

from ml.models.model_pipeline import ModelPipeline
from ml.models.model_handler import ModelHandler
from ml.data_processing.data_pipeline import DataPipeline

app = FastAPI()

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Receives an audio file upload."""
    file_format = file.filename.split(".")[-1]
    print(f"Received audio file: {file.filename}, Format: {file_format}")

    # Convert UploadFile to AudioSegment
    audio_bytes = await file.read()  # Read file bytes
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_format)

    # Call inference function
    data_pipeline = DataPipeline()
    model_handler = ModelHandler()
    model_pipeline = ModelPipeline(data_pipeline, model_handler)
    prediction = model_pipeline.make_single_inference(audio, "model_learning_rate_0.001_dropout_0.3_1739994880.920523.pth")

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