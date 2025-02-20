
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
    model_pipeline = ModelPipeline()
    model_pipeline.make_single_inference(audio)

    return {"filename": file.filename, "format": file_format}

@app.get("/")
def read_root():
    return {"verification message": "This is respiracheck. No other endpoints currently available."}