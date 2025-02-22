
"""
To run the server, you can use:
uvicorn main:app --reload

View the app at: http://localhost:8000
"""
import requests
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Receives an audio file upload."""
    file_format = file.filename.split(".")[-1]
    print(f"Received audio file: {file.filename}, Format: {file_format}")

    # Save the file locally 
    with open(f"uploaded_audio.{file_format}", "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename, "format": file_format}

@app.get("/")
def read_root():
    return {"verification message": "This is respiracheck. No other endpoints currently available."}