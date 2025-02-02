
"""
To run the server, you can use:
uvicorn main:app --reload

View the app at: http://localhost:8000
"""
import requests
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"verification message": "This is respiracheck. No other endpoints currently available."}