from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from transformers import pipeline
from typing import List, Union
from pydantic import BaseModel
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv("key.env")
# Load model once at startup
pipe = pipeline("text-classification", model="Dc-4nderson/soulprint-classification")

# Simple API key authentication
# Ensure you set this in your environment variables or .env file 
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise RuntimeError("API_KEY environment variable is not set. Please set it in your key.env file.")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

app = FastAPI()

class TextRequest(BaseModel):
    age: int
    statement: Union[str, List[str]]

@app.post("/classify")
async def classify(
    request: TextRequest,
    api_key: str = Depends(verify_api_key)
):
    # Combine age and statement(s)
    if isinstance(request.statement, str):
        input_text = f"Age: {request.age}. Statement: {request.statement}"
        result = pipe(input_text)
    elif isinstance(request.statement, list):
        input_texts = [f"Age: {request.age}. Statement: {s}" for s in request.statement]
        result = pipe(input_texts)
    else:
        raise HTTPException(status_code=400, detail="Invalid input type")
    return {"result": result}