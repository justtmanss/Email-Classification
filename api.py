# api.py
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import tempfile
import shutil
import json

# Import our utility functions and model
from utils import PIIMasker, clean_email
from models import load_model, train_from_csv

class EmailRequest(BaseModel):
    email_body: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[Dict[str, Any]]
    masked_email: str
    category_of_the_email: str

class TrainRequest(BaseModel):
    model_type: str = "svm"  # Default to SVM

def setup_api_routes(app: FastAPI, model_path: str = "trained_model.pkl"):
    """Set up API routes for the application."""
    
    # Initialize PII masker
    masker = PIIMasker()
    
    # Load model if available
    model = load_model(model_path)
    
    @app.post("/classify-email", response_model=EmailResponse)
    async def classify_email(email_request: EmailRequest = Body(...)):
        """Classify an email and mask PII entities."""
        email_text = email_request.email_body
        
        # 1. Apply PII masking
        masked_email, entities = masker.mask_text(email_text)
        
        # 2. Clean the masked email for classification
        cleaned_email = clean_email(masked_email)
        
        # 3. Classify the email
        if model is None:
            # If model is not loaded, return a placeholder
            category = "Unknown (Model not loaded)"
        else:
            try:
                category = model.predict([cleaned_email])[0]
            except Exception as e:
                print(f"Classification error: {str(e)}")
                category = "Error in classification"
        
        # 4. Prepare the response
        response = {
            "input_email_body": email_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        return response
    
    @app.post("/train-model")
    async def train_model(
        file: UploadFile = File(...), 
        model_type: str = Form("svm")
    ):
        """Train a new model using the uploaded dataset."""
        try:
            # Save the uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            try:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            finally:
                temp_file.close()
            
            # Train the model
            metrics = train_from_csv(temp_file_path, model_type, model_path)
            
            # Clean up
            os.unlink(temp_file_path)
            
            # Reload the model
            nonlocal model
            model = load_model(model_path)
            
            return JSONResponse(content={
                "message": "Model trained and saved successfully",
                "metrics": {
                    "accuracy": float(metrics["accuracy"]),
                    "best_params": metrics["best_params"]
                }
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "error": f"Error training model: {str(e)}"
            })
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Email Classification API is running", "model_loaded": model is not None}

def create_app(model_path: str = "trained_model.pkl") -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Email Classification API",
        description="API for classifying emails and masking PII data",
        version="1.0.0"
    )
    
    setup_api_routes(app, model_path)
    return app