"""
FastAPI REST API - Health MLOps Project
Real-time health risk prediction API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Health MLOps API",
    description="Real-time health risk prediction using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {}
scaler = None


@app.on_event("startup")
async def load_models():
    """Load trained models on API startup"""
    global models, scaler

    try:
        # Load Random Forest
        rf_path = "models/baseline/rf_regressor.pkl"
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                models["random_forest"] = pickle.load(f)
            print(f"✓ Loaded Random Forest model")

        # Load XGBoost
        xgb_path = "models/baseline/xgb_regressor.pkl"
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                models["xgboost"] = pickle.load(f)
            print(f"✓ Loaded XGBoost model")

        # Load feature names and scaler
        data_path = "data/processed/centralized_data.pkl"
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                if 'scaler' in data:
                    scaler = data['scaler']
                print(f"✓ Loaded data preprocessing pipeline")

        print(f"✅ API ready with {len(models)} models loaded")

    except Exception as e:
        print(f"⚠️ Error loading models: {e}")


# Request/Response models
class HealthData(BaseModel):
    """Health metrics input"""
    avg_heart_rate: float = 75.0
    avg_spo2: float = 97.0
    avg_body_temp: float = 36.8
    avg_steps: int = 8000
    avg_pm25: float = 35.0
    avg_pm10: float = 50.0
    avg_no2: float = 25.0
    avg_aqi: float = 75.0
    avg_temperature: float = 22.0
    avg_humidity: float = 60.0
    avg_pressure: float = 1013.0

    class Config:
        schema_extra = {
            "example": {
                "avg_heart_rate": 85.0,
                "avg_spo2": 96.0,
                "avg_body_temp": 37.2,
                "avg_steps": 6000,
                "avg_pm25": 55.0,
                "avg_pm10": 75.0,
                "avg_no2": 35.0,
                "avg_aqi": 95.0,
                "avg_temperature": 28.0,
                "avg_humidity": 70.0,
                "avg_pressure": 1010.0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output"""
    model: str
    predicted_sick_percentage: float
    risk_level: str
    confidence: str
    recommendations: List[str]


# API Endpoints
@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "message": "Health MLOps API is running",
        "version": "1.0.0",
        "models_loaded": len(models)
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "ok",
        "models": {
            "random_forest": "random_forest" in models,
            "xgboost": "xgboost" in models
        },
        "scaler_loaded": scaler is not None
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "default_model": "random_forest" if "random_forest" in models else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HealthData, model_name: Optional[str] = "random_forest"):
    """
    Predict health risk based on input metrics

    - **data**: Health metrics (heart rate, SpO2, air quality, etc.)
    - **model_name**: Model to use (random_forest, xgboost)
    """

    # Check if model exists
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )

    try:
        # Prepare input features
        features = np.array([[
            data.avg_heart_rate,
            data.avg_spo2,
            data.avg_body_temp,
            data.avg_steps,
            data.avg_pm25,
            data.avg_pm10,
            data.avg_no2,
            data.avg_aqi,
            data.avg_temperature,
            data.avg_humidity,
            data.avg_pressure
        ]])

        # Make prediction
        model = models[model_name]
        prediction = model.predict(features)[0]

        # Clip to valid range
        prediction = np.clip(prediction, 0, 100)

        # Determine risk level
        if prediction < 5:
            risk_level = "Low Risk"
            confidence = "High"
            recommendations = [
                "Continue healthy habits",
                "Maintain regular exercise",
                "Monitor air quality periodically"
            ]
        elif prediction < 15:
            risk_level = "Moderate Risk"
            confidence = "Medium"
            recommendations = [
                "Monitor symptoms closely",
                "Reduce outdoor activities on high pollution days",
                "Stay hydrated",
                "Consider wearing a mask outdoors"
            ]
        else:
            risk_level = "High Risk"
            confidence = "High"
            recommendations = [
                "⚠️ Consult healthcare provider",
                "Avoid outdoor activities",
                "Use air purifier indoors",
                "Monitor vital signs regularly",
                "Keep emergency contacts ready"
            ]

        return PredictionResponse(
            model=model_name,
            predicted_sick_percentage=round(prediction, 2),
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(data_list: List[HealthData], model_name: str = "random_forest"):
    """Batch prediction for multiple samples"""

    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    predictions = []
    for data in data_list:
        result = await predict(data, model_name)
        predictions.append(result)

    return {
        "model": model_name,
        "count": len(predictions),
        "predictions": predictions
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
