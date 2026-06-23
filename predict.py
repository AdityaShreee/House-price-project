"""
House Price Prediction - Inference Script
============================================
Loads the trained model + scaler (the same .pkl files you upload to S3)
and makes predictions on new house data.

Usage:
    python predict.py
"""

import joblib
import json
import numpy as np
import pandas as pd
import os

MODEL_DIR = "models"


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "house_price_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler, metadata


def predict(model, scaler, metadata, input_dict):
    """
    input_dict: dict with keys matching metadata['features'], e.g.
        {
            "MedInc": 5.2, "HouseAge": 25, "AveRooms": 6.1,
            "AveBedrms": 1.05, "Population": 1200, "AveOccup": 3.0,
            "Latitude": 34.1, "Longitude": -118.3
        }
    """
    features = metadata["features"]
    row = pd.DataFrame([[input_dict[f] for f in features]], columns=features)
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    return pred


if __name__ == "__main__":
    model, scaler, metadata = load_artifacts()
    print(f"Loaded model: {metadata['best_model']}")
    print(f"Expected features: {metadata['features']}\n")

    # Example house
    example = {
        "MedInc": 5.2,
        "HouseAge": 25,
        "AveRooms": 6.1,
        "AveBedrms": 1.05,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.1,
        "Longitude": -118.3,
    }

    predicted_value = predict(model, scaler, metadata, example)
    print(f"Input: {example}")
    print(f"Predicted median house value: ${predicted_value * 100000:,.2f}")
