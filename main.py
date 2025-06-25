from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("xgb_volatility_model.pkl")

# Create API
app = FastAPI()

# Input schema
class VolatilityInput(BaseModel):
    lag_return_1: float
    lag_return_3: float
    lag_sentiment_1: float

# Endpoint
@app.post("/predict")
def predict(input: VolatilityInput):
    data = pd.DataFrame([input.dict()])
    prediction = model.predict(data)[0]
    return {"predicted_volatility_7d": float(prediction)}