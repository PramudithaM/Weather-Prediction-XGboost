from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
from xgboost import XGBClassifier
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Rain Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at server startup, not per request
model = XGBClassifier()
model.load_model("app/model/rain_model.json")
model_columns = joblib.load("app/model/model_columns.pkl")

# Locations the model knows (Adelaide = baseline, has no dummy column)
KNOWN_LOCATIONS = {c[len("Location_"):] for c in model_columns if c.startswith("Location_")} | {"Adelaide"}

WindDir = Literal["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]

class WeatherInput(BaseModel):
    Location: str
    Month: int = Field(ge=1, le=12)
    MinTemp: float
    MaxTemp: float
    Rainfall: float = Field(ge=0)
    WindGustDir: WindDir
    WindGustSpeed: float = Field(ge=0)
    WindDir9am: WindDir
    WindDir3pm: WindDir
    WindSpeed9am: float = Field(ge=0)
    WindSpeed3pm: float = Field(ge=0)
    Humidity9am: float = Field(ge=0, le=100)
    Humidity3pm: float = Field(ge=0, le=100)
    Pressure9am: float
    Pressure3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: Literal["Yes", "No"]

@app.get("/")
def home():
    return {"message": "Rain Prediction API is running"}

@app.post("/predict")
def predict(data: WeatherInput):
    if data.Location not in KNOWN_LOCATIONS:
        raise HTTPException(status_code=422, detail=f"Unknown location. Known: {sorted(KNOWN_LOCATIONS)}")

    d = data.model_dump()

    # 1) Start all 107 columns at 0, in the exact order used during training
    row = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # 2) Fill in the numeric + binary columns
    for col in ["MinTemp","MaxTemp","Rainfall","WindGustSpeed","WindSpeed9am","WindSpeed3pm",
                "Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","Month"]:
        row.at[0, col] = d[col]
    row.at[0, "RainToday"] = 1 if d["RainToday"] == "Yes" else 0

    # 3) One-hot encoding: set the matching column to 1 if it exists (baseline category = all zeros)
    for prefix in ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]:
        col = f"{prefix}_{d[prefix]}"
        if col in row.columns:
            row.at[0, col] = 1

    # 4) Predict
    proba = float(model.predict_proba(row)[0][1])
    return {
        "rain_tomorrow": proba >= 0.5,
        "probability": round(proba, 4),
    }