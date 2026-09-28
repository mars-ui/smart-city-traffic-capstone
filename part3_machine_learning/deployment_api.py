import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    SCRIPT_DIR
    / "models"
    / "random_forest_regressor.joblib"
)

METRICS_FILE = (
    SCRIPT_DIR
    / "results"
    / "supervised_metrics.json"
)

LOG_FILE = SCRIPT_DIR / "part3.log"


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# =========================================================
# Load trained model and model feature list
# =========================================================
model = joblib.load(
    MODEL_FILE
)

with open(
    METRICS_FILE,
    "r",
    encoding="utf-8"
) as file:
    metrics_data = json.load(file)

FEATURES = metrics_data["features"]

logger.info(
    "Deployment model loaded successfully"
)


# =========================================================
# FastAPI application
# =========================================================
app = FastAPI(
    title="Smart City Traffic Prediction API",
    description=(
        "Deployment simulation for predicting traffic volume "
        "using the trained Random Forest regression model."
    ),
    version="1.0"
)


# =========================================================
# Input schema
# =========================================================
class TrafficInput(BaseModel):

    hour: int = Field(
        ge=0,
        le=23
    )

    day_of_week: int = Field(
        ge=0,
        le=6,
        description="Monday=0, Sunday=6"
    )

    holiday_flag: int = Field(
        default=0,
        ge=0,
        le=1
    )

    temp: float = Field(
        default=290.0,
        description="Temperature in Kelvin"
    )

    rain_1h: float = Field(
        default=0.0,
        ge=0
    )

    snow_1h: float = Field(
        default=0.0,
        ge=0
    )

    clouds_all: float = Field(
        default=20.0,
        ge=0,
        le=100
    )

    weather_main: str = Field(
        default="clear"
    )


# =========================================================
# Feature engineering for API request
# =========================================================
def build_feature_row(data: TrafficInput):

    weather = (
        data.weather_main
        .strip()
        .lower()
    )

    is_weekend = int(
        data.day_of_week >= 5
    )

    low_visibility_conditions = {
        "mist",
        "fog",
        "haze",
        "smoke"
    }

    severe_weather_conditions = {
        "rain",
        "snow",
        "thunderstorm",
        "squall"
    }

    severity_map = {
        "clear": 0,
        "clouds": 0,
        "mist": 1,
        "fog": 1,
        "haze": 1,
        "smoke": 1,
        "drizzle": 1,
        "rain": 2,
        "snow": 2,
        "thunderstorm": 3,
        "squall": 3
    }

    values = {
        feature: 0.0
        for feature in FEATURES
    }

    base_values = {
        "hour": data.hour,
        "day_of_week": data.day_of_week,
        "is_weekend": is_weekend,

        "hour_sin": np.sin(
            2 * np.pi * data.hour / 24
        ),

        "hour_cos": np.cos(
            2 * np.pi * data.hour / 24
        ),

        "day_sin": np.sin(
            2 * np.pi * data.day_of_week / 7
        ),

        "day_cos": np.cos(
            2 * np.pi * data.day_of_week / 7
        ),

        "holiday_flag": data.holiday_flag,

        "temp": data.temp,
        "rain_1h": data.rain_1h,
        "snow_1h": data.snow_1h,
        "clouds_all": data.clouds_all,

        "is_low_visibility": int(
            weather in low_visibility_conditions
        ),

        "is_severe_weather": int(
            weather in severe_weather_conditions
        ),

        "weather_severity": (
            severity_map.get(
                weather,
                0
            )
        )
    }

    for feature, value in base_values.items():

        if feature in values:
            values[feature] = value

    weather_feature = (
        f"weather_{weather}"
    )

    if weather_feature in values:
        values[weather_feature] = 1

    return pd.DataFrame(
        [
            [
                values[feature]
                for feature in FEATURES
            ]
        ],
        columns=FEATURES
    )


# =========================================================
# API endpoints
# =========================================================
@app.get("/")
def root():

    return {
        "service": "Smart City Traffic Prediction API",
        "status": "PASS",
        "model": "Random Forest Regressor"
    }


@app.post("/predict")
def predict_traffic(
    request: TrafficInput
):

    logger.info(
        "Prediction request received: hour=%s, "
        "day_of_week=%s, weather=%s",
        request.hour,
        request.day_of_week,
        request.weather_main
    )

    model_input = build_feature_row(
        request
    )

    predicted_traffic = float(
        model.predict(
            model_input
        )[0]
    )

    response = {
        "predicted_traffic_volume": round(
            predicted_traffic,
            2
        ),
        "model": "Random Forest Regressor",
        "status": "success"
    }

    logger.info(
        "Traffic prediction generated successfully"
    )

    return response


# =========================================================
# Local self-test
# =========================================================
def run_self_test():

    sample = TrafficInput(
        hour=10,
        day_of_week=1,
        holiday_flag=0,
        temp=290.0,
        rain_1h=0.0,
        snow_1h=0.0,
        clouds_all=20.0,
        weather_main="clear"
    )

    result = predict_traffic(
        sample
    )

    print(
        "\n=== DEPLOYMENT API SELF-TEST ==="
    )

    print(
        f"Input: Tuesday 10:00, clear weather"
    )

    print(
        "Predicted traffic volume: "
        f"{result['predicted_traffic_volume']:.2f}"
    )

    print(
        "API status: "
        f"{result['status']}"
    )


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run API prediction self-test"
    )

    args = parser.parse_args()

    if args.test:

        run_self_test()

    else:

        logger.info(
            "Starting FastAPI deployment simulation"
        )

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000
        )


if __name__ == "__main__":
    main()