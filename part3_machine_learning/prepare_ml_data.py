import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "part2_python"
    / "output"
    / "cleaned_traffic.csv"
)

DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "ml_ready_traffic.csv"
LOG_FILE = SCRIPT_DIR / "part3.log"


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# =========================================================
# Weather definitions
# =========================================================

# Operational definition used for the proxy label.
# The capstone specifies SEVERE_WEATHER but does not prescribe
# the exact category list.
SEVERE_WEATHER = {
    "thunderstorm",
    "snow",
    "squall",
    "rain"
}

LOW_VISIBILITY_WEATHER = {
    "mist",
    "fog",
    "haze",
    "smoke"
}


# =========================================================
# Data loading
# =========================================================
def load_data():
    """Load the cleaned Part 2 dataset."""

    try:
        df = pd.read_csv(INPUT_FILE)

        logger.info(
            "Cleaned traffic dataset loaded: %s rows, %s columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except FileNotFoundError:
        logger.error(
            "Input file was not found: %s",
            INPUT_FILE.name,
            exc_info=True
        )
        return None

    except Exception:
        logger.error(
            "Unable to load the cleaned traffic dataset",
            exc_info=True
        )
        return None


# =========================================================
# Part 3 feature preparation
# =========================================================
def prepare_ml_data(df):
    """Create the ML-ready features and proxy accident-risk label."""

    df = df.copy()

    required_columns = [
        "date_time",
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "weather_description",
        "traffic_volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # Standardise categorical fields
    # -----------------------------------------------------
    df["weather_main"] = (
        df["weather_main"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["weather_description"] = (
        df["weather_description"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["holiday"] = (
        df["holiday"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Parse date/time
    # -----------------------------------------------------
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_dates = df["date_time"].isna().sum()

    if invalid_dates > 0:
        logger.warning(
            "%s records removed because date_time was invalid",
            invalid_dates
        )

        df = df.dropna(
            subset=["date_time"]
        )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------
    df["hour"] = df["date_time"].dt.hour

    df["day_of_week"] = (
        df["date_time"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Cyclical hour encoding
    # -----------------------------------------------------
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # -----------------------------------------------------
    # Cyclical day-of-week encoding
    # -----------------------------------------------------
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    logger.info(
        "Created time and cyclical features"
    )

    # -----------------------------------------------------
    # Holiday flag
    # -----------------------------------------------------
    df["holiday_flag"] = (
        df["holiday"]
        .fillna("none")
        .ne("none")
        .astype(int)
    )

    logger.info(
        "Created holiday flag"
    )

    # -----------------------------------------------------
    # Low-visibility indicator
    # -----------------------------------------------------
    description_low_visibility = (
        df["weather_description"]
        .fillna("")
        .str.contains(
            r"mist|fog|haze|smoke",
            regex=True
        )
    )

    df["is_low_visibility"] = (
        df["weather_main"].isin(
            LOW_VISIBILITY_WEATHER
        )
        | description_low_visibility
    ).astype(int)

    # -----------------------------------------------------
    # Severe-weather indicator
    # -----------------------------------------------------
    df["is_severe_weather"] = (
        df["weather_main"]
        .isin(SEVERE_WEATHER)
        .astype(int)
    )

    # Numeric weather severity indicator for later models
    weather_severity_map = {
        "clear": 0,
        "clouds": 0,
        "mist": 1,
        "haze": 1,
        "smoke": 1,
        "fog": 1,
        "drizzle": 1,
        "rain": 2,
        "snow": 2,
        "thunderstorm": 3,
        "squall": 3
    }

    df["weather_severity"] = (
        df["weather_main"]
        .map(weather_severity_map)
        .fillna(0)
        .astype(int)
    )

    logger.info(
        "Created weather severity and visibility indicators"
    )

    # -----------------------------------------------------
    # Quartile-based congestion category
    # Required by Part 3 instructions
    # -----------------------------------------------------
    q1, q2, q3 = (
        df["traffic_volume"]
        .quantile([0.25, 0.50, 0.75])
        .values
    )

    logger.debug(
        "Congestion quartiles: Q1=%.2f, Q2=%.2f, Q3=%.2f",
        q1,
        q2,
        q3
    )

    def bucket(value):
        if value <= q1:
            return "Low"
        elif value <= q2:
            return "Medium"
        elif value <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = (
        df["traffic_volume"]
        .apply(bucket)
    )

    logger.info(
        "Created quartile-based congestion category"
    )

    # -----------------------------------------------------
    # Proxy accident-risk target
    # -----------------------------------------------------
    high_congestion = (
        df["congestion_category"]
        .isin(["High", "Severe"])
    )

    risky_weather = (
        df["weather_main"]
        .isin(SEVERE_WEATHER)
        | (df["is_low_visibility"] == 1)
    )

    df["high_risk"] = (
        high_congestion
        & risky_weather
    ).astype(int)

    logger.info(
        "Created proxy high_risk label"
    )

    logger.info(
        "High-risk class distribution: %s",
        df["high_risk"]
        .value_counts()
        .to_dict()
    )

    # -----------------------------------------------------
    # Weather one-hot encoding
    # Keep original weather_main as well for later analysis
    # -----------------------------------------------------
    weather_encoded = pd.get_dummies(
        df["weather_main"],
        prefix="weather",
        dtype=int
    )

    df = pd.concat(
        [
            df,
            weather_encoded
        ],
        axis=1
    )

    logger.info(
        "Created weather one-hot encoded features"
    )

    logger.info(
        "Final ML-ready dataset: %s rows, %s columns",
        df.shape[0],
        df.shape[1]
    )

    return df


# =========================================================
# Main workflow
# =========================================================
def main():
    """Prepare the Part 3 ML dataset."""

    try:
        df = load_data()

        if df is None:
            logger.error(
                "Part 3 data preparation stopped because loading failed"
            )
            return 1

        ml_df = prepare_ml_data(df)

        ml_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        logger.info(
            "ML-ready dataset saved successfully: data/%s",
            OUTPUT_FILE.name
        )

        logger.info(
            "Part 3 data preparation completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "Part 3 data preparation failed",
            exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )