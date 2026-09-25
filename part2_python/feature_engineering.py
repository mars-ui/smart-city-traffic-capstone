import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

log_file = Path(__file__).resolve().parent / "pipeline.log"

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def create_features(df):
    """Create ML-ready traffic and weather features."""

    df = df.copy()

    logger.info(
        "Dataset shape before feature engineering: %s rows, %s columns",
        df.shape[0],
        df.shape[1]
    )

    # -------------------------------------------------
    # 1. Time features
    # -------------------------------------------------
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek

    # Monday = 0 ... Sunday = 6
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    logger.info(
        "Created hour, day_of_week and is_weekend features"
    )

    # -------------------------------------------------
    # 2. Cyclical hour encoding
    # -------------------------------------------------
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    logger.info(
        "Created cyclical hour_sin and hour_cos features"
    )

    # -------------------------------------------------
    # 3. Derived weather indicator
    # -------------------------------------------------
    adverse_weather = [
        "rain",
        "snow",
        "thunderstorm",
        "drizzle",
        "mist",
        "fog",
        "squall"
    ]

    df["adverse_weather"] = (
        df["weather_main"]
        .isin(adverse_weather)
        .astype(int)
    )

    logger.info(
        "Created adverse_weather indicator"
    )

    # -------------------------------------------------
    # 4. Encode weather categories
    # -------------------------------------------------
    df = pd.get_dummies(
        df,
        columns=["weather_main"],
        prefix="weather",
        dtype=int
    )

    logger.info(
        "Encoded weather_main using one-hot encoding"
    )

    # -------------------------------------------------
    # 5. Scale continuous numerical variables
    # -------------------------------------------------
    scaler = StandardScaler()

    columns_to_scale = [
        "temp",
        "rain_1h",
        "clouds_all"
    ]

    scaled_values = scaler.fit_transform(
        df[columns_to_scale]
    )

    df["temp_scaled"] = scaled_values[:, 0]
    df["rain_1h_scaled"] = scaled_values[:, 1]
    df["clouds_all_scaled"] = scaled_values[:, 2]

    logger.info(
        "Scaled temp, rain_1h and clouds_all"
    )

    # -------------------------------------------------
    # 6. Data-driven congestion category
    # -------------------------------------------------
    lower_threshold = df["traffic_volume"].quantile(0.33)
    upper_threshold = df["traffic_volume"].quantile(0.66)

    logger.debug(
        "Congestion thresholds calculated: "
        "33rd percentile = %.2f, 66th percentile = %.2f",
        lower_threshold,
        upper_threshold
    )

    conditions = [
        df["traffic_volume"] <= lower_threshold,
        df["traffic_volume"] <= upper_threshold
    ]

    categories = [
        "low",
        "medium"
    ]

    df["congestion_category"] = np.select(
        conditions,
        categories,
        default="high"
    )

    logger.info(
        "Created data-driven congestion_category"
    )

    logger.info(
        "Dataset shape after feature engineering: %s rows, %s columns",
        df.shape[0],
        df.shape[1]
    )

    return df


def main():

    project_root = Path(__file__).resolve().parent.parent

    input_file = (
        project_root
        / "part2_python"
        / "output"
        / "cleaned_traffic.csv"
    )

    output_file = (
        project_root
        / "part2_python"
        / "output"
        / "feature_engineered_traffic.csv"
    )

    try:
        df = pd.read_csv(input_file)

        logger.info(
            "Cleaned dataset loaded for feature engineering"
        )

        feature_df = create_features(df)

        feature_df.to_csv(
            output_file,
            index=False
        )

        logger.info(
            "Feature-engineered dataset saved to: %s",
            output_file
        )

        logger.info(
            "Feature engineering completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "Feature engineering failed",
            exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())