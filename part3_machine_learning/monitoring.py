import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    SCRIPT_DIR
    / "data"
    / "ml_ready_traffic.csv"
)

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

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MONITORING_FILE = (
    RESULTS_DIR
    / "monitoring_report.json"
)

DRIFT_FILE = (
    RESULTS_DIR
    / "feature_drift.csv"
)

LOG_FILE = (
    SCRIPT_DIR
    / "part3.log"
)


# =========================================================
# Monitoring thresholds
# =========================================================

# Raise an error-drift alert if current MAE exceeds
# the clean baseline MAE by more than 25%.
ERROR_DRIFT_THRESHOLD = 1.25

# Standardised mean difference above 0.50 is treated
# as meaningful feature-distribution drift.
FEATURE_DRIFT_THRESHOLD = 0.50


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger(__name__)


def configure_logging():
    """Configure console and file logging."""

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            formatter
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

        logger.addHandler(
            console_handler
        )


# =========================================================
# Load monitoring assets
# =========================================================
def load_assets():
    """
    Load the ML-ready dataset, fitted production model
    architecture and feature list.
    """

    df = pd.read_csv(
        DATA_FILE
    )

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce",
    )

    invalid_dates = (
        df["date_time"]
        .isna()
        .sum()
    )

    if invalid_dates > 0:
        logger.warning(
            "%s records removed because date_time was invalid",
            invalid_dates,
        )

    df = (
        df
        .dropna(
            subset=["date_time"]
        )
        .sort_values(
            "date_time"
        )
        .reset_index(
            drop=True
        )
    )

    production_model = joblib.load(
        MODEL_FILE
    )

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(
            file
        )

    features = metrics[
        "features"
    ]

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Monitoring dataset is missing features: {missing_features}"
        )

    logger.info(
        "Monitoring assets loaded successfully: "
        "%s rows, %s model features",
        len(df),
        len(features),
    )

    return (
        df,
        production_model,
        features,
    )


# =========================================================
# Create chronological monitoring windows
# =========================================================
def create_monitoring_windows(df):
    """
    Split chronologically into:

    - earliest 70%: training window
    - next 10%: baseline/reference window
    - latest 20%: simulated live/current window
    """

    total_records = len(
        df
    )

    train_end = int(
        total_records * 0.70
    )

    baseline_end = int(
        total_records * 0.80
    )

    if (
        train_end <= 0
        or baseline_end <= train_end
        or baseline_end >= total_records
    ):
        raise ValueError(
            "Dataset is too small to create 70/10/20 monitoring windows"
        )

    train_df = (
        df.iloc[
            :train_end
        ]
        .copy()
    )

    baseline_df = (
        df.iloc[
            train_end:baseline_end
        ]
        .copy()
    )

    current_df = (
        df.iloc[
            baseline_end:
        ]
        .copy()
    )

    logger.info(
        "Monitoring training window: %s records",
        len(train_df),
    )

    logger.info(
        "Baseline/reference window: %s records",
        len(baseline_df),
    )

    logger.info(
        "Current/live simulation window: %s records",
        len(current_df),
    )

    logger.debug(
        "Training date range: %s to %s",
        train_df["date_time"].min(),
        train_df["date_time"].max(),
    )

    logger.debug(
        "Baseline date range: %s to %s",
        baseline_df["date_time"].min(),
        baseline_df["date_time"].max(),
    )

    logger.debug(
        "Current date range: %s to %s",
        current_df["date_time"].min(),
        current_df["date_time"].max(),
    )

    return (
        train_df,
        baseline_df,
        current_df,
    )


# =========================================================
# Train monitoring simulation model
# =========================================================
def train_monitoring_model(
    production_model,
    train_df,
    features,
):
    """
    Clone the Random Forest configuration and train only
    on the earliest chronological 70%.

    The production model file is not overwritten.
    """

    monitoring_model = clone(
        production_model
    )

    X_train = train_df[
        features
    ]

    y_train = train_df[
        "traffic_volume"
    ]

    monitoring_model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Monitoring simulation model trained on chronological "
        "training window"
    )

    return monitoring_model


# =========================================================
# Prediction error monitoring
# =========================================================
def check_prediction_error(
    model,
    baseline_df,
    current_df,
    features,
):
    """
    Compare MAE on an unseen baseline window against MAE on
    the later simulated-live window.
    """

    X_baseline = baseline_df[
        features
    ]

    y_baseline = baseline_df[
        "traffic_volume"
    ]

    baseline_predictions = model.predict(
        X_baseline
    )

    baseline_mae = mean_absolute_error(
        y_baseline,
        baseline_predictions,
    )

    X_current = current_df[
        features
    ]

    y_current = current_df[
        "traffic_volume"
    ]

    current_predictions = model.predict(
        X_current
    )

    current_mae = mean_absolute_error(
        y_current,
        current_predictions,
    )

    allowed_mae = (
        baseline_mae
        * ERROR_DRIFT_THRESHOLD
    )

    error_alert = bool(
        current_mae
        > allowed_mae
    )

    percentage_change = (
        (
            current_mae
            - baseline_mae
        )
        / baseline_mae
        * 100
        if baseline_mae != 0
        else 0.0
    )

    if error_alert:
        logger.warning(
            "Prediction error drift detected: "
            "baseline MAE %.2f, current MAE %.2f, "
            "threshold %.2f",
            baseline_mae,
            current_mae,
            allowed_mae,
        )
    else:
        logger.info(
            "Prediction error check passed: "
            "baseline MAE %.2f, current MAE %.2f, "
            "threshold %.2f",
            baseline_mae,
            current_mae,
            allowed_mae,
        )

    return {
        "baseline_mae": float(
            baseline_mae
        ),
        "current_mae": float(
            current_mae
        ),
        "mae_percentage_change": float(
            percentage_change
        ),
        "alert_threshold_mae": float(
            allowed_mae
        ),
        "error_drift_alert": error_alert,
    }


# =========================================================
# Feature distribution drift
# =========================================================
def check_feature_drift(
    baseline_df,
    current_df,
):
    """
    Calculate standardised mean difference (SMD)
    between baseline and current distributions.
    """

    monitored_features = [
        "hour",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_severity",
    ]

    rows = []

    for feature in monitored_features:
        baseline_mean = (
            baseline_df[
                feature
            ]
            .mean()
        )

        current_mean = (
            current_df[
                feature
            ]
            .mean()
        )

        baseline_std = (
            baseline_df[
                feature
            ]
            .std()
        )

        if (
            pd.isna(
                baseline_std
            )
            or baseline_std == 0
        ):
            smd = 0.0

        else:
            smd = abs(
                current_mean
                - baseline_mean
            ) / baseline_std

        drift_detected = bool(
            smd
            > FEATURE_DRIFT_THRESHOLD
        )

        rows.append(
            {
                "feature": feature,
                "baseline_mean": float(
                    baseline_mean
                ),
                "current_mean": float(
                    current_mean
                ),
                "standardised_mean_difference": float(
                    smd
                ),
                "drift_threshold": FEATURE_DRIFT_THRESHOLD,
                "drift_detected": drift_detected,
            }
        )

        if drift_detected:
            logger.warning(
                "Feature drift detected for %s: SMD=%.3f",
                feature,
                smd,
            )

        else:
            logger.info(
                "Feature drift check passed for %s: SMD=%.3f",
                feature,
                smd,
            )

    drift_df = pd.DataFrame(
        rows
    )

    drift_df.to_csv(
        DRIFT_FILE,
        index=False,
    )

    logger.info(
        "Feature drift report saved: results/%s",
        DRIFT_FILE.name,
    )

    return drift_df


# =========================================================
# System alert status
# =========================================================
def determine_status(
    error_result,
    drift_df,
):
    """Return overall PASS or ALERT monitoring status."""

    error_alert = (
        error_result[
            "error_drift_alert"
        ]
    )

    feature_alert = bool(
        drift_df[
            "drift_detected"
        ].any()
    )

    if (
        error_alert
        or feature_alert
    ):
        status = (
            "ALERT / Requires investigation"
        )

        logger.warning(
            "Monitoring status: %s",
            status,
        )

    else:
        status = (
            "PASS / Normal"
        )

        logger.info(
            "Monitoring status: %s",
            status,
        )

    return (
        status,
        feature_alert,
    )


# =========================================================
# Main
# =========================================================
def main():
    """Run chronological model-monitoring simulation."""

    configure_logging()

    try:
        (
            df,
            production_model,
            features,
        ) = load_assets()

        (
            train_df,
            baseline_df,
            current_df,
        ) = create_monitoring_windows(
            df
        )

        monitoring_model = train_monitoring_model(
            production_model,
            train_df,
            features,
        )

        error_result = check_prediction_error(
            monitoring_model,
            baseline_df,
            current_df,
            features,
        )

        drift_df = check_feature_drift(
            baseline_df,
            current_df,
        )

        (
            status,
            feature_alert,
        ) = determine_status(
            error_result,
            drift_df,
        )

        report = {
            "monitoring_method": {
                "chronological_split": (
                    "70% training / 10% baseline / 20% current"
                ),
                "training_window": (
                    "Earliest 70% of chronologically ordered observations"
                ),
                "baseline_window": (
                    "Next 10% of chronologically ordered observations"
                ),
                "current_window": (
                    "Latest 20% of chronologically ordered observations"
                ),
                "prediction_error_rule": (
                    "Alert when current MAE exceeds "
                    "125% of unseen baseline MAE"
                ),
                "feature_drift_rule": (
                    "Alert when standardised mean difference "
                    "exceeds 0.50"
                ),
                "monitoring_model_note": (
                    "A fresh clone of the Random Forest configuration "
                    "is trained only on the earliest 70% so baseline and "
                    "current monitoring windows remain out-of-sample."
                ),
            },

            "window_sizes": {
                "training_records": len(
                    train_df
                ),
                "baseline_records": len(
                    baseline_df
                ),
                "current_records": len(
                    current_df
                ),
            },

            "window_dates": {
                "training_start": str(
                    train_df["date_time"].min()
                ),
                "training_end": str(
                    train_df["date_time"].max()
                ),
                "baseline_start": str(
                    baseline_df["date_time"].min()
                ),
                "baseline_end": str(
                    baseline_df["date_time"].max()
                ),
                "current_start": str(
                    current_df["date_time"].min()
                ),
                "current_end": str(
                    current_df["date_time"].max()
                ),
            },

            "prediction_error_monitoring": (
                error_result
            ),

            "feature_distribution_alert": (
                feature_alert
            ),

            "features_with_drift": (
                drift_df.loc[
                    drift_df[
                        "drift_detected"
                    ],
                    "feature",
                ]
                .tolist()
            ),

            "overall_status": (
                status
            ),
        }

        with open(
            MONITORING_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
            )

        logger.info(
            "Monitoring report saved: results/%s",
            MONITORING_FILE.name,
        )

        # -----------------------------------------------------
        # User-facing monitoring summary
        # -----------------------------------------------------
        print(
            "\n=== MODEL MONITORING REPORT ==="
        )

        print(
            "Method: chronological 70% train / "
            "10% baseline / 20% current"
        )

        print(
            "Baseline MAE: "
            f"{error_result['baseline_mae']:.2f}"
        )

        print(
            "Current MAE : "
            f"{error_result['current_mae']:.2f}"
        )

        print(
            "MAE change  : "
            f"{error_result['mae_percentage_change']:.2f}%"
        )

        print(
            "Error threshold: "
            f"{error_result['alert_threshold_mae']:.2f}"
        )

        print(
            "Features with detected drift: "
            f"{report['features_with_drift']}"
        )

        print(
            "\nSYSTEM STATUS: "
            f"{status}"
        )

        logger.info(
            "Model monitoring workflow completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "Model monitoring workflow failed",
            exc_info=True,
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )