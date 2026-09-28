import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    exist_ok=True
)

MONITORING_FILE = (
    RESULTS_DIR
    / "monitoring_report.json"
)

DRIFT_FILE = (
    RESULTS_DIR
    / "feature_drift.csv"
)

LOG_FILE = SCRIPT_DIR / "part3.log"


# =========================================================
# Monitoring thresholds
# =========================================================

# Raise an error-drift alert if the monitored MAE
# is more than 25% above the model's baseline MAE.
ERROR_DRIFT_THRESHOLD = 1.25

# Standardised mean difference above 0.50 is treated
# as meaningful feature distribution drift.
FEATURE_DRIFT_THRESHOLD = 0.50


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(
        formatter
    )

    console_handler = logging.StreamHandler()
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
    """Load dataset, trained model and baseline metrics."""

    df = pd.read_csv(
        DATA_FILE
    )

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
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

    model = joblib.load(
        MODEL_FILE
    )

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(
            file
        )

    features = metrics[
        "features"
    ]

    baseline_mae = metrics[
        "regression"
    ][
        "Random Forest Regressor"
    ][
        "mae"
    ]

    logger.info(
        "Monitoring assets loaded successfully"
    )

    return (
        df,
        model,
        features,
        float(baseline_mae)
    )


# =========================================================
# Split reference and monitoring windows
# =========================================================
def create_monitoring_windows(df):
    """
    Use the earlier 80 percent as the reference distribution
    and the latest 20 percent as the simulated live window.
    """

    split_index = int(
        len(df) * 0.80
    )

    reference_df = (
        df.iloc[
            :split_index
        ]
        .copy()
    )

    current_df = (
        df.iloc[
            split_index:
        ]
        .copy()
    )

    logger.info(
        "Reference monitoring window: %s records",
        len(reference_df)
    )

    logger.info(
        "Current monitoring window: %s records",
        len(current_df)
    )

    return (
        reference_df,
        current_df
    )


# =========================================================
# Prediction error drift
# =========================================================
def check_prediction_error(
    model,
    current_df,
    features,
    baseline_mae
):
    """Compare monitored MAE with the stored model baseline."""

    X_current = current_df[
        features
    ]

    y_current = current_df[
        "traffic_volume"
    ]

    predictions = model.predict(
        X_current
    )

    current_mae = mean_absolute_error(
        y_current,
        predictions
    )

    allowed_mae = (
        baseline_mae
        * ERROR_DRIFT_THRESHOLD
    )

    error_alert = bool(
        current_mae > allowed_mae
    )

    if error_alert:

        logger.warning(
            "Prediction error drift detected: "
            "current MAE %.2f exceeds threshold %.2f",
            current_mae,
            allowed_mae
        )

    else:

        logger.info(
            "Prediction error check passed: "
            "current MAE %.2f, threshold %.2f",
            current_mae,
            allowed_mae
        )

    return {
        "baseline_mae": float(
            baseline_mae
        ),
        "current_mae": float(
            current_mae
        ),
        "alert_threshold_mae": float(
            allowed_mae
        ),
        "error_drift_alert": error_alert
    }


# =========================================================
# Feature distribution drift
# =========================================================
def check_feature_drift(
    reference_df,
    current_df
):
    """
    Calculate standardised mean difference (SMD)
    between reference and monitored distributions.
    """

    monitored_features = [
        "hour",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_severity"
    ]

    rows = []

    for feature in monitored_features:

        reference_mean = (
            reference_df[
                feature
            ].mean()
        )

        current_mean = (
            current_df[
                feature
            ].mean()
        )

        reference_std = (
            reference_df[
                feature
            ].std()
        )

        if (
            pd.isna(reference_std)
            or reference_std == 0
        ):

            smd = 0.0

        else:

            smd = abs(
                current_mean
                - reference_mean
            ) / reference_std

        drift_detected = bool(
            smd
            > FEATURE_DRIFT_THRESHOLD
        )

        rows.append(
            {
                "feature": feature,
                "reference_mean": float(
                    reference_mean
                ),
                "current_mean": float(
                    current_mean
                ),
                "standardised_mean_difference":
                    float(smd),
                "drift_threshold":
                    FEATURE_DRIFT_THRESHOLD,
                "drift_detected":
                    drift_detected
            }
        )

        if drift_detected:

            logger.warning(
                "Feature drift detected for %s: "
                "SMD=%.3f",
                feature,
                smd
            )

        else:

            logger.info(
                "Feature drift check passed for %s: "
                "SMD=%.3f",
                feature,
                smd
            )

    drift_df = pd.DataFrame(
        rows
    )

    drift_df.to_csv(
        DRIFT_FILE,
        index=False
    )

    logger.info(
        "Feature drift report saved: results/%s",
        DRIFT_FILE.name
    )

    return drift_df


# =========================================================
# System alert status
# =========================================================
def determine_status(
    error_result,
    drift_df
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
            status
        )

    else:

        status = (
            "PASS / Normal"
        )

        logger.info(
            "Monitoring status: %s",
            status
        )

    return (
        status,
        feature_alert
    )


# =========================================================
# Main
# =========================================================
def main():

    try:

        (
            df,
            model,
            features,
            baseline_mae
        ) = load_assets()

        (
            reference_df,
            current_df
        ) = create_monitoring_windows(
            df
        )

        error_result = (
            check_prediction_error(
                model,
                current_df,
                features,
                baseline_mae
            )
        )

        drift_df = (
            check_feature_drift(
                reference_df,
                current_df
            )
        )

        (
            status,
            feature_alert
        ) = determine_status(
            error_result,
            drift_df
        )

        report = {
            "monitoring_method": {
                "reference_window":
                    "Earlier 80% of chronologically ordered observations",
                "current_window":
                    "Latest 20% of chronologically ordered observations",
                "prediction_error_rule":
                    "Alert when monitored MAE exceeds 125% of baseline MAE",
                "feature_drift_rule":
                    "Alert when standardised mean difference exceeds 0.50"
            },

            "prediction_error_monitoring":
                error_result,

            "feature_distribution_alert":
                feature_alert,

            "features_with_drift":
                drift_df.loc[
                    drift_df[
                        "drift_detected"
                    ],
                    "feature"
                ].tolist(),

            "overall_status":
                status
        }

        with open(
            MONITORING_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        logger.info(
            "Monitoring report saved: results/%s",
            MONITORING_FILE.name
        )

        print(
            "\n=== MODEL MONITORING REPORT ==="
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
            exc_info=True
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )