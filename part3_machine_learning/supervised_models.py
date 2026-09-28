import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    SCRIPT_DIR
    / "data"
    / "ml_ready_traffic.csv"
)

MODEL_DIR = SCRIPT_DIR / "models"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
# Features
# =========================================================
def get_feature_columns(df):
    """
    Use a common feature set for classification and regression.

    traffic_volume and congestion_category are deliberately
    excluded to prevent target leakage.
    """

    base_features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "holiday_flag",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "is_low_visibility",
        "is_severe_weather",
        "weather_severity",
    ]

    weather_features = [
        column
        for column in df.columns
        if column.startswith("weather_")
        and column not in {
            "weather_main",
            "weather_description",
            "weather_severity",
        }
    ]

    feature_columns = (
        base_features
        + weather_features
    )

    feature_columns = list(
        dict.fromkeys(feature_columns)
    )

    logger.info(
        "Common model feature set contains %s features",
        len(feature_columns)
    )

    return feature_columns


# =========================================================
# Classification
# =========================================================
def run_classification(df, features):
    """Train and evaluate two classification algorithms."""

    logger.info(
        "Starting classification modelling"
    )

    X = df[features].copy()
    y = df["high_risk"].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -----------------------------------------------------
    # Model 1: Logistic Regression
    # -----------------------------------------------------
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    logistic_model.fit(
        X_train_scaled,
        y_train,
    )

    logistic_pred = logistic_model.predict(
        X_test_scaled
    )

    logistic_prob = logistic_model.predict_proba(
        X_test_scaled
    )[:, 1]

    logistic_metrics = {
        "accuracy": accuracy_score(
            y_test,
            logistic_pred,
        ),
        "precision": precision_score(
            y_test,
            logistic_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            logistic_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            logistic_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            logistic_prob,
        ),
    }

    logger.info(
        "Logistic Regression classification completed"
    )

    # -----------------------------------------------------
    # Model 2: Random Forest Classifier
    # -----------------------------------------------------
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf_classifier.fit(
        X_train,
        y_train,
    )

    rf_pred = rf_classifier.predict(
        X_test
    )

    rf_prob = rf_classifier.predict_proba(
        X_test
    )[:, 1]

    rf_metrics = {
        "accuracy": accuracy_score(
            y_test,
            rf_pred,
        ),
        "precision": precision_score(
            y_test,
            rf_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            rf_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            rf_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            rf_prob,
        ),
    }

    logger.info(
        "Random Forest classification completed"
    )

    # Save models
    joblib.dump(
        logistic_model,
        MODEL_DIR / "logistic_classifier.joblib",
    )

    joblib.dump(
        rf_classifier,
        MODEL_DIR / "random_forest_classifier.joblib",
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "classification_scaler.joblib",
    )

    logger.info(
        "Classification models saved successfully"
    )

    return {
        "Logistic Regression": logistic_metrics,
        "Random Forest Classifier": rf_metrics,
    }


# =========================================================
# Regression
# =========================================================
def run_regression(df, features):
    """Train and evaluate two regression algorithms."""

    logger.info(
        "Starting regression modelling"
    )

    X = df[features].copy()
    y = df["traffic_volume"].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -----------------------------------------------------
    # Model 1: Linear Regression
    # -----------------------------------------------------
    linear_model = LinearRegression()

    linear_model.fit(
        X_train_scaled,
        y_train,
    )

    linear_pred = linear_model.predict(
        X_test_scaled
    )

    linear_metrics = {
        "mae": mean_absolute_error(
            y_test,
            linear_pred,
        ),
        "r2": r2_score(
            y_test,
            linear_pred,
        ),
    }

    logger.info(
        "Linear Regression completed"
    )

    # -----------------------------------------------------
    # Model 2: Random Forest Regressor
    # -----------------------------------------------------
    rf_regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
    )

    rf_regressor.fit(
        X_train,
        y_train,
    )

    rf_regression_pred = rf_regressor.predict(
        X_test
    )

    rf_regression_metrics = {
        "mae": mean_absolute_error(
            y_test,
            rf_regression_pred,
        ),
        "r2": r2_score(
            y_test,
            rf_regression_pred,
        ),
    }

    logger.info(
        "Random Forest regression completed"
    )

    # Save models
    joblib.dump(
        linear_model,
        MODEL_DIR / "linear_regression.joblib",
    )

    joblib.dump(
        rf_regressor,
        MODEL_DIR / "random_forest_regressor.joblib",
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "regression_scaler.joblib",
    )

    logger.info(
        "Regression models saved successfully"
    )

    return {
        "Linear Regression": linear_metrics,
        "Random Forest Regressor": rf_regression_metrics,
    }


# =========================================================
# Main
# =========================================================
def main():
    """Run supervised classification and regression."""

    try:
        df = pd.read_csv(
            INPUT_FILE
        )

        logger.info(
            "ML-ready dataset loaded: %s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )

        feature_columns = get_feature_columns(
            df
        )

        classification_results = (
            run_classification(
                df,
                feature_columns,
            )
        )

        regression_results = (
            run_regression(
                df,
                feature_columns,
            )
        )

        results = {
            "classification": classification_results,
            "regression": regression_results,
            "features": feature_columns,
        }

        results_file = (
            RESULTS_DIR
            / "supervised_metrics.json"
        )

        with open(
            results_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=4,
            )

        logger.info(
            "Supervised model metrics saved: results/%s",
            results_file.name,
        )

        # User-facing summary
        print(
            "\n=== CLASSIFICATION RESULTS ==="
        )

        for model, metrics in classification_results.items():
            print(f"\n{model}")
            print(
                f"Accuracy : {metrics['accuracy']:.4f}"
            )
            print(
                f"Precision: {metrics['precision']:.4f}"
            )
            print(
                f"Recall   : {metrics['recall']:.4f}"
            )
            print(
                f"F1-score : {metrics['f1']:.4f}"
            )
            print(
                f"ROC AUC  : {metrics['roc_auc']:.4f}"
            )

        print(
            "\n=== REGRESSION RESULTS ==="
        )

        for model, metrics in regression_results.items():
            print(f"\n{model}")
            print(
                f"MAE: {metrics['mae']:.2f}"
            )
            print(
                f"R2 : {metrics['r2']:.4f}"
            )

        logger.info(
            "Supervised ML workflow completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "Supervised ML workflow failed",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )