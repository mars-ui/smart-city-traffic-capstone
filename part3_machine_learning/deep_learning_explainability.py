import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# Reproducibility
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)


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
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = SCRIPT_DIR / "part3.log"


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
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
# Feature selection
# =========================================================
def get_feature_columns(df):
    """Use the same traffic prediction features as Task 1."""

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

    return list(
        dict.fromkeys(
            base_features + weather_features
        )
    )


# =========================================================
# Neural network
# =========================================================
def train_neural_network(df, features):
    """Train a feed-forward neural network for traffic demand."""

    logger.info(
        "Starting neural network demand prediction"
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

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(X_train_scaled.shape[1],)
            ),
            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),
            tf.keras.layers.Dropout(
                0.20
            ),
            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),
            tf.keras.layers.Dense(
                1
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=25,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = (
        model.predict(
            X_test_scaled,
            verbose=0
        )
        .reshape(-1)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    logger.info(
        "Neural network completed: MAE=%.2f, R2=%.4f",
        mae,
        r2
    )

    model_path = (
        MODEL_DIR
        / "traffic_neural_network.keras"
    )

    model.save(model_path)

    joblib.dump(
        scaler,
        MODEL_DIR
        / "neural_network_scaler.joblib"
    )

    logger.info(
        "Neural network model saved: models/%s",
        model_path.name
    )

    # -----------------------------------------------------
    # Training history plot
    # -----------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Neural Network Training History"
    )
    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "Mean Squared Error"
    )
    plt.legend()
    plt.tight_layout()

    history_path = (
        FIGURES_DIR
        / "neural_network_training.png"
    )

    plt.savefig(
        history_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Training history figure saved: figures/%s",
        history_path.name
    )

    return {
        "mae": float(mae),
        "r2": float(r2),
        "epochs_completed": len(
            history.history["loss"]
        ),
    }


# =========================================================
# SHAP explainability
# =========================================================
def run_shap_explainability(df, features):
    """
    Explain the Random Forest traffic-volume model from Task 1.

    The neural network and Random Forest solve the same
    traffic-demand prediction problem. A tree-based model
    is used here because TreeSHAP provides efficient,
    interpretable feature-attribution values.
    """

    logger.info(
        "Starting SHAP explainability analysis"
    )

    rf_model_path = (
        MODEL_DIR
        / "random_forest_regressor.joblib"
    )

    rf_model = joblib.load(
        rf_model_path
    )

    X = df[features].copy()

    # Small reproducible sample keeps SHAP execution fast.
    sample_size = min(
        500,
        len(X)
    )

    X_sample = X.sample(
        n=sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        rf_model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    importance_file = (
        RESULTS_DIR
        / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    logger.info(
        "SHAP feature importance saved: results/%s",
        importance_file.name
    )

    # -----------------------------------------------------
    # SHAP summary plot
    # -----------------------------------------------------
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=15,
    )

    plt.tight_layout()

    shap_plot_path = (
        FIGURES_DIR
        / "shap_feature_importance.png"
    )

    plt.savefig(
        shap_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "SHAP figure saved: figures/%s",
        shap_plot_path.name
    )

    return (
        importance_df
        .head(10)
        .to_dict(
            orient="records"
        )
    )


# =========================================================
# Main
# =========================================================
def main():
    """Run deep learning and explainability."""

    try:

        df = pd.read_csv(
            INPUT_FILE
        )

        logger.info(
            "ML-ready dataset loaded for deep learning: "
            "%s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )

        features = get_feature_columns(
            df
        )

        neural_metrics = (
            train_neural_network(
                df,
                features,
            )
        )

        top_shap_features = (
            run_shap_explainability(
                df,
                features,
            )
        )

        results = {
            "neural_network": neural_metrics,
            "explainability_method": (
                "SHAP TreeExplainer applied to the "
                "Random Forest regression model trained "
                "on the same traffic-volume prediction task."
            ),
            "top_shap_features": top_shap_features,
        }

        results_file = (
            RESULTS_DIR
            / "deep_learning_results.json"
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
            "Deep-learning results saved: results/%s",
            results_file.name,
        )

        print(
            "\n=== NEURAL NETWORK RESULTS ==="
        )

        print(
            f"MAE: {neural_metrics['mae']:.2f}"
        )

        print(
            f"R2 : {neural_metrics['r2']:.4f}"
        )

        print(
            f"Epochs completed: "
            f"{neural_metrics['epochs_completed']}"
        )

        print(
            "\n=== TOP SHAP FEATURES ==="
        )

        for item in top_shap_features:
            print(
                f"{item['feature']}: "
                f"{item['mean_abs_shap']:.2f}"
            )

        logger.info(
            "Deep learning and explainability workflow "
            "completed successfully"
        )

        return 0

    except Exception:

        logger.error(
            "Deep learning and explainability workflow failed",
            exc_info=True,
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )