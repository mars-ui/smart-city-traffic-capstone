import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_FILE = (
    SCRIPT_DIR
    / "results"
    / "supervised_metrics.json"
)

MODEL_DIR = SCRIPT_DIR / "models"

MLFLOW_DB = SCRIPT_DIR / "mlflow.db"

LOG_FILE = SCRIPT_DIR / "part3.log"

VERSION_FILE = (
    SCRIPT_DIR
    / "results"
    / "model_versions.csv"
)


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
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# =========================================================
# MLflow setup
# =========================================================
def configure_mlflow():
    """Configure local SQLite MLflow tracking."""

    tracking_uri = (
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "smart_city_traffic_models"
    )

    logger.info(
        "MLflow tracking configured"
    )


# =========================================================
# Log experiment
# =========================================================
def log_model_run(
    run_name,
    model_name,
    model_path,
    metrics,
    model_type,
    version,
):
    """Log one trained model and its evaluation metrics."""

    model = joblib.load(
        model_path
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "model_type",
            model_type
        )

        mlflow.log_param(
            "model_version",
            version
        )

        for metric_name, value in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(value)
            )

        mlflow.sklearn.log_model(
            sk_model=model,
         name="model",
         skops_trusted_types=[
        "sklearn.tree._tree.Tree"
        ]
        )

        mlflow.set_tag(
            "project",
            "Smart City Traffic Capstone"
        )

        mlflow.set_tag(
            "version",
            version
        )

        logger.info(
            "MLflow run completed: %s",
            run_name
        )


# =========================================================
# Main
# =========================================================
def main():

    try:

        configure_mlflow()

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

        classification = (
            results["classification"]
        )

        regression = (
            results["regression"]
        )

        versions = [
            {
                "version": "classification_v1",
                "model": "Logistic Regression",
                "task": "classification",
                **classification[
                    "Logistic Regression"
                ],
            },
            {
                "version": "classification_v2",
                "model": "Random Forest Classifier",
                "task": "classification",
                **classification[
                    "Random Forest Classifier"
                ],
            },
            {
                "version": "regression_v1",
                "model": "Linear Regression",
                "task": "regression",
                **regression[
                    "Linear Regression"
                ],
            },
            {
                "version": "regression_v2",
                "model": "Random Forest Regressor",
                "task": "regression",
                **regression[
                    "Random Forest Regressor"
                ],
            },
        ]

        pd.DataFrame(
            versions
        ).to_csv(
            VERSION_FILE,
            index=False
        )

        logger.info(
            "Model version comparison saved: results/model_versions.csv"
        )

        log_model_run(
            run_name="classification_v1_logistic",
            model_name="Logistic Regression",
            model_path=(
                MODEL_DIR
                / "logistic_classifier.joblib"
            ),
            metrics=classification[
                "Logistic Regression"
            ],
            model_type="classification",
            version="v1",
        )

        log_model_run(
            run_name="classification_v2_random_forest",
            model_name="Random Forest Classifier",
            model_path=(
                MODEL_DIR
                / "random_forest_classifier.joblib"
            ),
            metrics=classification[
                "Random Forest Classifier"
            ],
            model_type="classification",
            version="v2",
        )

        log_model_run(
            run_name="regression_v1_linear",
            model_name="Linear Regression",
            model_path=(
                MODEL_DIR
                / "linear_regression.joblib"
            ),
            metrics=regression[
                "Linear Regression"
            ],
            model_type="regression",
            version="v1",
        )

        log_model_run(
            run_name="regression_v2_random_forest",
            model_name="Random Forest Regressor",
            model_path=(
                MODEL_DIR
                / "random_forest_regressor.joblib"
            ),
            metrics=regression[
                "Random Forest Regressor"
            ],
            model_type="regression",
            version="v2",
        )

        print(
            "\n=== MLFLOW EXPERIMENT TRACKING COMPLETE ==="
        )

        print(
            "Experiment: smart_city_traffic_models"
        )

        print(
            "Tracked model versions: 4"
        )

        print(
            "Tracking database: mlflow.db"
        )

        print(
            "Model comparison: results/model_versions.csv"
        )

        logger.info(
            "MLflow workflow completed successfully"
        )

        return 0

    except Exception:

        logger.error(
            "MLflow workflow failed",
            exc_info=True
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )