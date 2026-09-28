import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_FILE = (
    RESULTS_DIR
    / "supervised_metrics.json"
)

MODEL_DIR = SCRIPT_DIR / "models"

MLFLOW_DB = SCRIPT_DIR / "mlflow.db"

LOG_FILE = SCRIPT_DIR / "part3.log"

VERSION_FILE = (
    RESULTS_DIR
    / "model_versions.csv"
)

MLFLOW_RUNS_FILE = (
    RESULTS_DIR
    / "mlflow_runs.csv"
)

EXPERIMENT_NAME = (
    "smart_city_traffic_models"
)


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger(__name__)


def configure_logging():
    """Configure console and file logging for this entry-point script."""

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
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
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

    experiment = mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    logger.info(
        "MLflow tracking configured for experiment: %s",
        EXPERIMENT_NAME,
    )

    return experiment


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
    """Log one trained model and return sanitized run evidence."""

    model = joblib.load(
        model_path
    )

    run_id = None

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        run_id = run.info.run_id

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "model_type",
            model_type,
        )

        mlflow.log_param(
            "model_version",
            version,
        )

        for metric_name, value in metrics.items():
            mlflow.log_metric(
                metric_name,
                float(value),
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        mlflow.set_tag(
            "project",
            "Smart City Traffic Capstone",
        )

        mlflow.set_tag(
            "version",
            version,
        )

        logger.info(
            "MLflow run completed: %s",
            run_name,
        )

    client = MlflowClient()

    run_info = client.get_run(
        run_id
    )

    evidence = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": run_info.info.experiment_id,
        "run_name": run_name,
        "run_id": run_id,
        "model_name": model_name,
        "model_type": model_type,
        "model_version": version,
        "status": run_info.info.status,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
    }

    return evidence


# =========================================================
# Save sanitized MLflow evidence
# =========================================================
def save_mlflow_evidence(run_records):
    """
    Save a sanitized summary of MLflow experiments.

    Environment-specific fields such as artifact locations,
    local usernames and tracking paths are intentionally omitted.
    """

    evidence_columns = [
        "experiment_name",
        "experiment_id",
        "run_name",
        "run_id",
        "model_name",
        "model_type",
        "model_version",
        "status",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "mae",
        "r2",
    ]

    evidence_df = pd.DataFrame(
        run_records,
        columns=evidence_columns,
    )

    evidence_df.to_csv(
        MLFLOW_RUNS_FILE,
        index=False,
    )

    logger.info(
        "Sanitized MLflow evidence saved: results/%s",
        MLFLOW_RUNS_FILE.name,
    )


# =========================================================
# Main
# =========================================================
def main():
    """Track model experiments, versions and metrics with MLflow."""

    configure_logging()

    try:
        configure_mlflow()

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            results = json.load(
                file
            )

        classification = (
            results["classification"]
        )

        regression = (
            results["regression"]
        )

        # -----------------------------------------------------
        # Model version comparison
        # -----------------------------------------------------
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
            index=False,
        )

        logger.info(
            "Model version comparison saved: results/%s",
            VERSION_FILE.name,
        )

        # -----------------------------------------------------
        # MLflow experiment runs
        # -----------------------------------------------------
        run_records = []

        run_records.append(
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
        )

        run_records.append(
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
        )

        run_records.append(
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
        )

        run_records.append(
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
        )

        save_mlflow_evidence(
            run_records
        )

        # -----------------------------------------------------
        # User-facing summary
        # -----------------------------------------------------
        print(
            "\n=== MLFLOW EXPERIMENT TRACKING COMPLETE ==="
        )

        print(
            f"Experiment: {EXPERIMENT_NAME}"
        )

        print(
            f"Tracked model versions: {len(run_records)}"
        )

        print(
            "Local tracking database: mlflow.db "
            "(excluded from GitHub)"
        )

        print(
            "Model comparison: results/model_versions.csv"
        )

        print(
            "Sanitized run evidence: results/mlflow_runs.csv"
        )

        logger.info(
            "MLflow workflow completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "MLflow workflow failed",
            exc_info=True,
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )