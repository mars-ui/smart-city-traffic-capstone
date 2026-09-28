import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR = SCRIPT_DIR / "models"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = SCRIPT_DIR / "part3.log"


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
# K-means evaluation
# =========================================================
def evaluate_kmeans(X_scaled):
    """
    Evaluate candidate K-means solutions using inertia and
    silhouette score.

    Silhouette score is calculated on a reproducible sample
    to keep evaluation efficient for the full traffic dataset.
    """

    logger.info(
        "Evaluating K-means candidate cluster counts from k=2 to k=8"
    )

    evaluation_rows = []

    silhouette_sample_size = min(
        10000,
        len(X_scaled),
    )

    for k in range(2, 9):
        candidate_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        candidate_labels = candidate_model.fit_predict(
            X_scaled
        )

        silhouette = silhouette_score(
            X_scaled,
            candidate_labels,
            sample_size=silhouette_sample_size,
            random_state=42,
        )

        evaluation_rows.append(
            {
                "k": k,
                "inertia": candidate_model.inertia_,
                "silhouette_score": silhouette,
            }
        )

        logger.info(
            "K-means evaluation: k=%s | inertia=%.2f | silhouette=%.4f",
            k,
            candidate_model.inertia_,
            silhouette,
        )

    evaluation_df = pd.DataFrame(
        evaluation_rows
    )

    evaluation_file = (
        RESULTS_DIR
        / "kmeans_evaluation.csv"
    )

    evaluation_df.to_csv(
        evaluation_file,
        index=False,
    )

    logger.info(
        "K-means evaluation saved: results/%s",
        evaluation_file.name,
    )

    best_row = evaluation_df.loc[
        evaluation_df["silhouette_score"].idxmax()
    ]

    best_k = int(
        best_row["k"]
    )

    best_silhouette = float(
        best_row["silhouette_score"]
    )

    logger.info(
        "Selected k=%s based on highest silhouette score %.4f",
        best_k,
        best_silhouette,
    )

    # -----------------------------------------------------
    # Evaluation figure
    # -----------------------------------------------------
    fig, axis_1 = plt.subplots(
        figsize=(8, 5)
    )

    axis_1.plot(
        evaluation_df["k"],
        evaluation_df["inertia"],
        marker="o",
    )

    axis_1.set_xlabel(
        "Number of clusters (k)"
    )

    axis_1.set_ylabel(
        "Inertia"
    )

    axis_1.set_title(
        "K-means Evaluation: Inertia and Silhouette Score"
    )

    axis_2 = axis_1.twinx()

    axis_2.plot(
        evaluation_df["k"],
        evaluation_df["silhouette_score"],
        marker="s",
    )

    axis_2.set_ylabel(
        "Silhouette Score"
    )

    axis_1.grid(
        alpha=0.3
    )

    fig.tight_layout()

    evaluation_figure = (
        FIGURES_DIR
        / "kmeans_evaluation.png"
    )

    fig.savefig(
        evaluation_figure,
        dpi=300,
    )

    plt.close(fig)

    logger.info(
        "K-means evaluation figure saved: figures/%s",
        evaluation_figure.name,
    )

    return (
        evaluation_df,
        best_k,
        best_silhouette,
    )


# =========================================================
# K-means clustering
# =========================================================
def run_kmeans(df):
    """Evaluate and cluster traffic operating conditions."""

    logger.info(
        "Starting K-means clustering"
    )

    cluster_features = [
        "hour",
        "weather_severity",
        "traffic_volume",
        "is_weekend",
    ]

    missing_features = [
        feature
        for feature in cluster_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing clustering features: {missing_features}"
        )

    X = df[
        cluster_features
    ].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    (
        evaluation_df,
        best_k,
        best_silhouette,
    ) = evaluate_kmeans(
        X_scaled
    )

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    df = df.copy()

    df["cluster"] = kmeans.fit_predict(
        X_scaled
    )

    # -----------------------------------------------------
    # Save fitted model and preprocessing object
    # -----------------------------------------------------
    joblib.dump(
        kmeans,
        MODEL_DIR / "kmeans_model.joblib",
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "kmeans_scaler.joblib",
    )

    logger.info(
        "K-means model and scaler saved successfully"
    )

    # -----------------------------------------------------
    # Cluster profiles
    # -----------------------------------------------------
    profile = (
        df.groupby("cluster")
        .agg(
            records=(
                "cluster",
                "size",
            ),
            avg_hour=(
                "hour",
                "mean",
            ),
            avg_traffic=(
                "traffic_volume",
                "mean",
            ),
            avg_weather_severity=(
                "weather_severity",
                "mean",
            ),
            weekend_share=(
                "is_weekend",
                "mean",
            ),
        )
        .round(2)
    )

    interpretations = {}

    for cluster_id, row in profile.iterrows():
        if row["avg_traffic"] >= 4500:
            traffic_label = "high traffic"
        elif row["avg_traffic"] >= 2500:
            traffic_label = "moderate traffic"
        else:
            traffic_label = "low traffic"

        if row["avg_weather_severity"] >= 1.5:
            weather_label = "more severe weather"
        elif row["avg_weather_severity"] >= 0.5:
            weather_label = "mixed weather"
        else:
            weather_label = "mostly mild weather"

        if row["weekend_share"] >= 0.5:
            day_label = "weekend-oriented"
        else:
            day_label = "weekday-oriented"

        interpretations[str(cluster_id)] = (
            f"Cluster {cluster_id}: {traffic_label}, "
            f"{weather_label}, {day_label}, with an "
            f"average hour near {row['avg_hour']:.1f}."
        )

    profile_file = (
        RESULTS_DIR
        / "kmeans_cluster_profiles.csv"
    )

    profile.to_csv(
        profile_file
    )

    logger.info(
        "K-means clustering completed with %s clusters",
        best_k,
    )

    logger.info(
        "Cluster profiles saved: results/%s",
        profile_file.name,
    )

    return (
        df,
        profile,
        interpretations,
        evaluation_df,
        best_k,
        best_silhouette,
    )


# =========================================================
# Association-rule preparation
# =========================================================
def create_transaction_features(df):
    """Create categorical variables for association rules."""

    result = df.copy()

    result["time_of_day"] = pd.cut(
        result["hour"],
        bins=[
            -1,
            5,
            11,
            16,
            20,
            23,
        ],
        labels=[
            "Overnight",
            "Morning",
            "Midday",
            "EveningPeak",
            "Night",
        ],
    )

    result["day_type"] = result[
        "is_weekend"
    ].map(
        {
            0: "Weekday",
            1: "Weekend",
        }
    )

    result["weather_group"] = pd.cut(
        result["weather_severity"],
        bins=[
            -1,
            0,
            1,
            3,
        ],
        labels=[
            "MildWeather",
            "ReducedVisibility",
            "SevereWeather",
        ],
    )

    return result


# =========================================================
# Association-rule mining
# =========================================================
def mine_rules(df):
    """
    Mine association rules where congestion level is the
    consequent.

    Antecedents use time of day, day type and weather group.
    """

    logger.info(
        "Starting association-rule mining"
    )

    df = create_transaction_features(
        df
    )

    antecedent_columns = [
        "time_of_day",
        "day_type",
        "weather_group",
    ]

    congestion_levels = [
        "Low",
        "Medium",
        "High",
        "Severe",
    ]

    total_rows = len(
        df
    )

    rules = []

    # Generate one- and two-condition antecedents
    for size in [
        1,
        2,
    ]:
        for column_combo in combinations(
            antecedent_columns,
            size,
        ):
            grouped = (
                df.groupby(
                    list(column_combo),
                    observed=True,
                )
                .size()
                .reset_index(
                    name="antecedent_count"
                )
            )

            for _, group_row in grouped.iterrows():
                mask = pd.Series(
                    True,
                    index=df.index,
                )

                antecedent_parts = []

                for column in column_combo:
                    value = group_row[
                        column
                    ]

                    mask &= (
                        df[column]
                        == value
                    )

                    antecedent_parts.append(
                        f"{column}={value}"
                    )

                antecedent_count = int(
                    mask.sum()
                )

                if antecedent_count == 0:
                    continue

                antecedent_support = (
                    antecedent_count
                    / total_rows
                )

                # Avoid extremely rare rules
                if antecedent_support < 0.02:
                    continue

                for congestion in congestion_levels:
                    consequent_mask = (
                        df["congestion_category"]
                        == congestion
                    )

                    joint_count = int(
                        (
                            mask
                            & consequent_mask
                        ).sum()
                    )

                    if joint_count == 0:
                        continue

                    support = (
                        joint_count
                        / total_rows
                    )

                    confidence = (
                        joint_count
                        / antecedent_count
                    )

                    consequent_support = (
                        consequent_mask.mean()
                    )

                    if consequent_support == 0:
                        continue

                    lift = (
                        confidence
                        / consequent_support
                    )

                    rules.append(
                        {
                            "antecedent": " AND ".join(
                                antecedent_parts
                            ),
                            "consequent": (
                                f"congestion={congestion}"
                            ),
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                        }
                    )

    rules_columns = [
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
    ]

    rules_df = pd.DataFrame(
        rules,
        columns=rules_columns,
    )

    if rules_df.empty:
        logger.warning(
            "No association rules were generated"
        )

        top_rules = rules_df.copy()

    else:
        rules_df = rules_df.sort_values(
            [
                "lift",
                "confidence",
                "support",
            ],
            ascending=False,
        )

        top_rules = (
            rules_df[
                rules_df["confidence"] >= 0.20
            ]
            .head(15)
            .copy()
        )

    rules_file = (
        RESULTS_DIR
        / "association_rules.csv"
    )

    top_rules.to_csv(
        rules_file,
        index=False,
    )

    logger.info(
        "Association-rule mining completed: %s rules retained",
        len(top_rules),
    )

    logger.info(
        "Association rules saved: results/%s",
        rules_file.name,
    )

    return top_rules


# =========================================================
# Main
# =========================================================
def main():
    """Run unsupervised learning tasks."""

    configure_logging()

    try:
        df = pd.read_csv(
            INPUT_FILE
        )

        logger.info(
            "ML-ready dataset loaded for unsupervised learning: "
            "%s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )

        (
            clustered_df,
            profile,
            interpretations,
            evaluation_df,
            best_k,
            best_silhouette,
        ) = run_kmeans(
            df
        )

        top_rules = mine_rules(
            clustered_df
        )

        summary = {
            "kmeans_evaluation": {
                "selected_k": best_k,
                "best_silhouette_score": best_silhouette,
                "selection_method": (
                    "Highest silhouette score among k=2 through k=8; "
                    "inertia also recorded for cluster-quality assessment."
                ),
                "evaluated_candidates": (
                    evaluation_df
                    .round(6)
                    .to_dict(
                        orient="records"
                    )
                ),
            },
            "cluster_interpretations": interpretations,
            "top_association_rules": (
                top_rules
                .head(5)
                .round(6)
                .to_dict(
                    orient="records"
                )
            ),
        }

        summary_file = (
            RESULTS_DIR
            / "unsupervised_summary.json"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
                default=str,
            )

        logger.info(
            "Unsupervised summary saved: results/%s",
            summary_file.name,
        )

        # -----------------------------------------------------
        # User-facing results
        # -----------------------------------------------------
        print(
            "\n=== K-MEANS MODEL SELECTION ==="
        )

        print(
            evaluation_df
            .round(
                {
                    "inertia": 2,
                    "silhouette_score": 4,
                }
            )
            .to_string(
                index=False
            )
        )

        print(
            f"\nSelected k: {best_k}"
        )

        print(
            f"Best silhouette score: "
            f"{best_silhouette:.4f}"
        )

        print(
            "\n=== K-MEANS CLUSTER PROFILES ==="
        )

        print(
            profile.to_string()
        )

        print(
            "\n=== CLUSTER INTERPRETATIONS ==="
        )

        for text in interpretations.values():
            print(
                text
            )

        print(
            "\n=== TOP ASSOCIATION RULES ==="
        )

        if top_rules.empty:
            print(
                "No rules met the selected thresholds."
            )
        else:
            print(
                top_rules[
                    [
                        "antecedent",
                        "consequent",
                        "support",
                        "confidence",
                        "lift",
                    ]
                ]
                .head(10)
                .round(3)
                .to_string(
                    index=False
                )
            )

        logger.info(
            "Unsupervised learning workflow completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "Unsupervised learning workflow failed",
            exc_info=True,
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )