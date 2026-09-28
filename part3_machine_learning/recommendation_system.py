import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


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
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "recommended_travel_windows.csv"
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
# Load data
# =========================================================
def load_data():
    """Load the Part 3 ML-ready traffic dataset."""

    df = pd.read_csv(
        INPUT_FILE
    )

    logger.info(
        "Recommendation dataset loaded: %s rows, %s columns",
        df.shape[0],
        df.shape[1]
    )

    df["day_type"] = df[
        "is_weekend"
    ].map(
        {
            0: "weekday",
            1: "weekend"
        }
    )

    df["weather_main"] = (
        df["weather_main"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# =========================================================
# Build recommendation table
# =========================================================
def build_recommendation_table(df):
    """
    Build historical lower-traffic travel windows.

    Hours from 05:00 to 22:00 are used to provide practical
    travel recommendations rather than overnight periods.
    """

    practical = df[
        df["hour"].between(
            5,
            22
        )
    ].copy()

    recommendations = []

    # -----------------------------------------------------
    # Overall recommendations by day type
    # -----------------------------------------------------
    for day_type in [
        "weekday",
        "weekend"
    ]:

        subset = practical[
            practical["day_type"]
            == day_type
        ]

        hourly = (
            subset.groupby(
                "hour"
            )
            .agg(
                average_traffic=(
                    "traffic_volume",
                    "mean"
                ),
                observations=(
                    "traffic_volume",
                    "size"
                )
            )
            .reset_index()
            .sort_values(
                "average_traffic"
            )
        )

        for _, row in hourly.head(3).iterrows():

            recommendations.append(
                {
                    "day_type": day_type,
                    "weather": "all",
                    "start_hour": int(
                        row["hour"]
                    ),
                    "end_hour": (
                        int(row["hour"]) + 1
                    ) % 24,
                    "average_traffic": round(
                        float(
                            row[
                                "average_traffic"
                            ]
                        ),
                        2
                    ),
                    "observations": int(
                        row["observations"]
                    )
                }
            )

    # -----------------------------------------------------
    # Recommendations by day type and weather
    # -----------------------------------------------------
    weather_values = sorted(
        practical[
            "weather_main"
        ]
        .dropna()
        .unique()
    )

    for day_type in [
        "weekday",
        "weekend"
    ]:

        for weather in weather_values:

            subset = practical[
                (
                    practical["day_type"]
                    == day_type
                )
                &
                (
                    practical["weather_main"]
                    == weather
                )
            ]

            if len(subset) < 30:
                continue

            hourly = (
                subset.groupby(
                    "hour"
                )
                .agg(
                    average_traffic=(
                        "traffic_volume",
                        "mean"
                    ),
                    observations=(
                        "traffic_volume",
                        "size"
                    )
                )
                .reset_index()
            )

            # Avoid recommendations based on extremely
            # small hourly samples.
            hourly = hourly[
                hourly["observations"] >= 5
            ]

            if hourly.empty:
                continue

            hourly = hourly.sort_values(
                "average_traffic"
            )

            for _, row in hourly.head(3).iterrows():

                recommendations.append(
                    {
                        "day_type": day_type,
                        "weather": weather,
                        "start_hour": int(
                            row["hour"]
                        ),
                        "end_hour": (
                            int(row["hour"]) + 1
                        ) % 24,
                        "average_traffic": round(
                            float(
                                row[
                                    "average_traffic"
                                ]
                            ),
                            2
                        ),
                        "observations": int(
                            row["observations"]
                        )
                    }
                )

    recommendation_df = pd.DataFrame(
        recommendations
    )

    recommendation_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        "Travel recommendation table saved: results/%s",
        OUTPUT_FILE.name
    )

    return recommendation_df


# =========================================================
# Get recommendation
# =========================================================
def recommend(
    recommendation_df,
    day_type,
    weather=None
):
    """Return the best historical travel window."""

    day_type = day_type.lower()

    selected_weather = (
        weather.lower().strip()
        if weather
        else "all"
    )

    subset = recommendation_df[
        (
            recommendation_df[
                "day_type"
            ]
            == day_type
        )
        &
        (
            recommendation_df[
                "weather"
            ]
            == selected_weather
        )
    ]

    # If the requested weather has insufficient observations,
    # fall back to the overall day-type recommendation.
    if subset.empty and weather:

        logger.warning(
            "No reliable recommendation for weather '%s'; "
            "using overall %s traffic pattern",
            weather,
            day_type
        )

        selected_weather = "all"

        subset = recommendation_df[
            (
                recommendation_df[
                    "day_type"
                ]
                == day_type
            )
            &
            (
                recommendation_df[
                    "weather"
                ]
                == "all"
            )
        ]

    if subset.empty:
        raise ValueError(
            "No recommendation could be generated."
        )

    best = (
        subset.sort_values(
            "average_traffic"
        )
        .iloc[0]
    )

    start_hour = int(
        best["start_hour"]
    )

    end_hour = int(
        best["end_hour"]
    )

    avg_traffic = float(
        best["average_traffic"]
    )

    if selected_weather == "all":

        message = (
            f"For a {day_type} journey, consider travelling "
            f"between {start_hour:02d}:00 and "
            f"{end_hour:02d}:00. Historical traffic volume "
            f"during this window averages approximately "
            f"{avg_traffic:,.0f} vehicles."
        )

    else:

        message = (
            f"For a {day_type} journey during "
            f"{selected_weather} weather, consider travelling "
            f"between {start_hour:02d}:00 and "
            f"{end_hour:02d}:00. Historical traffic volume "
            f"during this window averages approximately "
            f"{avg_traffic:,.0f} vehicles."
        )

    return message


# =========================================================
# Command line interface
# =========================================================
def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Smart City Traffic Travel-Time Recommendation System"
        )
    )

    parser.add_argument(
        "--day-type",
        choices=[
            "weekday",
            "weekend"
        ],
        default="weekday",
        help="Journey day type"
    )

    parser.add_argument(
        "--weather",
        default=None,
        help=(
            "Optional weather condition such as "
            "clear, clouds, rain or snow"
        )
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():

    try:

        args = parse_args()

        logger.info(
            "Recommendation requested: day_type=%s, weather=%s",
            args.day_type,
            args.weather
        )

        df = load_data()

        recommendation_df = (
            build_recommendation_table(
                df
            )
        )

        message = recommend(
            recommendation_df,
            args.day_type,
            args.weather
        )

        # Direct output for the end user
        print(
            "\n=== TRAVEL RECOMMENDATION ==="
        )

        print(
            message
        )

        logger.info(
            "Recommendation system completed successfully"
        )

        return 0

    except Exception:

        logger.error(
            "Recommendation system failed",
            exc_info=True
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )