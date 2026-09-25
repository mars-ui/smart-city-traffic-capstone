import pandas as pd
import argparse
import logging
import sys
from pathlib import Path


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


def load_processed_data():
    """Load the feature-engineered traffic dataset."""

    file_path = (
        Path(__file__).resolve().parent
        / "output"
        / "feature_engineered_traffic.csv"
    )

    df = pd.read_csv(file_path)
    df["date_time"] = pd.to_datetime(df["date_time"])

    return df


def query_date(df, date_text):
    """Show traffic records and summary for a specific date."""

    try:
        selected_date = pd.to_datetime(
            date_text,
            format="%Y-%m-%d",
            errors="raise"
        ).date()

    except ValueError:
        logger.error(
            "Invalid date '%s'. Expected format: YYYY-MM-DD",
            date_text
        )

        print(
            "Invalid date. Please use YYYY-MM-DD, "
            "for example 2017-01-15."
        )
        return

    logger.info(
        "Command invoked: date | date=%s",
        date_text
    )

    result = df[
        df["date_time"].dt.date == selected_date
    ]

    if result.empty:
        print(f"No traffic records found for {date_text}.")
        return

    print(f"\nTraffic summary for {date_text}")
    print("-" * 45)

    print(
        f"Number of observations: {len(result)}"
    )

    print(
        f"Average traffic volume: "
        f"{result['traffic_volume'].mean():.0f}"
    )

    print(
        f"Maximum traffic volume: "
        f"{result['traffic_volume'].max():.0f}"
    )

    print(
        f"Minimum traffic volume: "
        f"{result['traffic_volume'].min():.0f}"
    )

    display_columns = [
        "date_time",
        "traffic_volume",
        "weather_description"
    ]

    print("\nHourly records:")
    print(
        result[display_columns]
        .sort_values("date_time")
        .to_string(index=False)
    )


def high_traffic(df, threshold):
    """Identify periods where traffic exceeds a threshold."""

    logger.info(
        "Command invoked: high | threshold=%s",
        threshold
    )

    result = df[
        df["traffic_volume"] > threshold
    ].copy()

    print(
        f"\nTraffic periods above {threshold:.0f}"
    )
    print("-" * 45)

    print(
        f"Number of high-traffic observations: "
        f"{len(result)}"
    )

    if result.empty:
        print("No observations exceeded the threshold.")
        return

    top_periods = (
        result.sort_values(
            "traffic_volume",
            ascending=False
        )
        .head(10)
    )

    display_columns = [
        "date_time",
        "traffic_volume",
        "weather_description"
    ]

    print("\nTop 10 highest traffic observations:")

    print(
        top_periods[display_columns]
        .to_string(index=False)
    )


def compare_day_type(df):
    """Compare weekday and weekend traffic."""

    logger.info(
        "Command invoked: compare | no additional arguments"
    )

    summary = (
        df.groupby("is_weekend")["traffic_volume"]
        .agg(["mean", "median", "count"])
    )

    weekday = summary.loc[0]
    weekend = summary.loc[1]

    print("\nWeekday vs Weekend Traffic")
    print("-" * 45)

    print(
        f"Weekday average traffic: "
        f"{weekday['mean']:.0f}"
    )

    print(
        f"Weekend average traffic: "
        f"{weekend['mean']:.0f}"
    )

    print(
        f"Weekday median traffic: "
        f"{weekday['median']:.0f}"
    )

    print(
        f"Weekend median traffic: "
        f"{weekend['median']:.0f}"
    )

    difference = weekday["mean"] - weekend["mean"]

    print(
        f"Average difference "
        f"(weekday - weekend): "
        f"{difference:.0f}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Mini Traffic Analytics Application"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # Command 1: date
    date_parser = subparsers.add_parser(
        "date",
        help="Query traffic for a specific date"
    )

    date_parser.add_argument(
        "--date",
        required=True,
        help="Date in YYYY-MM-DD format"
    )

    # Command 2: high
    high_parser = subparsers.add_parser(
        "high",
        help="Identify high-traffic periods"
    )

    high_parser.add_argument(
        "--threshold",
        type=float,
        default=5500,
        help="Traffic-volume threshold (default: 5500)"
    )

    # Command 3: compare
    subparsers.add_parser(
        "compare",
        help="Compare weekday and weekend traffic"
    )

    args = parser.parse_args()

    try:
        df = load_processed_data()

        if args.command == "date":
            query_date(
                df,
                args.date
            )

        elif args.command == "high":

            if args.threshold < 0:
                logger.error(
                    "Invalid traffic threshold: %s",
                    args.threshold
                )

                print(
                    "Threshold must be zero or greater."
                )

                return 1

            high_traffic(
                df,
                args.threshold
            )

        elif args.command == "compare":
            compare_day_type(df)

        return 0

    except Exception:
        logger.error(
            "Traffic application failed",
            exc_info=True
        )

        print(
            "The application could not complete the request. "
            "Please check pipeline.log."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())