import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path


# -------------------------------------------------
# Paths
# -------------------------------------------------
script_dir = Path(__file__).resolve().parent

input_file = (
    script_dir
    / "output"
    / "feature_engineered_traffic.csv"
)

figures_dir = (
    script_dir
    / "figures"
)

log_file = (
    script_dir
    / "pipeline.log"
)


# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def create_visualizations(df):
    """Create and save traffic visualisations."""

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    interpretations = []

    # =================================================
    # 1. Average traffic volume by hour
    # =================================================
    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        hourly_traffic.index,
        hourly_traffic.values,
        marker="o"
    )

    plt.title(
        "Average Traffic Volume by Hour"
    )
    plt.xlabel(
        "Hour of Day"
    )
    plt.ylabel(
        "Average Traffic Volume"
    )

    plt.xticks(
        range(0, 24)
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    hourly_path = (
        figures_dir
        / "traffic_by_hour.png"
    )

    plt.savefig(
        hourly_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully: %s",
        hourly_path.relative_to(script_dir)
    )

    peak_hour = hourly_traffic.idxmax()
    peak_volume = hourly_traffic.max()

    low_hour = hourly_traffic.idxmin()
    low_volume = hourly_traffic.min()

    interpretations.append(
        f"1. Traffic by Hour: Average traffic is highest around "
        f"{peak_hour}:00 ({peak_volume:.0f} vehicles) and lowest around "
        f"{low_hour}:00 ({low_volume:.0f} vehicles). This indicates a "
        f"clear hourly traffic-demand pattern."
    )

    # =================================================
    # 2. Weekday versus weekend traffic
    # =================================================
    weekend_summary = (
        df.groupby("is_weekend")["traffic_volume"]
        .mean()
    )

    weekday_avg = weekend_summary.get(
        0,
        0
    )

    weekend_avg = weekend_summary.get(
        1,
        0
    )

    labels = [
        "Weekday",
        "Weekend"
    ]

    values = [
        weekday_avg,
        weekend_avg
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Average Traffic Volume: Weekday vs Weekend"
    )
    plt.xlabel(
        "Day Type"
    )
    plt.ylabel(
        "Average Traffic Volume"
    )

    plt.tight_layout()

    weekend_path = (
        figures_dir
        / "weekday_vs_weekend.png"
    )

    plt.savefig(
        weekend_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully: %s",
        weekend_path.relative_to(script_dir)
    )

    if weekday_avg > weekend_avg:
        difference = (
            weekday_avg
            - weekend_avg
        )

        comparison = (
            f"weekday traffic is higher than weekend traffic "
            f"by {difference:.0f} vehicles on average"
        )

    else:
        difference = (
            weekend_avg
            - weekday_avg
        )

        comparison = (
            f"weekend traffic is higher than weekday traffic "
            f"by {difference:.0f} vehicles on average"
        )

    interpretations.append(
        f"2. Weekday vs Weekend: Average weekday traffic is "
        f"{weekday_avg:.0f}, while average weekend traffic is "
        f"{weekend_avg:.0f}. Therefore, {comparison}."
    )

    # =================================================
    # 3. Temperature versus traffic volume
    # =================================================
    plt.figure(figsize=(9, 5))

    plt.scatter(
        df["temp"],
        df["traffic_volume"],
        alpha=0.15,
        s=10
    )

    plt.title(
        "Temperature vs Traffic Volume"
    )
    plt.xlabel(
        "Temperature (Kelvin)"
    )
    plt.ylabel(
        "Traffic Volume"
    )

    plt.tight_layout()

    temp_path = (
        figures_dir
        / "temperature_vs_traffic.png"
    )

    plt.savefig(
        temp_path,
        dpi=300
    )

    plt.close()

    logger.info(
        "Figure saved successfully: %s",
        temp_path.relative_to(script_dir)
    )

    correlation = df["temp"].corr(
        df["traffic_volume"]
    )

    interpretations.append(
        f"3. Temperature vs Traffic: The correlation between "
        f"temperature and traffic volume is {correlation:.3f}. "
        f"This indicates that temperature alone has only a limited "
        f"linear relationship with traffic demand."
    )

    # =================================================
    # Save interpretations
    # =================================================
    interpretation_file = (
        figures_dir
        / "visualization_interpretations.txt"
    )

    with open(
        interpretation_file,
        "w",
        encoding="utf-8"
    ) as file:
        for interpretation in interpretations:
            file.write(
                interpretation
                + "\n\n"
            )

    logger.info(
        "Visualisation interpretations saved to: %s",
        interpretation_file.relative_to(script_dir)
    )


def main():
    """Run the visualisation workflow."""

    try:
        df = pd.read_csv(
            input_file
        )

        logger.info(
            "Feature-engineered dataset loaded for visualisation: "
            "%s rows, %s columns",
            df.shape[0],
            df.shape[1]
        )

        create_visualizations(
            df
        )

        logger.info(
            "All visualisations completed successfully"
        )

        return 0

    except Exception:
        logger.error(
            "Visualisation process failed",
            exc_info=True
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )