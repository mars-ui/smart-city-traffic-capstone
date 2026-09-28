import logging
import sys
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]


def configure_logging():
    """Configure console and file logging for the pipeline entry point."""

    log_file = Path(__file__).resolve().parent / "pipeline.log"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def load_data(file_path):
    """Load traffic data from CSV."""

    try:
        df = pd.read_csv(file_path)

        logger.info(
            "Dataset loaded successfully: %s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )

        return df

    except FileNotFoundError:
        logger.error(
            "CSV file not found: %s",
            file_path,
            exc_info=True,
        )
        return None

    except Exception:
        logger.error(
            "Unexpected error while loading dataset",
            exc_info=True,
        )
        return None


def validate_schema(df):
    """Check that required columns exist."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Missing required columns: %s",
            missing_columns,
        )
        return False

    logger.info("Schema validation successful")
    return True


def clean_data(df):
    """Clean and validate the traffic dataset."""

    df = df.copy()

    # -------------------------------------------------
    # 1. Standardise categorical values
    # -------------------------------------------------
    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description",
    ]

    for column in categorical_columns:
        original = df[column].astype("string")
        cleaned = original.str.strip().str.lower()

        changed_count = (
            original.fillna("<NA>")
            != cleaned.fillna("<NA>")
        ).sum()

        df[column] = cleaned

        if changed_count > 0:
            logger.warning(
                "%s rows modified while standardising %s",
                changed_count,
                column,
            )
        else:
            logger.info(
                "No changes required for %s",
                column,
            )

    # -------------------------------------------------
    # 2. Parse and validate date/time
    # -------------------------------------------------
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce",
    )

    invalid_dates = df["date_time"].isna().sum()

    if invalid_dates > 0:
        logger.warning(
            "%s rows dropped because date_time could not be parsed",
            invalid_dates,
        )

        df = df.dropna(
            subset=["date_time"]
        )
    else:
        logger.info(
            "All date_time values parsed successfully"
        )

    # -------------------------------------------------
    # 3. Remove duplicate rows
    # -------------------------------------------------
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

        logger.warning(
            "%s duplicate rows removed",
            duplicate_count,
        )
    else:
        logger.info(
            "No duplicate rows found"
        )

    # Month used for monthly median imputation
    df["_month"] = df["date_time"].dt.month

    # -------------------------------------------------
    # 4. Handle impossible temperature values
    # -------------------------------------------------
    invalid_temp = (
        df["temp"].isna()
        | (df["temp"] <= 0)
    )

    invalid_temp_count = invalid_temp.sum()

    if invalid_temp_count > 0:
        df.loc[
            invalid_temp,
            "temp",
        ] = pd.NA

        for month in sorted(
            df["_month"].dropna().unique()
        ):
            month_mask = (
                df["_month"] == month
            )

            monthly_median = df.loc[
                month_mask
                & df["temp"].notna(),
                "temp",
            ].median()

            if pd.isna(monthly_median):
                monthly_median = (
                    df["temp"].median()
                )

            df.loc[
                month_mask
                & df["temp"].isna(),
                "temp",
            ] = monthly_median

        logger.warning(
            "%s impossible/missing temperature values "
            "imputed using monthly median",
            invalid_temp_count,
        )

    else:
        logger.info(
            "No impossible temperature values found"
        )

    # -------------------------------------------------
    # 5. Handle impossible rainfall values
    # -------------------------------------------------
    invalid_rain = (
        df["rain_1h"].isna()
        | (df["rain_1h"] < 0)
        | (df["rain_1h"] > 9000)
    )

    invalid_rain_count = invalid_rain.sum()

    if invalid_rain_count > 0:
        df.loc[
            invalid_rain,
            "rain_1h",
        ] = pd.NA

        for month in sorted(
            df["_month"].dropna().unique()
        ):
            month_mask = (
                df["_month"] == month
            )

            monthly_median = df.loc[
                month_mask
                & df["rain_1h"].notna(),
                "rain_1h",
            ].median()

            if pd.isna(monthly_median):
                monthly_median = (
                    df["rain_1h"].median()
                )

            df.loc[
                month_mask
                & df["rain_1h"].isna(),
                "rain_1h",
            ] = monthly_median

        logger.warning(
            "%s impossible/missing rainfall values "
            "imputed using monthly median",
            invalid_rain_count,
        )

    else:
        logger.info(
            "No impossible rainfall values found"
        )

    # Remove temporary helper column
    df = df.drop(
        columns=["_month"]
    )

    logger.info(
        "Cleaning finished: %s rows, %s columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def main():
    """Run the complete data-cleaning pipeline."""

    configure_logging()

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    input_file = (
        project_root
        / "data"
        / "Metro_Interstate_Traffic_Volume.csv"
    )

    output_dir = (
        Path(__file__)
        .resolve()
        .parent
        / "output"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        output_dir
        / "cleaned_traffic.csv"
    )

    try:
        # Step 1: Load data
        df = load_data(input_file)

        if df is None:
            logger.error(
                "Pipeline stopped because "
                "the dataset could not be loaded."
            )
            return 1

        # Step 2: Validate schema
        if not validate_schema(df):
            logger.error(
                "Pipeline stopped because "
                "schema validation failed."
            )
            return 1

        # Step 3: Clean data
        cleaned_df = clean_data(df)

        # Step 4: Save cleaned dataset
        cleaned_df.to_csv(
            output_file,
            index=False,
        )

        logger.info(
            "Cleaned dataset saved successfully to: %s",
            output_file.relative_to(
                Path(__file__)
                .resolve()
                .parent
            ),
        )

        logger.info(
            "Pipeline completed successfully."
        )

        return 0

    except Exception:
        logger.error(
            "Pipeline failed unexpectedly.",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())