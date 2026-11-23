"""
Machine Maintenance Prediction - Data Cleaning
------------------------------------------------
This script:
1. Loads the original machine readings CSV.
2. Inspects the dataset and records data-quality findings.
3. Adds a temperature-missing indicator BEFORE imputing temperature.
4. Imputes missing temperature with each machine's median temperature.
5. Preserves the original vibration field and creates vibration_abs = ABS(vibration).
6. Does NOT remove high absolute vibration values or other statistical outliers.
7. Parses timestamps and validates categorical/target values.
8. Saves a separate cleaned CSV and a text cleaning report.

Run:
    python src/clean_machine_readings.py   (from the repo root)

The original CSV is never overwritten.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# STEP 0: Define input and output file locations
# ------------------------------------------------------------
# The script lives in <repo>/src/. The original CSV is read from
# <repo>/data/ and output files are also saved in <repo>/data/.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
INPUT_FILE = DATA_DIR / "proj_machine_readings.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_machine_readings.csv"
REPORT_FILE = DATA_DIR / "data_cleaning_output.txt"

# Raw timestamps are day-first (e.g. 15/1/2025 19:33). Without an explicit
# format pandas guesses month-first, which turns day > 12 into NaT and
# silently swaps day/month for the rest.
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"   # raw timestamps are day-first, e.g. 15/1/2025 19:33


def log(message, report_lines):
    """Print a message and also store it for the text report."""
    print(message)
    report_lines.append(str(message))


def main():
    report_lines = []

    log("=" * 72, report_lines)
    log("MACHINE MAINTENANCE DATA CLEANING REPORT", report_lines)
    log("=" * 72, report_lines)

    # --------------------------------------------------------
    # STEP 1: Load the original dataset
    # --------------------------------------------------------
    # Read the CSV without modifying the source file.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place proj_machine_readings.csv in the repo's data/ folder."
        )

    df = pd.read_csv(INPUT_FILE)

    log("\nSTEP 1 - LOAD DATASET", report_lines)
    log(f"Input file: {INPUT_FILE.name}", report_lines)
    log(f"Dataset shape (rows, columns): {df.shape}", report_lines)
    log(f"Column names: {df.columns.tolist()}", report_lines)

    # Confirm that the expected project columns are present.
    expected_columns = [
        "timestamp", "machine_id", "temperature", "vibration",
        "humidity", "pressure", "energy_consumption",
        "machine_status", "anomaly_flag",
        "predicted_remaining_life", "failure_type",
        "downtime_risk", "maintenance_required"
    ]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Expected columns missing from CSV: {missing_columns}")

    log("\nFirst five rows:", report_lines)
    log(df.head().to_string(index=False), report_lines)

    log("\nOriginal data types:", report_lines)
    log(df.dtypes.to_string(), report_lines)

    # --------------------------------------------------------
    # STEP 2: Inspect missing values BEFORE cleaning
    # --------------------------------------------------------
    # This records the amount of missing data before any values
    # are filled, so the cleaning decisions can be documented.
    log("\nSTEP 2 - MISSING VALUE INSPECTION (BEFORE CLEANING)", report_lines)
    missing_before = df.isna().sum()
    missing_pct_before = (missing_before / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "missing_count": missing_before,
        "missing_percent": missing_pct_before
    })
    log(missing_summary.to_string(), report_lines)

    # --------------------------------------------------------
    # STEP 3: Add temperature missing indicator BEFORE imputation
    # --------------------------------------------------------
    # 1 means the original temperature was missing.
    # 0 means the original temperature was present.
    # This preserves information about sensor/data missingness,
    # even after temperature values have been imputed.
    df["temperature_missing"] = df["temperature"].isna().astype("int8")

    missing_temperature_count = int(df["temperature"].isna().sum())
    log("\nSTEP 3 - TEMPERATURE MISSING INDICATOR", report_lines)
    log(
        f"temperature_missing = 1 for {missing_temperature_count:,} "
        "originally missing temperature readings.",
        report_lines
    )
    log(
        f"Indicator counts:\n{df['temperature_missing'].value_counts().sort_index().to_string()}",
        report_lines
    )

    # --------------------------------------------------------
    # STEP 4: Impute temperature using each machine's median
    # --------------------------------------------------------
    # Each missing temperature is replaced by the median of the
    # available temperature readings for the same machine_id.
    # If a machine has no valid temperature readings at all,
    # its median will still be missing; use the overall median
    # only as a fallback for those exceptional cases.
    df["temperature"] = df.groupby("machine_id")["temperature"].transform(
        lambda machine_values: machine_values.fillna(machine_values.median())
    )

    remaining_temperature_missing = int(df["temperature"].isna().sum())
    if remaining_temperature_missing > 0:
        overall_temperature_median = df["temperature"].median()
        df["temperature"] = df["temperature"].fillna(overall_temperature_median)
        log(
            "\nFallback applied: remaining missing temperatures were filled "
            "with the overall median because their machine had no usable "
            "temperature values.",
            report_lines
        )

    log("\nSTEP 4 - TEMPERATURE IMPUTATION", report_lines)
    log(f"Missing temperatures before imputation: {missing_temperature_count:,}", report_lines)
    log(
        f"Missing temperatures after imputation: {int(df['temperature'].isna().sum()):,}",
        report_lines
    )
    log("Temperature was imputed using machine-specific medians.", report_lines)
    log(
        "The temperature_missing indicator was retained to identify "
        "originally missing readings.",
        report_lines
    )

    # --------------------------------------------------------
    # STEP 5: Preserve vibration and create absolute vibration
    # --------------------------------------------------------
    # The unit/sign convention for vibration has not been confirmed.
    # Therefore, do NOT overwrite or delete the original vibration.
    # Create a separate non-negative magnitude field for analysis.
    # Missing values, if any, remain missing in vibration_abs.
    df["vibration_abs"] = df["vibration"].abs()

    log("\nSTEP 5 - VIBRATION HANDLING", report_lines)
    log("Original 'vibration' column has been retained unchanged.", report_lines)
    log("Created 'vibration_abs' = ABS(vibration).", report_lines)
    log(
        f"Negative values in original vibration: {(df['vibration'] < 0).sum():,}",
        report_lines
    )
    log(
        "No vibration readings were removed or replaced merely because "
        "they were negative or statistically extreme.",
        report_lines
    )
    log(
        "Vibration is treated as a questionable sensor: its relationship "
        "with the 'Vibration Issue' failure_type should be interpreted "
        "cautiously because the observed relationship is weak/almost absent.",
        report_lines
    )

    # Summarise vibration magnitude by failure type as a diagnostic.
    if "failure_type" in df.columns:
        vibration_by_failure = df.groupby(
            "failure_type", observed=False
        )["vibration_abs"].agg(["count", "median", "mean", "max"])
        log("\nAbsolute vibration summary by failure_type (diagnostic only):", report_lines)
        log(vibration_by_failure.to_string(), report_lines)

    # --------------------------------------------------------
    # STEP 6: Check exact duplicate rows and repeated machine-time keys
    # --------------------------------------------------------
    # Exact duplicates are counted and removed.
    # Repeated machine_id + timestamp combinations are reported,
    # but are NOT automatically removed because their meaning needs
    # to be checked before deciding whether they are true duplicates.
    exact_duplicates = int(df.duplicated().sum())
    repeated_machine_time = int(
        df.duplicated(subset=["machine_id", "timestamp"]).sum()
    )

    log("\nSTEP 6 - DUPLICATE CHECKS", report_lines)
    log(f"Exact duplicate rows found: {exact_duplicates:,}", report_lines)
    log(
        "Repeated machine_id + timestamp rows (reported, not automatically removed): "
        f"{repeated_machine_time:,}",
        report_lines
    )

    if exact_duplicates > 0:
        df = df.drop_duplicates().copy()
        log(f"Removed {exact_duplicates:,} exact duplicate rows.", report_lines)
    else:
        log("No exact duplicate rows needed to be removed.", report_lines)

    # --------------------------------------------------------
    # STEP 7: Convert timestamp to datetime
    # --------------------------------------------------------
    # Convert timestamp strings into datetime values so time-based
    # analysis and feature extraction can be performed later.
    parsed_timestamps = pd.to_datetime(
        df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    invalid_timestamps_before = int(parsed_timestamps.isna().sum())

    df["timestamp"] = parsed_timestamps

    log("\nSTEP 7 - TIMESTAMP CONVERSION", report_lines)
    log("Converted timestamp to pandas datetime.", report_lines)
    log(f"Invalid/missing timestamps after conversion: {df['timestamp'].isna().sum():,}", report_lines)

    # Do not silently drop rows with invalid timestamps. Report them
    # so the project team can decide how to handle them.
    if invalid_timestamps_before > 0:
        log(
            "Attention: invalid timestamps were found. They were not "
            "automatically deleted.",
            report_lines
        )

    # --------------------------------------------------------
    # STEP 8: Validate categorical and target values
    # --------------------------------------------------------
    # These checks compare observed values with the categories
    # described in the project problem statement.
    log("\nSTEP 8 - CATEGORY AND TARGET VALIDATION", report_lines)

    expected_values = {
        "machine_status": {0, 1, 2},
        "anomaly_flag": {0, 1},
        "failure_type": {
            "Normal", "Vibration Issue", "Overheating",
            "Pressure Drop", "Electrical Fault"
        },
        "maintenance_required": {0, 1}
    }

    for column, allowed_values in expected_values.items():
        observed_values = set(df[column].dropna().unique())
        unexpected_values = observed_values - allowed_values
        log(
            f"{column}: observed={sorted(observed_values, key=str)}; "
            f"unexpected={sorted(unexpected_values, key=str)}",
            report_lines
        )

    log("\nTarget distribution:", report_lines)
    target_counts = df["maintenance_required"].value_counts(dropna=False).sort_index()
    target_percentages = (
        df["maintenance_required"].value_counts(normalize=True, dropna=False)
        .sort_index() * 100
    ).round(2)
    target_summary = pd.DataFrame({
        "count": target_counts,
        "percentage": target_percentages
    })
    log(target_summary.to_string(), report_lines)

    # --------------------------------------------------------
    # STEP 9: Statistical outlier inspection - NO OUTLIER REMOVAL
    # --------------------------------------------------------
    # Report the IQR fences and number of observations outside them.
    # These are diagnostic flags only. High values may represent
    # genuine fault signals, so this script does not delete or cap them.
    log("\nSTEP 9 - OUTLIER INSPECTION (NO AUTOMATIC REMOVAL)", report_lines)

    numeric_sensor_columns = [
        "temperature", "vibration", "vibration_abs",
        "humidity", "pressure", "energy_consumption"
    ]

    outlier_rows = []
    for column in numeric_sensor_columns:
        values = df[column].dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_count = int(((values < lower_fence) | (values > upper_fence)).sum())

        outlier_rows.append({
            "column": column,
            "min": values.min(),
            "max": values.max(),
            "IQR_lower_fence": lower_fence,
            "IQR_upper_fence": upper_fence,
            "outside_IQR_fences": outlier_count
        })

    outlier_summary = pd.DataFrame(outlier_rows).set_index("column")
    log(outlier_summary.round(3).to_string(), report_lines)
    log(
        "No high absolute values were removed, capped, or changed merely "
        "because they were statistical outliers.",
        report_lines
    )

    # --------------------------------------------------------
    # STEP 10: Final data-quality validation
    # --------------------------------------------------------
    # Check missing values and dimensions after the cleaning steps.
    log("\nSTEP 10 - FINAL VALIDATION", report_lines)
    log(f"Final dataset shape: {df.shape}", report_lines)
    log("\nFinal missing-value counts:", report_lines)
    log(df.isna().sum().to_string(), report_lines)
    log(f"\nFinal exact duplicate count: {df.duplicated().sum():,}", report_lines)
    log(
        f"temperature_missing indicator total: {int(df['temperature_missing'].sum()):,}",
        report_lines
    )
    log(
        f"vibration_abs missing count: {int(df['vibration_abs'].isna().sum()):,}",
        report_lines
    )

    # --------------------------------------------------------
    # STEP 11: Save cleaned data and the run report
    # --------------------------------------------------------
    # Save to a new file so the original dataset remains untouched.
    df.to_csv(OUTPUT_FILE, index=False)

    log("\nSTEP 11 - SAVE OUTPUT", report_lines)
    log(f"Cleaned dataset saved to: {OUTPUT_FILE.name}", report_lines)
    log(f"Cleaned dataset rows: {len(df):,}", report_lines)
    log(f"Cleaned dataset columns: {len(df.columns):,}", report_lines)

    log("\nIMPORTANT MODELLING NOTES", report_lines)
    log(
        "1. Do not use post-event or target-derived columns as predictors "
        "without checking when they become available; they may cause data leakage.",
        report_lines
    )
    log(
        "2. Keep the original vibration field and vibration_abs separate. "
        "Assess their usefulness during EDA/model validation.",
        report_lines
    )
    log(
        "3. Fit any future train/test preprocessing (including imputation, "
        "scaling, encoding and resampling) using training data only.",
        report_lines
    )
    log(
        "4. This cleaning script does not balance the target classes and "
        "does not remove statistical outliers.",
        report_lines
    )

    # Write the complete console output to a text file.
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nFull run output saved to: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
