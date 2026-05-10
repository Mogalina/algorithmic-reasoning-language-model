"""
Process the dataset CSV into aggregated difficulty distributions per company.

Reads data/dataset.csv and writes data/dataset_aggregated.csv with columns:
    company, days_until, easycount, hardcount, mediumcount

Usage:
    python scripts/process_distributions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset_aggregated.csv"


def filter_csv_pandas(input_filepath):
    df = pd.read_csv(input_filepath)
    columns_to_keep = ["company", "difficulty", "preparation_days", "title"]
    return df[columns_to_keep]


def build_distributions(input_path=DATASET_PATH, output_path=OUTPUT_PATH):
    processed = filter_csv_pandas(input_path)

    processed["days_until"] = processed["preparation_days"]
    processed = processed.drop(columns=["preparation_days"])

    count = (
        processed.groupby(["company", "days_until", "difficulty"])["title"]
        .count()
        .reset_index()
    )

    data_on_columns = count.pivot(
        index=["company", "days_until"],
        columns="difficulty",
        values="title",
    ).reset_index()

    data_on_columns = data_on_columns.rename(columns={
        "EASY": "easycount",
        "MEDIUM": "mediumcount",
        "HARD": "hardcount",
    })

    cols_to_fill = ["easycount", "mediumcount", "hardcount"]
    data_on_columns[cols_to_fill] = data_on_columns[cols_to_fill].fillna(0).astype(int)
    data_on_columns.columns.name = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_on_columns.to_csv(output_path, index=False)
    print(f"Wrote distributions to {output_path} ({len(data_on_columns)} rows)")

    return data_on_columns



if __name__ == "__main__":
    build_distributions()
