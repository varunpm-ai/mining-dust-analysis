"""
Synthetic Dataset Fixtures Factory for End-to-End Pipeline & Integration Testing.
Provides generator functions for all edge-case, real-world, and invalid test scenarios.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def create_synthetic_csv(path: Path, n_samples: int = 100) -> Path:
    """Generate clean synthetic CSV dataset matching legacy 11-variable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")
    locations = ["Mining Zone A", "Mining Zone B", "Processing Plant"]
    seasons = ["Monsoon", "Summer", "Winter"]

    df = pd.DataFrame({
        "YEAR": dates.year,
        "SEASON": [np.random.choice(seasons) for _ in range(n_samples)],
        "MONTH": dates.month_name(),
        "LOCATION": [np.random.choice(locations) for _ in range(n_samples)],
        "WEEK": dates.isocalendar().week.astype(int),
        "Temperature": np.random.uniform(15.0, 40.0, n_samples).round(1),
        "Relative_humidity": np.random.uniform(30.0, 85.0, n_samples).round(1),
        "RF-CUMULATIVE": np.random.exponential(10.0, n_samples).round(1),
        "Wind_speed": np.random.uniform(2.0, 14.0, n_samples).round(1),
        "PM10": np.random.uniform(50.0, 200.0, n_samples).round(1),
        "PM2.5": np.random.uniform(25.0, 100.0, n_samples).round(1),
    })
    df.to_csv(path, index=False)
    return path


def create_synthetic_xlsx(path: Path, n_samples: int = 100) -> Path:
    """Generate clean synthetic Excel (.xlsx) dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".temp.csv")
    create_synthetic_csv(csv_path, n_samples=n_samples)
    df = pd.read_csv(csv_path)
    df.to_excel(path, index=False, engine="openpyxl")
    if csv_path.exists():
        csv_path.unlink()
    return path


def create_synthetic_merged_header_xlsx(path: Path) -> Path:
    """Generate Excel file simulating 2-row merged header client operational layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_row1 = ["Time Context", "", "", "Operational Data", "", "", "Air Quality Targets", ""]
    header_row2 = ["YEAR", "SEASON", "LOCATION", "Road_width", "Distance", "Production_ROM", "PM10", "PM2.5"]

    data = [
        [2024, "Summer", "Zone A", 12.5, 450.0, 15000.0, 110.2, 55.4],
        [2024, "Summer", "Zone B", 10.0, 300.0, 12000.0, 95.8, 42.1],
        [2024, "Winter", "Zone A", 12.5, 450.0, 18000.0, 145.0, 78.3],
    ]

    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(zip(header_row1, header_row2)))
    df.to_excel(path, engine="openpyxl")
    return path


def create_invalid_schema_csv(path: Path) -> Path:
    """Generate CSV with unknown schema lacking all required operational/target variables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "Unrelated_ID": [1, 2, 3],
        "Arbitrary_Text": ["alpha", "beta", "gamma"],
        "Value_Column": [10.0, 20.0, 30.0],
    })
    df.to_csv(path, index=False)
    return path


def create_missing_pm10_csv(path: Path) -> Path:
    """Generate CSV containing PM2.5 target but missing PM10 column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p = create_synthetic_csv(path)
    df = pd.read_csv(p)
    df.drop(columns=["PM10"], inplace=True)
    df.to_csv(path, index=False)
    return path


def create_missing_pm2_5_csv(path: Path) -> Path:
    """Generate CSV containing PM10 target but missing PM2.5 column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p = create_synthetic_csv(path)
    df = pd.read_csv(p)
    df.drop(columns=["PM2.5"], inplace=True)
    df.to_csv(path, index=False)
    return path


def create_missing_weather_csv(path: Path) -> Path:
    """Generate CSV missing meteorological features (Temperature, Relative_humidity, Wind_speed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p = create_synthetic_csv(path)
    df = pd.read_csv(p)
    df.drop(columns=["Temperature", "Relative_humidity", "Wind_speed"], inplace=True)
    df.to_csv(path, index=False)
    return path


def create_duplicate_columns_csv(path: Path) -> Path:
    """Generate CSV with duplicate column names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_content = "YEAR,SEASON,PM10,PM10,Temperature\n2024,Summer,105.2,105.2,32.4\n2024,Winter,140.0,140.0,22.1\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_content)
    return path


def create_invalid_numeric_values_csv(path: Path) -> Path:
    """Generate CSV containing comma decimals ("31,5"), non-numeric strings ("N/A", "INVALID"), and missing values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "YEAR": [2024, 2024, 2024, 2024],
        "SEASON": ["Summer", "Winter", "Monsoon", "Summer"],
        "MONTH": ["July", "August", "September", "October"],
        "LOCATION": ["Zone A", "Zone B", "Zone A", "Zone C"],
        "Temperature": ["31,5", "INVALID", "28.4", None],
        "Relative_humidity": ["60.0", "N/A", "75,2", "50.0"],
        "Wind_speed": ["5.5", "6.2", "4,8", "3.1"],
        "PM10": ["115,4", "130.0", "95.2", "140,8"],
        "PM2.5": ["55,2", "62.0", "48.1", "70,0"],
    })
    df.to_csv(path, index=False)
    return path


def create_empty_dataset_csv(path: Path) -> Path:
    """Generate completely empty (0-byte) CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")
    return path
