"""
Unit Tests for Header Normalizer.
"""

import pandas as pd
import pytest

from src.ingestion.header_normalizer import HeaderNormalizer


def test_normalize_simple_header_aliases():
    """Verify mapping raw aliases (T, RH, WS) to standard names."""
    normalizer = HeaderNormalizer()
    df = pd.DataFrame(columns=["YEAR", "SEASON", "T", "RH", "RF-CUMULATIVE", "WS", "PM10", "PM2.5"])

    norm_df, transforms = normalizer.normalize(df)
    cols = list(norm_df.columns)

    assert "Temperature" in cols
    assert "Relative_humidity" in cols
    assert "Wind_speed" in cols
    assert "RF-CUMULATIVE" in cols
    assert len(transforms) > 0


def test_normalize_multiindex_columns():
    """Verify flattening MultiIndex columns into single string headers."""
    normalizer = HeaderNormalizer()
    multi_cols = pd.MultiIndex.from_tuples([
        ("METEOROLOGICAL DATA", "Temperature"),
        ("METEOROLOGICAL DATA", "Relative_humidity"),
        ("DUST METRICS", "PM10"),
    ])
    df = pd.DataFrame(columns=multi_cols)

    norm_df, transforms = normalizer.normalize(df)
    cols = list(norm_df.columns)

    assert "METEOROLOGICAL DATA Temperature" in cols or "Temperature" in cols
    assert "PM10" in cols or "DUST METRICS PM10" in cols
    assert any(t["operation"] == "flatten_multiindex" for t in transforms)


def test_resolve_duplicate_column_names():
    """Verify duplicate column names append incrementing integer suffixes."""
    normalizer = HeaderNormalizer()
    df = pd.DataFrame(columns=["PM10", "PM10", "PM10", "PM2.5"])

    norm_df, transforms = normalizer.normalize(df)
    cols = list(norm_df.columns)

    assert cols == ["PM10", "PM10_2", "PM10_3", "PM2.5"]
    assert any(t["operation"] == "deduplicate_column" for t in transforms)


def test_drop_unnamed_empty_columns():
    """Verify empty Unnamed columns are identified and removed."""
    normalizer = HeaderNormalizer()
    df = pd.DataFrame({
        "LOCATION": ["HAUL ROAD", "MINE FACE"],
        "__UNNAMED_1__": [None, None],
        "PM10": [55.0, 60.0],
    })

    norm_df, transforms = normalizer.normalize(df)
    assert "__UNNAMED_1__" not in norm_df.columns
    assert len(norm_df.columns) == 2
