"""
Deterministic Header Normalization Module for Mine Dust Data Ingestion.

Handles:
- MultiIndex & multi-row Excel headers
- Merged-cell sub-headers
- 'Unnamed' column artifacts
- Duplicate column resolution
- Whitespace & special character cleaning
- Transformation logging
"""

import logging
import re
from typing import List, Tuple, Union, Dict, Any, Optional
import pandas as pd

from src.utils.exceptions import HeaderNormalizationError
from src.utils.logger import setup_logger

logger = setup_logger("ingestion.header_normalizer")


class HeaderNormalizer:
    """
    Normalizes and flattens column headers from CSV/Excel files deterministically.
    """

    def __init__(self, alias_map: Optional[Dict[str, str]] = None):
        """
        Args:
            alias_map (Dict[str, str], optional): Mapping from raw header aliases to standard names.
        """
        self.alias_map = alias_map or {
            "YEAR": "YEAR",
            "SEASON": "SEASON",
            "MONTH": "MONTH",
            "LOCATION": "LOCATION",
            "WEEK": "WEEK",
            "T": "Temperature",
            "TEMP": "Temperature",
            "TEMPERATURE": "Temperature",
            "RH": "Relative_humidity",
            "RELATIVE HUMIDITY": "Relative_humidity",
            "RELATIVE_HUMIDITY": "Relative_humidity",
            "RF-CUMULATIVE": "RF-CUMULATIVE",
            "RF_CUMULATIVE": "RF-CUMULATIVE",
            "RAINFALL": "RF-CUMULATIVE",
            "WS": "Wind_speed",
            "WIND SPEED": "Wind_speed",
            "WIND_SPEED": "Wind_speed",
            "PM10": "PM10",
            "PM2.5": "PM2.5",
            "ROAD WIDTH": "Road_width",
            "ROAD_WIDTH": "Road_width",
            "DISTANCE": "Distance",
            "PRODUCTION ROM": "Production_ROM",
            "PRODUCTION_ROM": "Production_ROM",
            "PRODUCTION OB": "Production_OB",
            "PRODUCTION_OB": "Production_OB",
            "PRODUCTION ROM+OB": "Production_ROM+OB",
            "PRODUCTION_ROM+OB": "Production_ROM+OB",
            "VEHICLE CAPACITY": "Vehicle_capacity",
            "VEHICLE_CAPACITY": "Vehicle_capacity",
        }

    def normalize(self, df: pd.DataFrame, job_id: str = "N/A") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Normalize DataFrame column headers in-place or returned copy.

        Args:
            df (pd.DataFrame): Raw ingested DataFrame.
            job_id (str): Pipeline execution job ID.

        Returns:
            Tuple[pd.DataFrame, List[Dict[str, Any]]]: (Normalized DataFrame, transformation logs)
        """
        extra_ctx = {"job_id": job_id}
        transformations: List[Dict[str, Any]] = []
        result_df = df.copy()

        # Step 1: Flatten MultiIndex columns if present
        if isinstance(result_df.columns, pd.MultiIndex):
            flattened_cols = []
            for col_tuple in result_df.columns.values:
                clean_parts = [
                    str(p).strip()
                    for p in col_tuple
                    if pd.notna(p) and not str(p).startswith("Unnamed:")
                ]
                flattened_name = " ".join(clean_parts) if clean_parts else "Unnamed_Column"
                flattened_cols.append(flattened_name)
            result_df.columns = flattened_cols
            transformations.append({
                "operation": "flatten_multiindex",
                "reason": "DataFrame contained multi-level MultiIndex headers.",
            })
            logger.info("Flattened MultiIndex column headers.", extra=extra_ctx)

        # Step 2: Clean individual column names
        raw_headers = list(result_df.columns)
        cleaned_headers = []

        for idx, header in enumerate(raw_headers):
            str_header = str(header).strip()

            # Remove 'Unnamed: X' or 'Unnamed: X_level_Y' artifacts
            if re.match(r"^Unnamed:\s*\d+", str_header, re.IGNORECASE):
                cleaned_headers.append(f"__UNNAMED_{idx}__")
                continue

            # Replace line breaks, multiple spaces with single space
            str_header = re.sub(r"[\r\n\t]+", " ", str_header)
            str_header = re.sub(r"\s+", " ", str_header).strip()

            # Remove pandas duplicate suffix e.g. "PM10.1" -> "PM10", preserving "PM2.5"
            upper_header = str_header.upper()
            if upper_header not in self.alias_map and re.search(r"\.\d+$", str_header):
                candidate = re.sub(r"\.\d+$", "", str_header)
                if candidate.upper() in self.alias_map:
                    str_header = candidate
                    upper_header = str_header.upper()

            # Check alias map against full header AND trailing sub-header tokens
            matched_std_header = None

            # 1. Direct match on full string
            if upper_header in self.alias_map:
                matched_std_header = self.alias_map[upper_header]
            else:
                # 2. Check tokens e.g. "METEOROLOGICAL T" -> trailing token "T"
                tokens = upper_header.split()
                for token in reversed(tokens):
                    clean_token = token if token == "PM2.5" else re.sub(r"\.\d+$", "", token)
                    if clean_token in self.alias_map:
                        matched_std_header = self.alias_map[clean_token]
                        break

            if matched_std_header:
                transformations.append({
                    "column_index": idx,
                    "original": str_header,
                    "normalized": matched_std_header,
                    "operation": "map_alias",
                    "reason": f"Mapped raw header '{str_header}' to standard name '{matched_std_header}'",
                })
                cleaned_headers.append(matched_std_header)
            else:
                cleaned_headers.append(str_header)

        result_df.columns = cleaned_headers

        # Step 3: Handle Unnamed artifact columns
        unnamed_cols = [c for c in result_df.columns if str(c).startswith("__UNNAMED_")]
        cols_to_drop = []
        for col in unnamed_cols:
            # If the column is completely empty, mark for dropping
            if result_df[col].isna().all() or (result_df[col].astype(str).str.strip() == "").all():
                cols_to_drop.append(col)
                transformations.append({
                    "column": col,
                    "operation": "drop_empty_unnamed_column",
                    "reason": "Dropped empty 'Unnamed' Excel artifact column.",
                })

        if cols_to_drop:
            result_df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped {len(cols_to_drop)} empty Unnamed Excel columns.", extra=extra_ctx)

        # Step 4: Resolve duplicate column names deterministically
        final_headers = []
        header_counts: Dict[str, int] = {}

        for col in result_df.columns:
            if col not in header_counts:
                header_counts[col] = 1
                final_headers.append(col)
            else:
                header_counts[col] += 1
                deduped_col = f"{col}_{header_counts[col]}"
                final_headers.append(deduped_col)
                transformations.append({
                    "original": col,
                    "normalized": deduped_col,
                    "operation": "deduplicate_column",
                    "reason": f"Resolved duplicate column name by appending _{header_counts[col]}.",
                })
                logger.warning(f"Duplicate column '{col}' detected. Renamed to '{deduped_col}'.", extra=extra_ctx)

        result_df.columns = final_headers

        # Step 5: Drop completely empty columns (all NaN) if DataFrame contains rows
        if len(result_df) > 0:
            all_nan_cols = [c for c in result_df.columns if result_df[c].isna().all()]
            if all_nan_cols:
                result_df.drop(columns=all_nan_cols, inplace=True)
                for c in all_nan_cols:
                    transformations.append({
                        "column": c,
                        "operation": "drop_all_nan_column",
                        "reason": f"Dropped column '{c}' because all values are missing (NaN).",
                    })
                logger.info(f"Dropped {len(all_nan_cols)} completely empty columns: {all_nan_cols}", extra=extra_ctx)

        return result_df, transformations
