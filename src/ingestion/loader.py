"""
Robust Data Ingestion Loader for CSV and Excel Formats.

Supports:
- CSV, XLSX, XLS file formats
- File path, string stream, and BytesIO buffer inputs
- Automatic multi-row header detection
- String-numeric coersion and decimal sanitization
- Complete transformation auditing
"""

import io
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.ingestion.header_normalizer import HeaderNormalizer
from src.utils.exceptions import IngestionError
from src.utils.logger import setup_logger

logger = setup_logger("ingestion.loader")


class DataLoader:
    """
    Production-grade data ingestion loader for CSV and Excel datasets.
    """

    def __init__(self, normalizer: Optional[HeaderNormalizer] = None):
        self.normalizer = normalizer or HeaderNormalizer()

    def load_dataset(
        self,
        file_input: Union[str, Path, io.BytesIO],
        file_type: Optional[str] = None,
        job_id: str = "N/A",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Alias for load method."""
        return self.load(file_input=file_input, file_type=file_type, job_id=job_id)

    def load(
        self,
        file_input: Union[str, Path, io.BytesIO],
        file_type: Optional[str] = None,
        job_id: str = "N/A",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load dataset from CSV or Excel file or buffer.

        Args:
            file_input: File path string, Path object, or BytesIO buffer.
            file_type (str, optional): 'csv' or 'excel'/'xlsx'. Auto-detected from path extension if omitted.
            job_id (str): Pipeline execution job ID.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Loaded & normalized DataFrame, Ingestion Audit Manifest)
        """
        extra_ctx = {"job_id": job_id}
        audit_log: Dict[str, Any] = {
            "job_id": job_id,
            "format": None,
            "raw_rows": 0,
            "raw_columns": 0,
            "final_rows": 0,
            "final_columns": 0,
            "transformations": [],
        }

        # 1. Determine file format
        determined_type = self._determine_file_type(file_input, file_type)
        audit_log["format"] = determined_type
        logger.info(f"Starting dataset ingestion. Detected format: {determined_type}", extra=extra_ctx)

        # 2. Read raw DataFrame
        try:
            if determined_type == "csv":
                raw_df = self._read_csv(file_input)
            elif determined_type in ("excel", "xlsx", "xls"):
                raw_df = self._read_excel(file_input)
            else:
                raise IngestionError(
                    message=f"Unsupported file format: {determined_type}",
                    user_message="Unsupported file format. Please upload a CSV or Excel (.xlsx) file.",
                )
        except Exception as e:
            if isinstance(e, IngestionError):
                raise e
            logger.error(f"Failed to read dataset: {e}", exc_info=True, extra=extra_ctx)
            raise IngestionError(
                message=f"File reading exception: {str(e)}",
                user_message="Unable to parse the dataset file. Please verify file integrity and try again.",
                details={"error": str(e)},
            )

        audit_log["raw_rows"] = len(raw_df)
        audit_log["raw_columns"] = len(raw_df.columns)
        logger.info(f"Loaded raw DataFrame with shape: {raw_df.shape}", extra=extra_ctx)

        # 3. Header Normalization
        norm_df, header_transformations = self.normalizer.normalize(raw_df, job_id=job_id)
        audit_log["transformations"].extend(header_transformations)

        # 4. Numerical String Parsing & Sanitization (Pre-cleaner pass)
        sanitized_df, val_transformations = self._sanitize_values(norm_df, job_id=job_id)
        audit_log["transformations"].extend(val_transformations)

        audit_log["final_rows"] = len(sanitized_df)
        audit_log["final_columns"] = len(sanitized_df.columns)

        logger.info(
            f"Completed dataset ingestion. Final shape: {sanitized_df.shape}, "
            f"Total recorded transformations: {len(audit_log['transformations'])}",
            extra=extra_ctx,
        )

        return sanitized_df, audit_log

    def _determine_file_type(
        self,
        file_input: Union[str, Path, io.BytesIO],
        file_type: Optional[str],
    ) -> str:
        """Infer file type from input parameter or file extension."""
        if file_type:
            ft = file_type.lower().strip(".")
            if ft in ("csv", "excel", "xlsx", "xls"):
                return "csv" if ft == "csv" else "excel"

        if isinstance(file_input, (str, Path)):
            ext = Path(file_input).suffix.lower()
            if ext == ".csv":
                return "csv"
            elif ext in (".xlsx", ".xls"):
                return "excel"

        # Default fallback for buffer if unprovided
        return "csv"

    def _read_csv(self, file_input: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
        """Read CSV dataset with header auto-detection."""
        return pd.read_csv(file_input, low_memory=False)

    def _read_excel(self, file_input: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
        """
        Read Excel dataset supporting 1-row, 2-row, and MultiIndex headers.
        """
        # Read header rows to inspect multi-level headers
        if isinstance(file_input, io.BytesIO):
            file_input.seek(0)
            excel_file = pd.ExcelFile(file_input, engine="openpyxl")
            file_input.seek(0)
        else:
            excel_file = pd.ExcelFile(file_input, engine="openpyxl")

        df_head = excel_file.parse(sheet_name=0, nrows=5, header=None)

        # Check if first 2 rows are text/headers (multi-row header pattern)
        header_rows = 1
        if len(df_head) >= 2:
            row0_str_ratio = df_head.iloc[0].apply(lambda x: isinstance(x, str)).mean()
            row1_str_ratio = df_head.iloc[1].apply(lambda x: isinstance(x, str)).mean()
            # If both row 0 and row 1 are predominantly strings/text headers
            if row0_str_ratio > 0.4 and row1_str_ratio > 0.4:
                header_rows = [0, 1]

        if isinstance(file_input, io.BytesIO):
            file_input.seek(0)

        df = pd.read_excel(
            file_input,
            header=header_rows,
            engine="openpyxl",
        )
        return df

    def _sanitize_values(
        self,
        df: pd.DataFrame,
        job_id: str = "N/A",
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Sanitize string values representing numeric values, stripping whitespace,
        and handling malformed decimal representations safely.
        """
        extra_ctx = {"job_id": job_id}
        transformations: List[Dict[str, Any]] = []
        sanitized = df.copy()

        for col in sanitized.columns:
            # Strip whitespace from string columns
            if sanitized[col].dtype == object or isinstance(sanitized[col].dtype, pd.StringDtype):
                str_series = sanitized[col].astype(str).str.strip()

                # Attempt numeric conversion for columns containing numeric strings
                # Replace comma decimals e.g., "30,5" -> "30.5"
                comma_fixed = str_series.str.replace(r"^(\d+),(\d+)$", r"\1.\2", regex=True)

                # Check if column values can be safely converted to float/int
                numeric_converted = pd.to_numeric(comma_fixed, errors="coerce")
                non_null_orig_count = str_series.replace(["nan", "NaN", "None", ""], np.nan).dropna().count()
                non_null_num_count = numeric_converted.dropna().count()

                # If majority (>80%) of non-null values convert cleanly to numbers
                if non_null_orig_count > 0 and (non_null_num_count / non_null_orig_count) >= 0.8:
                    coerced_nans = non_null_orig_count - non_null_num_count
                    sanitized[col] = numeric_converted
                    transformations.append({
                        "column": col,
                        "operation": "convert_numeric_string",
                        "affected_rows": int(non_null_num_count),
                        "coerced_invalid_nans": int(coerced_nans),
                        "reason": f"Converted string column '{col}' to numeric representation.",
                    })
                    if coerced_nans > 0:
                        logger.warning(
                            f"Column '{col}': Coerced {coerced_nans} non-numeric string values to NaN.",
                            extra=extra_ctx,
                        )

        return sanitized, transformations


DatasetLoader = DataLoader
