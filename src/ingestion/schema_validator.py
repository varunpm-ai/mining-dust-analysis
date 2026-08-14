"""
Schema Validation Module for Ingested Datasets.

Validates column presence, dataset requirements, and schema compliance against configs/schema.yaml.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from src.utils.configuration import get_config, AppConfig
from src.utils.exceptions import SchemaValidationError
from src.utils.logger import setup_logger

logger = setup_logger("ingestion.schema_validator")


class SchemaValidator:
    """
    Validates ingested DataFrame structure and contents against system schema rules.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()

    def validate_schema(
        self,
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        job_id: str = "N/A",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Alias method for validate."""
        return self.validate(df, job_id=job_id)

    def validate(
        self,
        df: pd.DataFrame,
        job_id: str = "N/A",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate dataset against expected schema requirements.

        Args:
            df (pd.DataFrame): Ingested, header-normalized DataFrame.
            job_id (str): Pipeline execution job ID.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, validation_report)

        Raises:
            SchemaValidationError: If critical targets or mandatory features are missing.
        """
        extra_ctx = {"job_id": job_id}
        report: Dict[str, Any] = {
            "job_id": job_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "targets_found": [],
            "targets_missing": [],
            "legacy_features_found": [],
            "legacy_features_missing": [],
            "operational_features_found": [],
            "operational_features_missing": [],
            "validation_errors": [],
            "validation_warnings": [],
        }

        # 1. Target Column Check
        configured_targets = self.config.target_columns  # ["PM10", "PM2.5"]
        for target in configured_targets:
            if target in df.columns:
                report["targets_found"].append(target)
            else:
                report["targets_missing"].append(target)
                report["validation_errors"].append(
                    f"Target column '{target}' is missing from the uploaded dataset."
                )

        if report["targets_missing"]:
            err_msg = f"Missing required target column(s): {', '.join(report['targets_missing'])}"
            logger.error(err_msg, extra=extra_ctx)
            raise SchemaValidationError(
                message=err_msg,
                user_message=f"Dataset is missing required target columns: {', '.join(report['targets_missing'])}. Please ensure your file includes PM10 and PM2.5 concentrations.",
                details={"missing_targets": report["targets_missing"]},
            )

        # 2. Required Legacy / Environmental Features Check
        for feature_info in self.config.legacy_features:
            fname = feature_info["name"]
            is_req = feature_info.get("required", False)
            if fname in df.columns:
                report["legacy_features_found"].append(fname)
            else:
                report["legacy_features_missing"].append(fname)
                if is_req:
                    report["validation_errors"].append(f"Required environmental feature '{fname}' is missing.")

        # 3. Operational Features Check
        for feature_info in self.config.operational_features:
            fname = feature_info["name"]
            if fname in df.columns:
                report["operational_features_found"].append(fname)
            else:
                report["operational_features_missing"].append(fname)

        # Check critical required features
        missing_critical_req = [
            f["name"] for f in self.config.legacy_features if f.get("required", False) and f["name"] not in df.columns
        ]
        if missing_critical_req:
            err_msg = f"Missing mandatory feature(s): {', '.join(missing_critical_req)}"
            logger.error(err_msg, extra=extra_ctx)
            raise SchemaValidationError(
                message=err_msg,
                user_message=f"Dataset is missing mandatory variables: {', '.join(missing_critical_req)}. Please verify your file schema.",
                details={"missing_required_features": missing_critical_req},
            )

        # 4. Data Quality Warnings (Row count & NaNs)
        if len(df) < 10:
            report["validation_warnings"].append("Dataset contains fewer than 10 rows. Model training may be unreliable.")

        logger.info(
            f"Schema validation succeeded. Found targets: {report['targets_found']}, "
            f"Legacy features: {len(report['legacy_features_found'])}, "
            f"Operational features: {len(report['operational_features_found'])}",
            extra=extra_ctx,
        )

        is_valid = len(report["validation_errors"]) == 0
        return is_valid, report
