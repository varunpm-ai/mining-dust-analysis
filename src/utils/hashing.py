"""
Hashing Utilities for Dataset and Artifact Integrity Verification.
"""

import hashlib
from pathlib import Path
from typing import Union
import pandas as pd


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of a pandas DataFrame's content."""
    hashed_values = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(hashed_values.tobytes()).hexdigest()


# Aliases for API compatibility
calculate_file_hash = compute_file_hash
calculate_dataframe_hash = compute_dataframe_hash
