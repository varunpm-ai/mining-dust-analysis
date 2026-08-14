"""
Statistical Analysis Engine Package for Mine Dust Platform.
"""

import sys
import pathlib
import importlib.util

# Ensure Python standard library 'statistics' is bound in sys.modules so third-party libraries (e.g. seaborn) import correctly
if "statistics" not in sys.modules or not hasattr(sys.modules["statistics"], "NormalDist"):
    try:
        _stdlib_stat_path = pathlib.Path(sys.base_prefix) / "Lib" / "statistics.py"
        if _stdlib_stat_path.exists():
            _spec = importlib.util.spec_from_file_location("statistics", _stdlib_stat_path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            sys.modules["statistics"] = _mod
    except Exception:
        pass

from src.statistics.correlation import CorrelationAnalyzer
from src.statistics.hypothesis import HypothesisTester
from src.statistics.diagnostics import VarianceInflationFactorCalculator, RegressionDiagnostics
from src.statistics.regression import StatisticalRegressionAnalyzer

__all__ = [
    "CorrelationAnalyzer",
    "HypothesisTester",
    "VarianceInflationFactorCalculator",
    "RegressionDiagnostics",
    "StatisticalRegressionAnalyzer",
]
