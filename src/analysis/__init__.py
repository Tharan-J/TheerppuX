"""
Error Analysis, Consistency Checking, and Reporting Module for TheerppuX.
"""

from src.analysis.error_analysis import LegalErrorClassifier, ClassifiedError, ErrorTaxonomy
from src.analysis.consistency import CrossLanguageConsistencyChecker
from src.analysis.report import ReportGenerator

__all__ = [
    "LegalErrorClassifier",
    "ClassifiedError",
    "ErrorTaxonomy",
    "CrossLanguageConsistencyChecker",
    "ReportGenerator",
]
