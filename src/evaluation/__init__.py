"""
Comprehensive Evaluation Suite for Multilingual Legal Translation.
Provides generic NMT metrics (BLEU, chrF++, ROUGE, BERTScore) alongside
domain-specific metrics (Legal Entity F1, Number Accuracy, Legal Terminology Accuracy).
"""

from src.evaluation.bleu import BLEUMetric
from src.evaluation.chrf import ChrFMetric
from src.evaluation.rouge import ROUGEMetric
from src.evaluation.bertscore import BERTScoreMetric
from src.evaluation.entity_metrics import EntityPreservationMetric
from src.evaluation.terminology import LegalTerminologyMetric
from src.evaluation.evaluator import Evaluator, EvaluationReport

__all__ = [
    "BLEUMetric",
    "ChrFMetric",
    "ROUGEMetric",
    "BERTScoreMetric",
    "EntityPreservationMetric",
    "LegalTerminologyMetric",
    "Evaluator",
    "EvaluationReport",
]
