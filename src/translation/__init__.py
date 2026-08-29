"""
Translation Module for TheerppuX.
Supports multiple translation configurations: Baseline (OPUS-MT), IndicTrans2, and Legal-Aware Pipeline.
"""

from src.translation.base import TranslationModel, TranslationResult
from src.translation.baseline import BaselineTranslationModel
from src.translation.indictrans2 import IndicTrans2TranslationModel
from src.translation.legal_aware import LegalAwareTranslationModel
from src.translation.factory import ModelFactory

__all__ = [
    "TranslationModel",
    "TranslationResult",
    "BaselineTranslationModel",
    "IndicTrans2TranslationModel",
    "LegalAwareTranslationModel",
    "ModelFactory",
]
