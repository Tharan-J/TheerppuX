"""
Model Factory for instantiating Translation Models and Pipelines.
"""

from typing import Any, Dict, List, Optional
import logging
from src.translation.base import TranslationModel
from src.translation.baseline import BaselineTranslationModel
from src.translation.indictrans2 import IndicTrans2TranslationModel
from src.translation.legal_aware import LegalAwareTranslationModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "baseline": BaselineTranslationModel,
    "opus_mt": BaselineTranslationModel,
    "p1": BaselineTranslationModel,
    "indictrans2": IndicTrans2TranslationModel,
    "indictrans": IndicTrans2TranslationModel,
    "p2": IndicTrans2TranslationModel,
    "legal_aware": LegalAwareTranslationModel,
    "legal": LegalAwareTranslationModel,
    "p3": LegalAwareTranslationModel,
}


class ModelFactory:
    """Factory to create translation models and pipelines by identifier."""

    @staticmethod
    def create(name: str, config: Optional[Dict[str, Any]] = None) -> TranslationModel:
        """
        Instantiate a translation model by name.
        """
        canonical_name = name.strip().lower()
        model_cls = MODEL_REGISTRY.get(canonical_name)

        if model_cls is None:
            available = ", ".join(sorted(set(MODEL_REGISTRY.keys())))
            raise ValueError(
                f"Unknown model identifier '{name}'. Available model configurations are: {available}"
            )

        logger.info(f"Instantiating model configuration: '{canonical_name}' -> {model_cls.__name__}")
        return model_cls(config=config)

    @staticmethod
    def list_available_models() -> List[str]:
        """List distinct available model identifiers."""
        return ["baseline", "indictrans2", "legal_aware"]
