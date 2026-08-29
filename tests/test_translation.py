"""
Tests for Translation Interface, Model Factory, and Language Validation.
"""

import pytest
from src.config import load_config
from src.translation.factory import ModelFactory
from src.translation.base import TranslationModel


def test_language_validation_supported():
    config_ta = load_config("ta")
    assert config_ta.target_lang == "ta"
    assert config_ta.target_lang_name == "Tamil"

    config_ml = load_config("ml")
    assert config_ml.target_lang == "ml"
    assert config_ml.target_lang_name == "Malayalam"


def test_language_validation_unsupported():
    with pytest.raises(ValueError) as excinfo:
        load_config("hi")
    assert "Unsupported target language 'hi'" in str(excinfo.value)
    assert "Stage 1 supports" in str(excinfo.value)


def test_model_factory_registry():
    available = ModelFactory.list_available_models()
    assert "baseline" in available
    assert "indictrans2" in available
    assert "legal_aware" in available


def test_model_factory_invalid():
    with pytest.raises(ValueError) as excinfo:
        ModelFactory.create("unknown_model_xyz")
    assert "Unknown model identifier" in str(excinfo.value)
