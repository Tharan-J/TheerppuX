"""
Configuration Management for TheerppuX Legal Translation Pipeline.
Handles YAML loading, language validation, device auto-detection, and typed config access.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
import torch

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


SUPPORTED_TARGET_LANGUAGES = {
    "ta": "Tamil",
    "ml": "Malayalam",
}

LANGUAGE_CODE_MAP = {
    "ta": {
        "name": "Tamil",
        "indictrans2": "tam_Taml",
        "flores": "tam_Taml",
        "opus_prefix": ">>tam<< ",
    },
    "ml": {
        "name": "Malayalam",
        "indictrans2": "mal_Mlym",
        "flores": "mal_Mlym",
        "opus_prefix": "",
    },
}


def get_device_info(requested_device: str = "auto", fp16: bool = True) -> Dict[str, Any]:
    """
    Detect available computing hardware (CUDA, MPS, or CPU) and return device details.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
            device_name = torch.cuda.get_device_name(0)
            precision = "float16" if fp16 else "float32"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            device_name = "Apple Silicon MPS"
            precision = "float32"  # MPS often prefers float32
        else:
            device_str = "cpu"
            device_name = "CPU"
            precision = "float32"
    else:
        device_str = requested_device
        if device_str == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            precision = "float16" if fp16 else "float32"
        else:
            device_name = device_str.upper()
            precision = "float32"

    return {
        "device": device_str,
        "device_name": device_name,
        "precision": precision,
        "torch_device": torch.device(device_str),
        "use_fp16": precision == "float16" and device_str == "cuda",
    }


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class PipelineConfig:
    """Central configuration object for the legal translation pipeline."""

    def __init__(self, config_dict: Dict[str, Any], target_lang: str):
        if target_lang not in SUPPORTED_TARGET_LANGUAGES:
            supported = ", ".join(f"'{k}' ({v})" for k, v in SUPPORTED_TARGET_LANGUAGES.items())
            raise ValueError(
                f"Unsupported target language '{target_lang}'. "
                f"Stage 1 supports: {supported}."
            )

        self._raw = config_dict
        self.target_lang = target_lang
        self.target_lang_name = SUPPORTED_TARGET_LANGUAGES[target_lang]
        self.project_name = config_dict.get("project_name", "TheerppuX-Stage1")
        self.version = config_dict.get("version", "1.0.0")
        self.seed = config_dict.get("seed", 42)

        # Device info
        dev_cfg = config_dict.get("device", "auto")
        fp16_cfg = config_dict.get("fp16", True)
        self.device_info = get_device_info(dev_cfg, fp16_cfg)

        # Language codes
        self.lang_codes = LANGUAGE_CODE_MAP[target_lang]
        self.indictrans2_code = self.lang_codes["indictrans2"]
        self.flores_code = self.lang_codes["flores"]

        # Paths
        paths = config_dict.get("paths", {})
        self.data_dir = Path(paths.get("data_dir", "data"))
        self.raw_dir = Path(paths.get("raw_dir", "data/raw"))
        self.processed_dir = Path(paths.get("processed_dir", "data/processed"))
        self.references_dir = Path(paths.get("references_dir", "data/references"))
        self.outputs_dir = Path(paths.get("outputs_dir", "data/outputs"))
        self.models_dir = Path(paths.get("models_dir", "models"))

        # Sub-configs
        self.document = config_dict.get("document", {})
        self.preprocessing = config_dict.get("preprocessing", {})
        self.models = config_dict.get("models", {})
        self.evaluation = config_dict.get("evaluation", {})
        self.legal_terms = config_dict.get("legal_terms", {})
        self.human_evaluation = config_dict.get("human_evaluation", {})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "target_language": self.target_lang,
            "target_language_name": self.target_lang_name,
            "project_name": self.project_name,
            "version": self.version,
            "seed": self.seed,
            "device": self.device_info["device"],
            "device_name": self.device_info["device_name"],
            "precision": self.device_info["precision"],
            "indictrans2_code": self.indictrans2_code,
            "document": self.document,
            "preprocessing": self.preprocessing,
            "models": self.models,
            "evaluation": self.evaluation,
        }


def load_config(
    target_lang: str,
    config_dir: Optional[Path | str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """
    Load default YAML, merge target language YAML, apply overrides, and return PipelineConfig.
    """
    if target_lang not in SUPPORTED_TARGET_LANGUAGES:
        supported = ", ".join(f"'{k}' ({v})" for k, v in SUPPORTED_TARGET_LANGUAGES.items())
        raise ValueError(
            f"Unsupported target language '{target_lang}'. "
            f"Stage 1 supports: {supported}."
        )

    if config_dir is None:
        # Resolve config dir relative to workspace / repository root
        base_dir = Path(__file__).resolve().parent.parent
        config_dir = base_dir / "configs"
    else:
        config_dir = Path(config_dir)

    # 1. Load default.yaml
    default_path = config_dir / "default.yaml"
    config_data: Dict[str, Any] = {}
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # 2. Load language-specific YAML (tamil.yaml or malayalam.yaml)
    lang_file_name = "tamil.yaml" if target_lang == "ta" else "malayalam.yaml"
    lang_path = config_dir / lang_file_name
    if lang_path.exists():
        with open(lang_path, "r", encoding="utf-8") as f:
            lang_data = yaml.safe_load(f) or {}
            config_data = deep_merge(config_data, lang_data)

    # 3. Apply runtime overrides
    if overrides:
        config_data = deep_merge(config_data, overrides)

    return PipelineConfig(config_data, target_lang=target_lang)
