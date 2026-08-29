"""
Abstract Base Translation Model and Data Structures for TheerppuX.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Encapsulates translated chunks with latency and resource metadata."""
    model_name: str
    pipeline_type: str
    source_language: str
    target_language: str
    translations: List[str]
    chunk_ids: List[str]
    latency_seconds: float
    chars_per_second: float
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pipeline_type": self.pipeline_type,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "latency_seconds": round(self.latency_seconds, 3),
            "chars_per_second": round(self.chars_per_second, 1),
            "device": self.device,
            "translations": self.translations,
            "chunk_ids": self.chunk_ids,
            "metadata": self.metadata,
        }


class TranslationModel(ABC):
    """
    Abstract Base Class for all Translation Models and Pipelines.
    Subclasses must implement translate() and get_model_info().
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._device = None
        self._torch_device = None
        self._init_device()

    def _init_device(self):
        req_dev = self.config.get("device", "auto")
        if req_dev == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = req_dev
        self._torch_device = torch.device(self._device)

    @property
    def device(self) -> str:
        return self._device

    @property
    def torch_device(self) -> torch.device:
        return self._torch_device

    @abstractmethod
    def translate(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "ta",
    ) -> List[str]:
        """
        Translate a list of texts from source_lang to target_lang.
        Must return a list of translated strings matching input length.
        """
        raise NotImplementedError("Subclasses must implement translate()")

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return structured model metadata: name, architecture, parameters, device, precision.
        """
        raise NotImplementedError("Subclasses must implement get_model_info()")

    def translate_chunks(
        self,
        chunks: List[Any],  # ProcessedChunk or dict
        source_lang: str = "en",
        target_lang: str = "ta",
    ) -> TranslationResult:
        """
        Translate a list of ProcessedChunk objects, tracking latency and profiling data.
        """
        texts = [c.text if hasattr(c, "text") else c["text"] for c in chunks]
        chunk_ids = [c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]

        total_chars = sum(len(t) for t in texts)
        start_time = time.time()

        translations = self.translate(texts, source_lang=source_lang, target_lang=target_lang)

        latency = time.time() - start_time
        cps = total_chars / max(latency, 1e-6)

        info = self.get_model_info()

        return TranslationResult(
            model_name=info.get("name", "Unknown"),
            pipeline_type=info.get("pipeline_type", "generic"),
            source_language=source_lang,
            target_language=target_lang,
            translations=translations,
            chunk_ids=chunk_ids,
            latency_seconds=latency,
            chars_per_second=cps,
            device=self.device,
            metadata={
                "total_chunks": len(texts),
                "total_source_chars": total_chars,
                "model_info": info,
            },
        )
