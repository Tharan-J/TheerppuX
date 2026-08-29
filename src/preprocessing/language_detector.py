"""
Language Detector for Input Validation and Script Analysis.
Validates that input documents are in English and warns on mixed-language content.
"""

from typing import Any, Dict, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects language and script distribution in legal documents.
    """

    def __init__(self):
        self._langdetect_available = True
        try:
            import langdetect
            from langdetect import DetectorFactory
            # Enforce deterministic results
            DetectorFactory.seed = 0
        except ImportError:
            self._langdetect_available = False
            logger.warning("langdetect library not installed. Using script-based heuristic fallback.")

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect primary language and script characteristics of the text.
        """
        if not text or not text.strip():
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_english": False,
                "script_distribution": {},
            }

        # 1. Analyze script distribution via Unicode ranges
        script_dist = self._analyze_scripts(text)

        # 2. Statistical language detection
        detected_lang = "en"
        confidence = 1.0

        if self._langdetect_available and len(text.strip()) > 10:
            try:
                import langdetect
                probs = langdetect.detect_langs(text)
                if probs:
                    top = probs[0]
                    detected_lang = top.lang
                    confidence = top.prob
            except Exception as e:
                logger.debug(f"langdetect error on text slice: {e}")
                # Fallback to Latin script heuristic
                if script_dist.get("Latin", 0) > 0.6:
                    detected_lang = "en"
                    confidence = 0.8
        else:
            if script_dist.get("Latin", 0) > 0.6:
                detected_lang = "en"
                confidence = 0.85

        is_english = detected_lang == "en" and script_dist.get("Latin", 0) > 0.5

        return {
            "language": detected_lang,
            "confidence": float(confidence),
            "is_english": is_english,
            "script_distribution": script_dist,
        }

    def _analyze_scripts(self, text: str) -> Dict[str, float]:
        """Compute proportion of characters from different scripts."""
        total_alpha = 0
        counts = {
            "Latin": 0,
            "Tamil": 0,
            "Malayalam": 0,
            "Devanagari": 0,
            "Other": 0,
        }

        for char in text:
            if not char.isalpha():
                continue
            total_alpha += 1
            code = ord(char)
            # Unicode blocks
            if 0x0041 <= code <= 0x007A or 0x0061 <= code <= 0x007A or 0x00C0 <= code <= 0x024F:
                counts["Latin"] += 1
            elif 0x0B80 <= code <= 0x0BFF:
                counts["Tamil"] += 1
            elif 0x0D00 <= code <= 0x0D7F:
                counts["Malayalam"] += 1
            elif 0x0900 <= code <= 0x097F:
                counts["Devanagari"] += 1
            else:
                counts["Other"] += 1

        if total_alpha == 0:
            return {}

        return {k: round(v / total_alpha, 4) for k, v in counts.items() if v > 0}
