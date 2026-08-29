"""
Text Preprocessing, Legal-Aware Segmentation, and Language Detection Module.
"""

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.segmenter import TextSegmenter
from src.preprocessing.language_detector import LanguageDetector

__all__ = ["TextCleaner", "TextSegmenter", "LanguageDetector"]
