"""
OCR Extractor supporting Indian Languages (English, Tamil, Malayalam).
Provides primary PaddleOCR integration with fallback heuristics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
from src.document.models import PageContent, TextBlock

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Extracts text from scanned document images and PDF pages.
    Preserves page numbers, block coordinates, and detects language scripts.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_angle_cls: bool = True,
    ):
        self.languages = languages or ["en", "ta", "ml"]
        self.use_angle_cls = use_angle_cls
        self._paddle_ocr = None
        self._initialized = False

    def _init_engine(self, lang: str = "en") -> bool:
        """Lazy load PaddleOCR engine."""
        if self._initialized and self._paddle_ocr is not None:
            return True
        try:
            from paddleocr import PaddleOCR
            # Map language code to paddleocr language code
            paddle_lang = "ta" if "ta" in lang else ("ml" if "ml" in lang else "en")
            self._paddle_ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=paddle_lang)
            self._initialized = True
            logger.info(f"PaddleOCR initialized for language: {paddle_lang}")
            return True
        except ImportError:
            logger.warning(
                "PaddleOCR is not installed. To enable OCR for scanned PDFs, "
                "install paddleocr via `pip install paddleocr paddlepaddle`."
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            return False

    def extract_from_image(
        self,
        image_path: Union[str, Path],
        page_number: int = 1,
        lang: str = "en",
    ) -> PageContent:
        """Extract text and bounding blocks from a single page image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        if not self._init_engine(lang):
            return PageContent(
                page_number=page_number,
                text="",
                blocks=[],
                is_ocr=True,
                metadata={"ocr_engine": "none", "error": "PaddleOCR unavailable"},
            )

        try:
            result = self._paddle_ocr.ocr(str(image_path), cls=self.use_angle_cls)
            blocks: List[TextBlock] = []
            page_text_lines: List[str] = []

            if result and result[0]:
                for idx, line in enumerate(result[0]):
                    box, (text, score) = line
                    # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    x0 = min(pt[0] for pt in box)
                    y0 = min(pt[1] for pt in box)
                    x1 = max(pt[0] for pt in box)
                    y1 = max(pt[1] for pt in box)

                    clean_line = text.strip()
                    if clean_line:
                        page_text_lines.append(clean_line)
                        blocks.append(
                            TextBlock(
                                block_id=idx + 1,
                                text=clean_line,
                                bbox=[float(x0), float(y0), float(x1), float(y1)],
                                block_type="text",
                            )
                        )

            full_page_text = "\n".join(page_text_lines)
            return PageContent(
                page_number=page_number,
                text=full_page_text,
                blocks=blocks,
                is_ocr=True,
                metadata={"ocr_engine": "paddleocr", "confidence_available": True},
            )

        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return PageContent(
                page_number=page_number,
                text="",
                blocks=[],
                is_ocr=True,
                metadata={"ocr_engine": "paddleocr", "error": str(e)},
            )
