"""
PDF Extractor using pdfplumber with automatic scanned PDF detection and page preservation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import pdfplumber
from src.document.models import Document, PageContent, TextBlock
from src.document.ocr import OCRExtractor

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts text from PDF documents preserving page structure and layout blocks.
    Automatically detects scanned documents and falls back to OCR.
    """

    def __init__(
        self,
        scanned_threshold_chars: int = 50,
        ocr_enabled: bool = True,
        ocr_languages: Optional[List[str]] = None,
    ):
        self.scanned_threshold_chars = scanned_threshold_chars
        self.ocr_enabled = ocr_enabled
        self.ocr_extractor = OCRExtractor(languages=ocr_languages) if ocr_enabled else None

    def extract(self, file_path: Union[str, Path], document_id: Optional[str] = None) -> Document:
        """
        Extract text from a PDF file preserving per-page breakdown.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF document not found: {file_path}")

        doc_id = document_id or file_path.stem
        pages: List[PageContent] = []
        is_scanned_doc = False

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""
                clean_text = page_text.strip()

                # Check if page is scanned (very low text density)
                if len(clean_text) < self.scanned_threshold_chars:
                    is_scanned_doc = True
                    logger.info(
                        f"Page {page_num} in '{file_path.name}' has low text yield "
                        f"({len(clean_text)} chars). Checking for OCR fallback."
                    )
                    # Attempt OCR if enabled
                    if self.ocr_enabled and self.ocr_extractor:
                        try:
                            # Render page to image
                            page_image = page.to_image(resolution=300).original
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                                tmp_path = Path(tmp_img.name)
                                page_image.save(tmp_path)

                            ocr_content = self.ocr_extractor.extract_from_image(
                                image_path=tmp_path,
                                page_number=page_num,
                            )
                            tmp_path.unlink(missing_ok=True)

                            if len(ocr_content.text.strip()) > len(clean_text):
                                pages.append(ocr_content)
                                continue
                        except Exception as e:
                            logger.warning(f"OCR fallback failed on page {page_num}: {e}")

                # Extract layout blocks for digital PDF
                blocks: List[TextBlock] = []
                try:
                    # Extract words with bounding boxes grouped into lines
                    words = page.extract_words(keep_blank_chars=False)
                    if words:
                        # Group words into approximate lines by y0
                        lines: Dict[int, List[Dict[str, Any]]] = {}
                        for w in words:
                            # Group by approximate vertical position (rounded to nearest 5px)
                            line_key = int(w["top"] // 8) * 8
                            lines.setdefault(line_key, []).append(w)

                        for block_idx, (y_key, line_words) in enumerate(sorted(lines.items())):
                            line_words.sort(key=lambda x: x["x0"])
                            line_str = " ".join(w["text"] for w in line_words)
                            if line_str.strip():
                                x0 = min(w["x0"] for w in line_words)
                                top = min(w["top"] for w in line_words)
                                x1 = max(w["x1"] for w in line_words)
                                bottom = max(w["bottom"] for w in line_words)
                                blocks.append(
                                    TextBlock(
                                        block_id=block_idx + 1,
                                        text=line_str.strip(),
                                        bbox=[float(x0), float(top), float(x1), float(bottom)],
                                        block_type="text",
                                    )
                                )
                except Exception as e:
                    logger.debug(f"Block coordinate extraction skipped: {e}")

                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=clean_text,
                        blocks=blocks,
                        is_ocr=False,
                        metadata={"width": float(page.width), "height": float(page.height)},
                    )
                )

        doc = Document(
            document_id=doc_id,
            source_path=str(file_path.resolve()),
            pages=pages,
            file_type="pdf",
            metadata={
                "total_pages": total_pages,
                "is_scanned": is_scanned_doc,
                "file_size_bytes": file_path.stat().st_size,
            },
        )
        return doc
