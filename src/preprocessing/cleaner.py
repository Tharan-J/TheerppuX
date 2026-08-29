"""
Text Cleaner for Indian Legal Documents.
Performs Unicode normalization, OCR artifact filtering, de-hyphenation,
and header/footer removal while strictly preserving legal citations, sections, and numbers.
"""

from collections import Counter
import re
from typing import Dict, List, Optional, Set
import unicodedata
import logging

from src.document.models import Document, PageContent

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans extracted legal text while strictly preserving:
    - Section numbers (e.g. 'Section 302', 'u/s 482')
    - Case numbers (e.g. 'Crl.A. No. 1234/2023')
    - Dates (e.g. '12-03-2024', '12th March 2024')
    - Monetary amounts (e.g. '₹50,000', 'Rs. 50,000/-')
    - Court names, Party names, Statute citations
    """

    def __init__(
        self,
        unicode_norm: str = "NFC",
        clean_ocr_artifacts: bool = True,
        remove_headers_footers: bool = True,
    ):
        self.unicode_norm = unicode_norm
        self.clean_ocr_artifacts = clean_ocr_artifacts
        self.remove_headers_footers = remove_headers_footers

    def clean_text(self, text: str) -> str:
        """Clean a single string of legal text."""
        if not text:
            return ""

        # 1. Unicode Normalization
        text = unicodedata.normalize(self.unicode_norm, text)

        # 2. Fix broken cross-line hyphenations (e.g., "res-\npondent" -> "respondent")
        text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)

        # 3. Clean OCR artifacts if enabled
        if self.clean_ocr_artifacts:
            # Fix common OCR symbol garbles while keeping legal symbols
            # Remove excessive repetitive symbols (e.g., "_______", ".....", "~~~~")
            text = re.sub(r"([_~=\*\-–—]){4,}", " ", text)
            # Remove isolated unprintable or non-standard control characters (except newline, tab)
            text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")

        # 4. Normalize quotes and apostrophes to standard forms
        text = re.sub(r"[`‘’‛′]", "'", text)
        text = re.sub(r"[“”‟″]", '"', text)

        # 5. Normalize whitespace while preserving single newlines for paragraphs
        lines = [line.strip() for line in text.split("\n")]
        # Remove standalone page numbers from lines (e.g., "- 1 -", "Page 1 of 5", "1")
        cleaned_lines = []
        for line in lines:
            if self._is_standalone_page_number(line):
                continue
            cleaned_lines.append(line)

        # Collapse multiple blank lines into double newline
        reconstructed = "\n".join(cleaned_lines)
        reconstructed = re.sub(r"\n{3,}", "\n\n", reconstructed)
        # Collapse multiple spaces
        reconstructed = re.sub(r"[ \t]+", " ", reconstructed)

        return reconstructed.strip()

    def clean_document(self, doc: Document) -> Document:
        """
        Clean an entire Document object, removing cross-page repeated headers/footers
        and returning a cleaned Document instance.
        """
        headers_to_remove: Set[str] = set()
        footers_to_remove: Set[str] = set()

        if self.remove_headers_footers and len(doc.pages) > 2:
            headers_to_remove, footers_to_remove = self._detect_repeated_headers_footers(doc.pages)

        cleaned_pages: List[PageContent] = []
        for page in doc.pages:
            lines = [l.strip() for l in page.text.split("\n") if l.strip()]
            filtered_lines: List[str] = []

            for idx, line in enumerate(lines):
                # Filter top line if detected header
                if idx <= 2 and line in headers_to_remove:
                    continue
                # Filter bottom line if detected footer
                if idx >= len(lines) - 3 and line in footers_to_remove:
                    continue
                # Filter page numbers
                if self._is_standalone_page_number(line):
                    continue

                filtered_lines.append(line)

            page_raw = "\n".join(filtered_lines)
            cleaned_page_text = self.clean_text(page_raw)

            cleaned_pages.append(
                PageContent(
                    page_number=page.page_number,
                    text=cleaned_page_text,
                    blocks=page.blocks,
                    is_ocr=page.is_ocr,
                    metadata=page.metadata,
                )
            )

        return Document(
            document_id=doc.document_id,
            source_path=doc.source_path,
            pages=cleaned_pages,
            file_type=doc.file_type,
            metadata={
                **doc.metadata,
                "cleaned": True,
                "removed_headers": list(headers_to_remove),
                "removed_footers": list(footers_to_remove),
            },
        )

    def _is_standalone_page_number(self, line: str) -> bool:
        """Check if line is just a page number indicator."""
        line = line.strip()
        if not line:
            return False
        # Matches: "1", "12", "- 1 -", "[12]", "Page 1", "Page 1 of 12", "Pg. 4"
        if re.fullmatch(r"[-–—\[\(]?\s*\d{1,4}\s*[-–—\]\)]?", line):
            return True
        if re.fullmatch(r"Page\s+\d+(\s+of\s+\d+)?", line, re.IGNORECASE):
            return True
        if re.fullmatch(r"Pg\.?\s*\d+", line, re.IGNORECASE):
            return True
        return False

    def _detect_repeated_headers_footers(self, pages: List[PageContent]) -> tuple[Set[str], Set[str]]:
        """Identify lines that repeat at top or bottom across > 50% of pages."""
        top_lines = []
        bottom_lines = []

        for p in pages:
            lines = [l.strip() for l in p.text.split("\n") if l.strip()]
            if len(lines) >= 1:
                top_lines.append(lines[0])
            if len(lines) >= 2:
                top_lines.append(lines[1])
            if len(lines) >= 1:
                bottom_lines.append(lines[-1])
            if len(lines) >= 2:
                bottom_lines.append(lines[-2])

        threshold = max(2, len(pages) // 2)
        top_counts = Counter(top_lines)
        bottom_counts = Counter(bottom_lines)

        headers = {line for line, count in top_counts.items() if count >= threshold and len(line) > 3}
        footers = {line for line, count in bottom_counts.items() if count >= threshold and len(line) > 3}

        return headers, footers
