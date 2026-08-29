"""
Data models for document representation, page-level preservation, and chunk metadata.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class TextBlock:
    """Individual bounding block of text within a page."""
    block_id: int
    text: str
    bbox: Optional[List[float]] = None  # [x0, top, x1, bottom]
    block_type: str = "text"  # 'text', 'header', 'footer', 'table'


@dataclass
class PageContent:
    """Single page representation preserving page boundaries for citation/provenance."""
    page_number: int
    text: str
    blocks: List[TextBlock] = field(default_factory=list)
    is_ocr: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "blocks": [asdict(b) if isinstance(b, TextBlock) else b for b in self.blocks],
            "is_ocr": self.is_ocr,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageContent":
        blocks = [
            TextBlock(**b) if isinstance(b, dict) else b
            for b in data.get("blocks", [])
        ]
        return cls(
            page_number=data["page_number"],
            text=data["text"],
            blocks=blocks,
            is_ocr=data.get("is_ocr", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document:
    """Complete document representation with structured pages and provenance."""
    document_id: str
    source_path: str
    pages: List[PageContent] = field(default_factory=list)
    file_type: str = "pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_words(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": str(self.source_path),
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "total_words": self.total_words,
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        pages = [PageContent.from_dict(p) for p in data.get("pages", [])]
        return cls(
            document_id=data["document_id"],
            source_path=data["source_path"],
            pages=pages,
            file_type=data.get("file_type", "pdf"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProcessedChunk:
    """Preprocessed and segmented text chunk with legal section metadata."""
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    text: str
    section_type: str = "BODY"  # FACTS, ISSUES, ARGUMENTS, FINDINGS, ORDER, JUDGMENT, etc.
    sentence_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedChunk":
        return cls(**data)
