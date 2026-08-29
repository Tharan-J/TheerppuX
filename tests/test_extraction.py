"""
Tests for Document Loading and Page Extraction.
"""

from pathlib import Path
import pytest
from src.document.loader import DocumentLoader
from src.document.models import Document, PageContent, TextBlock


def test_document_loader_txt():
    demo_file = Path("data/raw/case_001.txt")
    if not demo_file.exists():
        pytest.skip("Demo file not found")

    loader = DocumentLoader(ocr_enabled=False)
    doc = loader.load(demo_file)

    assert isinstance(doc, Document)
    assert doc.document_id == "case_001"
    assert doc.total_pages >= 1
    assert doc.total_words > 50
    assert "HIGH COURT" in doc.full_text


def test_document_unsupported_format(tmp_path):
    invalid_file = tmp_path / "test.csv"
    invalid_file.write_text("a,b,c")

    loader = DocumentLoader()
    with pytest.raises(ValueError) as excinfo:
        loader.load(invalid_file)

    assert "Unsupported document format" in str(excinfo.value)


def test_document_serialization():
    page = PageContent(
        page_number=1,
        text="Sample text",
        blocks=[TextBlock(block_id=1, text="Sample text", bbox=[0, 0, 100, 20])],
    )
    doc = Document(
        document_id="test_doc",
        source_path="/path/test.pdf",
        pages=[page],
        file_type="pdf",
    )

    doc_dict = doc.to_dict()
    assert doc_dict["document_id"] == "test_doc"
    assert len(doc_dict["pages"]) == 1
    assert doc_dict["pages"][0]["blocks"][0]["block_id"] == 1

    restored = Document.from_dict(doc_dict)
    assert restored.document_id == doc.document_id
    assert restored.total_pages == 1
