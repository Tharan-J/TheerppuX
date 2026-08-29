"""
Tests for Text Preprocessing, Cleaning, and Legal-Aware Segmentation.
"""

from src.document.models import Document, PageContent
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.segmenter import TextSegmenter
from src.preprocessing.language_detector import LanguageDetector


def test_text_cleaner_preserves_legal_entities():
    cleaner = TextCleaner()
    raw_text = (
        "The accused was convicted u/s 302 IPC on 12-03-2024.\n"
        "He was ordered to pay a fine of ₹50,000/- in default.\n"
        "Page 1 of 5\n"
        "The res-\npondent filed Crl.A. No. 452/2023."
    )

    cleaned = cleaner.clean_text(raw_text)

    # De-hyphenation test
    assert "respondent" in cleaned
    # Section preservation test
    assert "u/s 302 IPC" in cleaned
    # Date preservation test
    assert "12-03-2024" in cleaned
    # Currency preservation test
    assert "₹50,000/-" in cleaned or "₹50,000" in cleaned
    # Standalone page number removed
    assert "Page 1 of 5" not in cleaned


def test_legal_aware_segmentation():
    segmenter = TextSegmenter()
    doc_text = (
        "IN THE HIGH COURT OF MADRAS\n\n"
        "FACTS OF THE CASE:\n"
        "The appellant was convicted under Section 302 IPC.\n\n"
        "ARGUMENTS BY COUNSEL:\n"
        "The Counsel submitted that incident was accidental.\n\n"
        "FINAL ORDER:\n"
        "The appeal is allowed in part."
    )

    doc = Document(
        document_id="test_case",
        source_path="test.txt",
        pages=[PageContent(page_number=1, text=doc_text)],
    )

    chunks = segmenter.segment_document(doc)

    assert len(chunks) >= 3
    section_types = [c.section_type for c in chunks]
    assert any("FACTS" in s for s in section_types)
    assert any("ARGUMENTS" in s for s in section_types)
    assert any("ORDER" in s for s in section_types)

    for c in chunks:
        assert c.chunk_id.startswith("test_case_chunk_")
        assert c.page_start == 1


def test_language_detector():
    detector = LanguageDetector()

    en_res = detector.detect("This is an English legal judgment from the High Court of Madras.")
    assert en_res["is_english"] is True
    assert en_res["language"] == "en"

    ta_res = detector.detect("இது சென்னை உயர் நீதிமன்றத்தின் தீர்ப்பு ஆகும்.")
    assert ta_res["is_english"] is False
    assert "Tamil" in ta_res["script_distribution"]
