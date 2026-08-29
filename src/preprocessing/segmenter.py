"""
Legal-Aware Text Segmenter for Indian Court Documents.
Preserves structural sections (FACTS, ARGUMENTS, ISSUES, ORDER, JUDGMENT)
and segments into model-compatible chunks with full provenance metadata.
"""

from typing import Any, Dict, List, Optional, Pattern
import re
import logging
from src.document.models import Document, PageContent, ProcessedChunk

logger = logging.getLogger(__name__)

# Standard Indian Legal Structural Headings
DEFAULT_LEGAL_HEADINGS = [
    ("FACTS", r"^(?:THE\s+)?FACTS(?:\s+OF\s+THE\s+CASE)?[:\.\s]*$"),
    ("BACKGROUND", r"^(?:BRIEF\s+)?BACKGROUND(?:\s+OF\s+THE\s+CASE)?[:\.\s]*$"),
    ("CONTENTIONS", r"^(?:RIVAL\s+)?CONTENTIONS(?:\s+OF\s+THE\s+PARTIES)?[:\.\s]*$"),
    ("ARGUMENTS", r"^(?:SUBMISSIONS|ARGUMENTS)(?:\s+BY\s+COUNSEL)?[:\.\s]*$"),
    ("ISSUES", r"^(?:POINTS\s+FOR\s+DETERMINATION|ISSUES\s+FRAMED|QUESTIONS\s+OF\s+LAW)[:\.\s]*$"),
    ("EVIDENCE", r"^(?:APPRECIATION\s+OF\s+)?EVIDENCE(?:\s+ON\s+RECORD)?[:\.\s]*$"),
    ("FINDINGS", r"^(?:FINDINGS|DISCUSSION\s+AND\s+FINDINGS)[:\.\s]*$"),
    ("REASONING", r"^(?:REASONS|REASONING\s+FOR\s+DECISION)[:\.\s]*$"),
    ("ORDER", r"^(?:FINAL\s+)?ORDER(?:\s+OF\s+THE\s+COURT)?[:\.\s]*$"),
    ("JUDGMENT", r"^(?:JUDGMENT|FINAL\s+JUDGMENT)[:\.\s]*$"),
    ("OPERATIVE_PORTION", r"^(?:OPERATIVE\s+PORTION|DISPOSITION|DECREE)[:\.\s]*$"),
]


class TextSegmenter:
    """
    Segments legal documents into coherent, legal-aware chunks.
    Preserves heading contexts, sentences, and page provenance.
    """

    def __init__(
        self,
        mode: str = "legal_aware",
        max_chunk_chars: int = 600,
        min_chunk_chars: int = 30,
        preserve_headings: bool = True,
    ):
        self.mode = mode
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.preserve_headings = preserve_headings

        # Compile legal heading regexes
        self._compiled_headings: List[tuple[str, Pattern]] = [
            (name, re.compile(pat, re.IGNORECASE | re.MULTILINE))
            for name, pat in DEFAULT_LEGAL_HEADINGS
        ]

    def segment_document(self, doc: Document) -> List[ProcessedChunk]:
        """
        Segment all pages of a Document into ProcessedChunk objects.
        """
        all_chunks: List[ProcessedChunk] = []
        current_section = "PREAMBLE"
        chunk_counter = 1

        for page in doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split page text into paragraphs
            paragraphs = re.split(r"\n\s*\n+", page_text)

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Check if paragraph is a section heading
                detected_heading = self._detect_section_heading(para)
                if detected_heading:
                    current_section = detected_heading
                    # If the heading itself has no subsequent content, we keep it as section marker
                    # Or attach it to the next chunk

                # Segment paragraph into sentences
                sentences = self._split_sentences(para)
                if not sentences:
                    continue

                # Group sentences into bounded chunks
                grouped_chunks = self._group_sentences_into_chunks(sentences)

                for chunk_text, sent_count in grouped_chunks:
                    chunk_id = f"{doc.document_id}_chunk_{chunk_counter:04d}"
                    chunk_counter += 1

                    all_chunks.append(
                        ProcessedChunk(
                            chunk_id=chunk_id,
                            document_id=doc.document_id,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            text=chunk_text,
                            section_type=current_section,
                            sentence_count=sent_count,
                            metadata={"char_length": len(chunk_text)},
                        )
                    )

        # Fallback if no chunks generated
        if not all_chunks and doc.pages:
            for p in doc.pages:
                if p.text.strip():
                    all_chunks.append(
                        ProcessedChunk(
                            chunk_id=f"{doc.document_id}_chunk_0001",
                            document_id=doc.document_id,
                            page_start=p.page_number,
                            page_end=p.page_number,
                            text=p.text.strip(),
                            section_type="BODY",
                            sentence_count=1,
                        )
                    )

        return all_chunks

    def _detect_section_heading(self, text: str) -> Optional[str]:
        """Check if paragraph or single line is a recognized legal heading."""
        text_clean = text.strip().strip(":.- ")
        # Headings are typically short (< 80 chars)
        if len(text_clean) > 80:
            return None

        for name, pattern in self._compiled_headings:
            if pattern.search(text_clean):
                return name

        # Check for generic numbered headings like "1. FACTS", "II. ARGUMENTS"
        numbered_match = re.match(
            r"^(?:[0-9]+|[IVXLCDM]+)[\.\)]\s*([A-Z\s]{4,30})$", text_clean
        )
        if numbered_match:
            heading_candidate = numbered_match.group(1).strip()
            for name, _ in self._compiled_headings:
                if name.replace("_", " ") in heading_candidate:
                    return name
            return heading_candidate

        return None

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while protecting legal abbreviations, citations, and sections.
        E.g., 'Sec. 302', 'v.', 'No.', 'Hon\'ble', 'i.e.', 'e.g.', 'Cr.P.C.', 'C.P.C.'
        """
        # Protect common legal abbreviations with placeholders
        protected_map = {}
        abbreviations = [
            r"Sec\.", r"u/s\.", r"u/s", r"v\.", r"vs\.", r"No\.", r"Nos\.",
            r"Hon\'ble", r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Adv\.",
            r"i\.e\.", r"e\.g\.", r"etc\.", r"viz\.", r"AIR", r"SCC",
            r"Cr\.P\.C\.", r"C\.P\.C\.", r"I\.P\.C\.", r"N\.I\.\s*Act",
            r"Ex\.P\.", r"P\.W\.", r"D\.W\.", r"C\.W\.", r"O\.S\.", r"C\.A\.",
            r"W\.P\.", r"Crl\.A\.", r"Crl\.O\.P\.", r"M\.P\."
        ]

        def replacer(match):
            key = f"__LEGAL_ABBR_{len(protected_map)}__"
            protected_map[key] = match.group(0)
            return key

        masked = text
        for abbr in abbreviations:
            masked = re.sub(abbr, replacer, masked, flags=re.IGNORECASE)

        # Protect decimal numbers and currency like 50.00 or 12.03.2024
        masked = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL_DOT__\2", masked)

        # Split on sentence terminals
        raw_sents = re.split(r"(?<=[.!?])\s+", masked)

        restored_sents = []
        for s in raw_sents:
            s = s.replace("__DECIMAL_DOT__", ".")
            for key, orig in protected_map.items():
                s = s.replace(key, orig)
            s = s.strip()
            if s:
                restored_sents.append(s)

        return restored_sents

    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[tuple[str, int]]:
        """
        Group adjacent sentences into chunks not exceeding max_chunk_chars.
        Returns list of (chunk_text, sentence_count).
        """
        chunks = []
        current_sentences = []
        current_length = 0

        for sent in sentences:
            sent_len = len(sent)

            # If adding this sentence exceeds max and we already have sentences in chunk
            if current_sentences and (current_length + sent_len + 1 > self.max_chunk_chars):
                chunk_str = " ".join(current_sentences).strip()
                chunks.append((chunk_str, len(current_sentences)))
                current_sentences = [sent]
                current_length = sent_len
            else:
                current_sentences.append(sent)
                current_length += sent_len + 1

        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            chunks.append((chunk_str, len(current_sentences)))

        return chunks
