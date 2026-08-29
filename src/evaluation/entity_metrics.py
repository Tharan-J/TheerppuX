"""
Legal Entity Preservation and Number Preservation Metrics.
Extracts and compares Person names, Courts, Locations, Dates, Statutes,
Sections, Case numbers, and Monetary values between source, reference, and candidate translations.
"""

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntitySpan:
    entity_type: str  # Person, Court, Date, Law, Section, Case_Number, Location, Money, Quantity
    text: str
    normalized: str


class EntityExtractor:
    """
    Rule-based and pattern-based legal entity extractor for English, Tamil, and Malayalam legal texts.
    """

    def __init__(self):
        # English patterns
        self._en_section_pat = re.compile(
            r"\b(?:Section|Sec\.|Sections|Secs\.|u/s)\s*(\d+[A-Z]?(?:\s*(?:of\s+the\s+|read\s+with\s+)?[A-Za-z\.\s]{2,25})?)",
            re.IGNORECASE,
        )
        self._en_case_no_pat = re.compile(
            r"\b(?:Crl\.A\.|Crl\.O\.P\.|W\.P\.|O\.S\.|C\.A\.|S\.L\.P\.|Civil\s+Appeal|Criminal\s+Appeal)\s*(?:\([A-Za-z]+\))?\s*No\.?\s*(\d+[\d/]*(?:\s+of\s+\d{4})?)",
            re.IGNORECASE,
        )
        self._en_money_pat = re.compile(
            r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?(?:\s*/-)?)|(\b[\d,]+\s*(?:rupees|lakhs|crores)\b)",
            re.IGNORECASE,
        )
        self._en_date_pat = re.compile(
            r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4})\b",
            re.IGNORECASE,
        )
        self._en_court_pat = re.compile(
            r"\b((?:Supreme|High|District|Sessions|Magistrate|Subordinate)\s+Court(?:\s+of\s+[A-Za-z]+)?)\b",
            re.IGNORECASE,
        )
        self._en_statute_pat = re.compile(
            r"\b((?:Indian\s+Penal\s+Code|Code\s+of\s+Criminal\s+Procedure|Code\s+of\s+Civil\s+Procedure|Negotiable\s+Instruments\s+Act|Constitution\s+of\s+India|Evidence\s+Act|Specific\s+Relief\s+Act|Motor\s+Vehicles\s+Act)(?:\s*,?\s*\d{4})?)\b",
            re.IGNORECASE,
        )

        # Tamil patterns
        self._ta_section_pat = re.compile(r"(?:பிரிவு|பிரிவுகள்|செக்ஷன்)\s*(\d+[A-Z]?)", re.IGNORECASE)
        self._ta_court_pat = re.compile(r"((?:உயர்|உச்ச|மாவட்ட|அமர்வு|நீதித்துறை\s+நடுவர்)\s*நீதிமன்றம்)")
        self._ta_money_pat = re.compile(r"(?:₹|ரூ\.?|ரூபாய்)\s*([\d,]+(?:\.\d+)?)|([\d,]+\s*ரூபாய்)")

        # Malayalam patterns
        self._ml_section_pat = re.compile(r"(?:വകുപ്പ്|വകുപ്പുകൾ|സെക്ഷൻ)\s*(\d+[A-Z]?)", re.IGNORECASE)
        self._ml_court_pat = re.compile(r"((?:ഹൈക്കോടതി|സുപ്രീം\s*കോടതി|ജില്ലാ\s*കോടതി|സെഷൻസ്\s*കോടതി|മജിസ്‌ട്രേറ്റ്\s*കോടതി))")
        self._ml_money_pat = re.compile(r"(?:₹|രൂപ)\s*([\d,]+(?:\.\d+)?)|([\d,]+\s*രൂപ)")

    def extract_entities(self, text: str, lang: str = "en") -> List[EntitySpan]:
        """Extract legal entity spans from text."""
        spans: List[EntitySpan] = []

        # 1. Numbers / Digits (universal ground truth)
        numbers = re.findall(r"\b\d+[\d,]*\b", text)
        for num in numbers:
            clean_num = num.replace(",", "")
            spans.append(EntitySpan(entity_type="Number", text=num, normalized=clean_num))

        # 2. Dates (universal pattern)
        dates = re.findall(
            r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", text
        )
        for d in dates:
            spans.append(EntitySpan(entity_type="Date", text=d, normalized=d.replace("/", "-").replace(".", "-")))

        # 3. Currency / Monetary
        if lang == "en":
            for m in self._en_money_pat.finditer(text):
                raw = m.group(0)
                digits = re.sub(r"[^\d]", "", raw)
                if digits:
                    spans.append(EntitySpan(entity_type="Money", text=raw, normalized=digits))
            for m in self._en_section_pat.finditer(text):
                raw = m.group(0)
                sec_no = re.search(r"\d+[A-Z]?", raw)
                norm = sec_no.group(0) if sec_no else raw
                spans.append(EntitySpan(entity_type="Section", text=raw, normalized=norm))
            for m in self._en_case_no_pat.finditer(text):
                raw = m.group(0)
                spans.append(EntitySpan(entity_type="Case_Number", text=raw, normalized=re.sub(r"\s+", "", raw)))
            for m in self._en_court_pat.finditer(text):
                spans.append(EntitySpan(entity_type="Court", text=m.group(0), normalized=m.group(0).lower()))
            for m in self._en_statute_pat.finditer(text):
                spans.append(EntitySpan(entity_type="Law", text=m.group(0), normalized=m.group(0).lower()))

        elif lang in ("ta", "tam_Taml"):
            for m in self._ta_money_pat.finditer(text):
                raw = m.group(0)
                digits = re.sub(r"[^\d]", "", raw)
                if digits:
                    spans.append(EntitySpan(entity_type="Money", text=raw, normalized=digits))
            for m in self._ta_section_pat.finditer(text):
                spans.append(EntitySpan(entity_type="Section", text=m.group(0), normalized=m.group(1)))
            for m in self._ta_court_pat.finditer(text):
                spans.append(EntitySpan(entity_type="Court", text=m.group(0), normalized=m.group(0)))

        elif lang in ("ml", "mal_Mlym"):
            for m in self._ml_money_pat.finditer(text):
                raw = m.group(0)
                digits = re.sub(r"[^\d]", "", raw)
                if digits:
                    spans.append(EntitySpan(entity_type="Money", text=raw, normalized=digits))
            for m in self._ml_section_pat.finditer(text):
                spans.append(EntitySpan(entity_type="Section", text=m.group(0), normalized=m.group(1)))
            for m in self._ml_court_pat.finditer(text):
                spans.append(EntitySpan(entity_type="Court", text=m.group(0), normalized=m.group(0)))

        return spans


class EntityPreservationMetric:
    """
    Computes Precision, Recall, and F1 for named legal entity types and number preservation.
    """

    def __init__(self):
        self.extractor = EntityExtractor()

    def compute(
        self,
        source_texts: List[str],
        candidate_texts: List[str],
        target_lang: str = "ta",
    ) -> Dict[str, Any]:
        """
        Compute entity preservation scores between source documents and model translations.
        """
        entity_types = ["Section", "Money", "Date", "Court", "Case_Number", "Law", "Number"]
        tp_counts = Counter()
        fp_counts = Counter()
        fn_counts = Counter()

        # Dedicated number fidelity counter
        total_source_numbers = 0
        preserved_source_numbers = 0
        corrupted_numbers = []

        for src, cand in zip(source_texts, candidate_texts):
            src_entities = self.extractor.extract_entities(src, lang="en")
            cand_entities = self.extractor.extract_entities(cand, lang=target_lang)

            src_by_type: Dict[str, Set[str]] = {}
            cand_by_type: Dict[str, Set[str]] = {}

            for e in src_entities:
                src_by_type.setdefault(e.entity_type, set()).add(e.normalized)
            for e in cand_entities:
                cand_by_type.setdefault(e.entity_type, set()).add(e.normalized)

            # Check entity matches
            for etype in entity_types:
                src_set = src_by_type.get(etype, set())
                cand_set = cand_by_type.get(etype, set())

                tp = len(src_set & cand_set)
                fn = len(src_set - cand_set)
                fp = len(cand_set - src_set)

                tp_counts[etype] += tp
                fn_counts[etype] += fn
                fp_counts[etype] += fp

            # Check numeric integrity
            src_nums = src_by_type.get("Number", set())
            cand_nums = cand_by_type.get("Number", set())
            for num in src_nums:
                total_source_numbers += 1
                if num in cand_nums:
                    preserved_source_numbers += 1
                else:
                    corrupted_numbers.append({"expected": num, "source_context": src[:100]})

        # Calculate per-type Precision, Recall, F1
        per_type_metrics = {}
        total_tp = sum(tp_counts.values())
        total_fp = sum(fp_counts.values())
        total_fn = sum(fn_counts.values())

        for etype in entity_types:
            tp = tp_counts[etype]
            fp = fp_counts[etype]
            fn = fn_counts[etype]

            precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 100.0
            recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_type_metrics[etype] = {
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1": round(f1, 2),
                "support": tp + fn,
            }

        # Overall Micro F1
        overall_prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 100.0
        overall_rec = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 100.0
        overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0

        number_accuracy = (
            (preserved_source_numbers / total_source_numbers * 100.0)
            if total_source_numbers > 0
            else 100.0
        )

        return {
            "entity_precision": round(overall_prec, 2),
            "entity_recall": round(overall_rec, 2),
            "entity_f1": round(overall_f1, 2),
            "number_accuracy": round(number_accuracy, 2),
            "total_numbers": total_source_numbers,
            "preserved_numbers": preserved_source_numbers,
            "corrupted_number_count": len(corrupted_numbers),
            "per_entity_type": per_type_metrics,
        }
