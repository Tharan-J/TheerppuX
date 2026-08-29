"""
Legal Terminology Preservation Metric.
Evaluates accuracy of domain-specific legal terms against configurable Tamil/Malayalam dictionaries.
Classifies errors: correct, omitted, mistranslated, left untranslated.
"""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TermEvaluationResult:
    term: str
    status: str  # 'correct', 'omitted', 'untranslated', 'mistranslated'
    source_context: str
    target_context: str
    expected_canonical: str


class LegalTerminologyMetric:
    """
    Evaluates legal terminology translation accuracy against domain dictionaries.
    """

    def __init__(self, legal_terms_dict: Optional[Dict[str, Any]] = None):
        self.terms_dict = legal_terms_dict or {}

    def compute(
        self,
        source_texts: List[str],
        candidate_texts: List[str],
        target_lang: str = "ta",
    ) -> Dict[str, Any]:
        """
        Compute Legal Term Accuracy and detailed error breakdown across documents.
        """
        if not self.terms_dict:
            return {
                "legal_term_accuracy": 100.0,
                "total_legal_terms": 0,
                "correct_terms": 0,
                "omitted_terms": 0,
                "untranslated_terms": 0,
                "mistranslated_terms": 0,
                "term_details": [],
            }

        total_occurrences = 0
        status_counts = Counter()
        details: List[Dict[str, Any]] = []

        # Antonym / Contradiction pairs for detecting mistranslations
        antonyms = {
            "conviction": ["விடுதலை", "വെറുതെ വിടൽ", "കുറ്റവിമുക്തനാക്കൽ"],
            "acquittal": ["தண்டனை", "ശിക്ഷ"],
            "allowed": ["தள்ளுபடி", "തള്ളി"],
            "dismissed": ["அனுமதிக்கப்பட்டது", "അനുവദിച്ചു"],
            "plaintiff": ["பிரதிவாதி", "പ്രതി"],
            "defendant": ["வாதி", "വാദിഭാഗം"],
            "petitioner": ["எதிர்மனுதாரர்", "എതിർകക്ഷി"],
            "respondent": ["மனுதாரர்", "ഹർജിക്കാരൻ"],
        }

        for src, cand in zip(source_texts, candidate_texts):
            src_lower = src.lower()

            for term_key, term_info in self.terms_dict.items():
                # Check if term or phrase is present in source text
                pattern = rf"\b{re.escape(term_key.replace('_', ' '))}\b"
                if re.search(pattern, src_lower):
                    total_occurrences += 1
                    canonical = term_info.get("canonical", "")
                    variants = term_info.get("variants", [canonical])

                    # 1. Check if correctly translated
                    is_correct = any(v in cand for v in variants)

                    # 2. Check if untranslated (raw English word in target)
                    is_untranslated = bool(re.search(pattern, cand.lower()))

                    # 3. Check for severe contradiction / mistranslation
                    is_contradiction = False
                    if term_key in antonyms:
                        contradictory_terms = antonyms[term_key]
                        if any(c_term in cand for c_term in contradictory_terms) and not is_correct:
                            is_contradiction = True

                    if is_correct:
                        status = "correct"
                    elif is_contradiction:
                        status = "mistranslated"
                    elif is_untranslated:
                        status = "untranslated"
                    else:
                        status = "omitted"

                    status_counts[status] += 1
                    details.append(
                        {
                            "term": term_key,
                            "status": status,
                            "expected": canonical,
                            "found_in_candidate": is_correct,
                        }
                    )

        accuracy = (
            (status_counts["correct"] / total_occurrences * 100.0)
            if total_occurrences > 0
            else 100.0
        )

        return {
            "legal_term_accuracy": round(accuracy, 2),
            "total_legal_terms": total_occurrences,
            "correct_terms": status_counts["correct"],
            "omitted_terms": status_counts["omitted"],
            "untranslated_terms": status_counts["untranslated"],
            "mistranslated_terms": status_counts["mistranslated"],
            "term_details": details[:50],  # sample details
        }
