"""
Legal Error Taxonomy & Automated Error Classification (E1 - E12).
Classifies translation errors into CRITICAL, MAJOR, and MINOR severities,
specifically detecting negation flips, number distortions, section corruptions, and meaning reversals.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set
import logging

from src.evaluation.entity_metrics import EntityExtractor

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class ErrorTaxonomy(str, Enum):
    E1_ENTITY_ERROR = "E1 — Entity error"
    E2_DATE_ERROR = "E2 — Date error"
    E3_NUMBER_MONEY_ERROR = "E3 — Number/money error"
    E4_LEGAL_SECTION_ERROR = "E4 — Legal-section error"
    E5_NEGATION_ERROR = "E5 — Negation error"
    E6_PARTY_ROLE_ERROR = "E6 — Party-role error"
    E7_LEGAL_TERM_ERROR = "E7 — Legal-term error"
    E8_OMISSION = "E8 — Omission"
    E9_ADDITION_HALLUCINATION = "E9 — Addition/hallucination"
    E10_MEANING_REVERSAL = "E10 — Meaning reversal"
    E11_FLUENCY_ERROR = "E11 — Fluency error"
    E12_FORMATTING_ERROR = "E12 — Formatting error"


@dataclass
class ClassifiedError:
    error_id: str
    error_type: ErrorTaxonomy
    severity: Severity
    description: str
    source_snippet: str
    candidate_snippet: str
    reference_snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "source_snippet": self.source_snippet,
            "candidate_snippet": self.candidate_snippet,
            "reference_snippet": self.reference_snippet,
            "metadata": self.metadata,
        }


class LegalErrorClassifier:
    """
    Automated detector and classifier for legal translation errors according to the E1-E12 taxonomy.
    """

    def __init__(self, legal_terms_dict: Optional[Dict[str, Any]] = None):
        self.terms_dict = legal_terms_dict or {}
        self.entity_extractor = EntityExtractor()

        # Negation keywords for English
        self._en_negations = {"not", "no", "never", "neither", "nor", "without", "failed", "denied", "unable", "dismissed"}
        # Tamil negative markers
        self._ta_negatives = {"இல்லை", "கூடாது", "மறுத்தது", "தவறியது", "தள்ளுபடி", "இல்லாமல்", "செல்லாது"}
        # Malayalam negative markers
        self._ml_negatives = {"ഇല്ല", "പാടില്ല", "നിരസിച്ചു", "പരാജയപ്പെട്ടു", "തള്ളി", "കൂടാതെ", "അസാധു"}

        # Critical outcome antonym pairs
        self._outcome_reversals = [
            ("convicted", "acquitted", ["விடுதலை", "வெറുതെ വിട്ടു", "കുറ്റവിമുക്തനാക്കി"], ["தண்டனை", "ശിക്ഷ"]),
            ("acquitted", "convicted", ["தண்டனை", "ശിക്ഷ"], ["விடுதலை", "കുറ്റവിമുക്തനാക്കി"]),
            ("allowed", "dismissed", ["தள்ளுபடி", "തള്ളി"], ["அனுமதிக்கப்பட்டது", "അനുവദിച്ചു"]),
            ("dismissed", "allowed", ["அனுமதிக்கப்பட்டது", "അനുവദിച്ചു"], ["தள்ளுபடி", "തള്ളി"]),
            ("guilty", "not guilty", ["குற்றமற்றவர்", "കുറ്റക്കാരനല്ല"], ["குற்றவாளி", "കുറ്റക്കാരൻ"]),
            ("not guilty", "guilty", ["குற்றவாளி", "കുറ്റക്കാരൻ"], ["குற்றமற்றவர்", "കുറ്റക്കാരനല്ല"]),
        ]

    def analyze(
        self,
        source_texts: List[str],
        candidate_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        target_lang: str = "ta",
    ) -> Dict[str, Any]:
        """
        Analyze translation corpus and classify all detected errors into E1-E12 taxonomy.
        """
        errors: List[ClassifiedError] = []
        error_count = 0

        for idx, (src, cand) in enumerate(zip(source_texts, candidate_texts)):
            ref = reference_texts[idx] if reference_texts and idx < len(reference_texts) else None

            # 1. E3: Number & Monetary Error Detection (CRITICAL)
            src_nums = set(re.findall(r"\b\d+[\d,]*\b", src))
            cand_nums = set(re.findall(r"\b\d+[\d,]*\b", cand))
            missing_nums = src_nums - cand_nums
            if missing_nums:
                for num in missing_nums:
                    clean_num = num.replace(",", "")
                    # Filter trivial single digits if desired, or flag all
                    if len(clean_num) >= 2 or "₹" in src or "Rs" in src:
                        error_count += 1
                        errors.append(
                            ClassifiedError(
                                error_id=f"ERR_{error_count:04d}",
                                error_type=ErrorTaxonomy.E3_NUMBER_MONEY_ERROR,
                                severity=Severity.CRITICAL,
                                description=f"Numeric/monetary figure '{num}' distorted or missing in translation",
                                source_snippet=src[:120],
                                candidate_snippet=cand[:120],
                                reference_snippet=ref[:120] if ref else None,
                                metadata={"missing_number": num},
                            )
                        )

            # 2. E4: Legal Section Error Detection (CRITICAL)
            src_sections = re.findall(r"\b(?:Section|Sec\.|u/s)\s*(\d+[A-Z]?)", src, re.IGNORECASE)
            for sec in src_sections:
                if sec not in cand:
                    error_count += 1
                    errors.append(
                        ClassifiedError(
                            error_id=f"ERR_{error_count:04d}",
                            error_type=ErrorTaxonomy.E4_LEGAL_SECTION_ERROR,
                            severity=Severity.CRITICAL,
                            description=f"Statutory section number '{sec}' missing or altered",
                            source_snippet=src[:120],
                            candidate_snippet=cand[:120],
                            reference_snippet=ref[:120] if ref else None,
                            metadata={"section": sec},
                        )
                    )

            # 3. E10: Meaning Reversal (CRITICAL)
            src_lower = src.lower()
            for src_term, opposing_term, false_markers, true_markers in self._outcome_reversals:
                if f" {src_term} " in f" {src_lower} ":
                    # Check if false marker (opposing meaning) was used instead
                    if any(fm in cand for fm in false_markers) and not any(tm in cand for tm in true_markers):
                        error_count += 1
                        errors.append(
                            ClassifiedError(
                                error_id=f"ERR_{error_count:04d}",
                                error_type=ErrorTaxonomy.E10_MEANING_REVERSAL,
                                severity=Severity.CRITICAL,
                                description=f"Critical outcome meaning reversal: '{src_term}' translated as opposing concept '{opposing_term}'",
                                source_snippet=src[:120],
                                candidate_snippet=cand[:120],
                                reference_snippet=ref[:120] if ref else None,
                                metadata={"source_concept": src_term, "reversed_to": opposing_term},
                            )
                        )

            # 4. E5: Negation Error Detection (CRITICAL)
            src_has_neg = any(w in src_lower.split() for w in self._en_negations)
            neg_markers = self._ta_negatives if target_lang == "ta" else self._ml_negatives
            cand_has_neg = any(m in cand for m in neg_markers)

            if src_has_neg and not cand_has_neg and len(src.split()) < 20:
                # Potential dropped negation in short legal clause
                error_count += 1
                errors.append(
                    ClassifiedError(
                        error_id=f"ERR_{error_count:04d}",
                        error_type=ErrorTaxonomy.E5_NEGATION_ERROR,
                        severity=Severity.CRITICAL,
                        description="Potential dropped negation marker in translated legal finding",
                        source_snippet=src[:120],
                        candidate_snippet=cand[:120],
                        reference_snippet=ref[:120] if ref else None,
                    )
                )

            # 5. E6: Party Role Error (CRITICAL / MAJOR)
            if "petitioner" in src_lower and "respondent" not in src_lower:
                opposing = "எதிர்மனுதாரர்" if target_lang == "ta" else "എതിർകക്ഷി"
                if opposing in cand:
                    error_count += 1
                    errors.append(
                        ClassifiedError(
                            error_id=f"ERR_{error_count:04d}",
                            error_type=ErrorTaxonomy.E6_PARTY_ROLE_ERROR,
                            severity=Severity.CRITICAL,
                            description="Party role reversed: 'petitioner' translated as 'respondent'",
                            source_snippet=src[:120],
                            candidate_snippet=cand[:120],
                            reference_snippet=ref[:120] if ref else None,
                        )
                    )

            # 6. E8: Severe Omission (MAJOR)
            if len(src.split()) > 10 and len(cand.strip()) < 10:
                error_count += 1
                errors.append(
                    ClassifiedError(
                        error_id=f"ERR_{error_count:04d}",
                        error_type=ErrorTaxonomy.E8_OMISSION,
                        severity=Severity.MAJOR,
                        description="Severe sentence-level content omission",
                        source_snippet=src[:120],
                        candidate_snippet=cand[:120],
                        reference_snippet=ref[:120] if ref else None,
                    )
                )

        # Summary Statistics
        total_sentences = len(source_texts)
        critical_errors = [e for e in errors if e.severity == Severity.CRITICAL]
        major_errors = [e for e in errors if e.severity == Severity.MAJOR]
        minor_errors = [e for e in errors if e.severity == Severity.MINOR]

        critical_error_rate = (len(critical_errors) / max(total_sentences, 1)) * 100.0

        taxonomy_counts = {}
        for et in ErrorTaxonomy:
            taxonomy_counts[et.value] = sum(1 for e in errors if e.error_type == et)

        return {
            "total_errors": len(errors),
            "total_evaluated_chunks": total_sentences,
            "critical_errors": len(critical_errors),
            "major_errors": len(major_errors),
            "minor_errors": len(minor_errors),
            "critical_error_rate_pct": round(critical_error_rate, 2),
            "taxonomy_breakdown": taxonomy_counts,
            "classified_errors": [e.to_dict() for e in errors],
        }
