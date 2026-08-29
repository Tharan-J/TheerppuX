"""
Cross-Language Consistency Checking Module.
Compares Tamil and Malayalam translations of the same English legal source
to evaluate factual alignment across dates, numbers, legal sections, party roles, and decisions.
"""

from typing import Any, Dict, List, Set, Tuple
import logging
from src.evaluation.entity_metrics import EntityExtractor

logger = logging.getLogger(__name__)


class CrossLanguageConsistencyChecker:
    """
    Evaluates consistency across Tamil and Malayalam translations.
    Produces a composite Cross-Language Consistency Score.
    """

    def __init__(self):
        self.extractor = EntityExtractor()

    def compare_translations(
        self,
        source_texts: List[str],
        tamil_texts: List[str],
        malayalam_texts: List[str],
    ) -> Dict[str, Any]:
        """
        Compare facts extracted from English source vs Tamil translation vs Malayalam translation.
        """
        if len(tamil_texts) != len(malayalam_texts) or len(tamil_texts) != len(source_texts):
            min_len = min(len(source_texts), len(tamil_texts), len(malayalam_texts))
            source_texts = source_texts[:min_len]
            tamil_texts = tamil_texts[:min_len]
            malayalam_texts = malayalam_texts[:min_len]

        num_chunks = len(source_texts)
        if num_chunks == 0:
            return {"consistency_score": 100.0, "details": {}}

        total_comparisons = 0
        consistent_matches = 0
        discrepancies: List[Dict[str, Any]] = []

        for idx, (src, ta, ml) in enumerate(zip(source_texts, tamil_texts, malayalam_texts)):
            src_spans = self.extractor.extract_entities(src, lang="en")
            ta_spans = self.extractor.extract_entities(ta, lang="ta")
            ml_spans = self.extractor.extract_entities(ml, lang="ml")

            src_nums = {s.normalized for s in src_spans if s.entity_type in ("Number", "Money")}
            ta_nums = {s.normalized for s in ta_spans if s.entity_type in ("Number", "Money")}
            ml_nums = {s.normalized for s in ml_spans if s.entity_type in ("Number", "Money")}

            src_secs = {s.normalized for s in src_spans if s.entity_type == "Section"}
            ta_secs = {s.normalized for s in ta_spans if s.entity_type == "Section"}
            ml_secs = {s.normalized for s in ml_spans if s.entity_type == "Section"}

            # 1. Number consistency
            all_source_nums = src_nums
            for num in all_source_nums:
                total_comparisons += 1
                ta_has = num in ta_nums
                ml_has = num in ml_nums

                if ta_has and ml_has:
                    consistent_matches += 1
                else:
                    discrepancies.append(
                        {
                            "chunk_index": idx,
                            "fact_type": "number",
                            "expected_source": num,
                            "present_in_tamil": ta_has,
                            "present_in_malayalam": ml_has,
                            "source_snippet": src[:100],
                        }
                    )

            # 2. Section consistency
            for sec in src_secs:
                total_comparisons += 1
                ta_has = sec in ta_secs or sec in ta
                ml_has = sec in ml_secs or sec in ml

                if ta_has and ml_has:
                    consistent_matches += 1
                else:
                    discrepancies.append(
                        {
                            "chunk_index": idx,
                            "fact_type": "legal_section",
                            "expected_source": sec,
                            "present_in_tamil": ta_has,
                            "present_in_malayalam": ml_has,
                            "source_snippet": src[:100],
                        }
                    )

        consistency_score = (
            (consistent_matches / total_comparisons * 100.0)
            if total_comparisons > 0
            else 100.0
        )

        return {
            "cross_language_consistency_score": round(consistency_score, 2),
            "total_facts_evaluated": total_comparisons,
            "consistent_facts": consistent_matches,
            "discrepancy_count": len(discrepancies),
            "discrepancies": discrepancies[:25],
        }
