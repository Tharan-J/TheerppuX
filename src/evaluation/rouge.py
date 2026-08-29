"""
ROUGE Evaluation Metric (Supplementary).
Computes ROUGE-1, ROUGE-2, and ROUGE-L scores using rouge-score.
Note: ROUGE is a supplementary lexical overlap metric and does NOT directly measure legal accuracy or fact preservation.
"""

from typing import Dict, List, Optional
import logging
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class ROUGEMetric:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    Tagged as supplementary: lexical overlap does not guarantee legal correctness.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=False)

    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute average ROUGE scores across predictions and references.
        """
        if not predictions or not references:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        r1_scores = []
        r2_scores = []
        rl_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rl_scores.append(scores["rougeL"].fmeasure)

        avg_r1 = sum(r1_scores) / max(len(r1_scores), 1) * 100.0
        avg_r2 = sum(r2_scores) / max(len(r2_scores), 1) * 100.0
        avg_rl = sum(rl_scores) / max(len(rl_scores), 1) * 100.0

        return {
            "rouge_1": round(float(avg_r1), 2),
            "rouge_2": round(float(avg_r2), 2),
            "rouge_l": round(float(avg_rl), 2),
            "note": "Supplementary metric: lexical overlap only",
        }

    def compute_sentence(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute sentence-level ROUGE scores."""
        scores = self.scorer.score(reference, prediction)
        return {
            "rouge_1": round(float(scores["rouge1"].fmeasure * 100.0), 2),
            "rouge_2": round(float(scores["rouge2"].fmeasure * 100.0), 2),
            "rouge_l": round(float(scores["rougeL"].fmeasure * 100.0), 2),
        }
