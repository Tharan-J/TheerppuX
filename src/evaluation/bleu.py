"""
BLEU Evaluation Metric for Tamil and Malayalam Legal Translation.
Computes standard n-gram precision (BLEU-1, BLEU-2, BLEU-3, BLEU-4) and overall SacreBLEU.
"""

from typing import Dict, List, Optional
import logging
import sacrebleu

logger = logging.getLogger(__name__)


class BLEUMetric:
    """
    Computes BLEU scores (BLEU-1, 2, 3, 4 and cumulative corpus BLEU) using SacreBLEU.
    Supports FLORES-200 and standard multilingual tokenizers.
    """

    def __init__(self, tokenize: str = "flores200"):
        self.tokenize = tokenize

    def compute(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute corpus BLEU across predictions and references.
        references: list of reference lists, e.g. [[ref1_doc1], [ref1_doc2]] or [[ref1_doc1, ref1_doc2], [ref2_doc1, ref2_doc2]]
        """
        if not predictions or not references:
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

        # Format references for sacrebleu: list of reference streams
        # If references is list of single reference strings per sample [[ref1], [ref2]]
        if isinstance(references[0], str):
            ref_streams = [references]
        elif isinstance(references[0], list):
            # Transpose if references is list of per-sample ref lists
            num_refs = max(len(r) for r in references)
            ref_streams = []
            for ref_idx in range(num_refs):
                stream = [r[ref_idx] if ref_idx < len(r) else r[0] for r in references]
                ref_streams.append(stream)
        else:
            ref_streams = [references]

        try:
            # Try specified tokenizer
            bleu = sacrebleu.corpus_bleu(
                predictions,
                ref_streams,
                tokenize=self.tokenize,
            )
        except Exception:
            # Fallback to standard 13a or none if flores200 unavailable in older sacrebleu
            bleu = sacrebleu.corpus_bleu(
                predictions,
                ref_streams,
                tokenize="13a",
            )

        precisions = bleu.precisions if hasattr(bleu, "precisions") else [0.0, 0.0, 0.0, 0.0]
        # Pad precisions if less than 4
        while len(precisions) < 4:
            precisions.append(0.0)

        return {
            "bleu": round(float(bleu.score), 2),
            "bleu_1": round(float(precisions[0]), 2),
            "bleu_2": round(float(precisions[1]), 2),
            "bleu_3": round(float(precisions[2]), 2),
            "bleu_4": round(float(precisions[3]), 2),
            "brevity_penalty": round(float(bleu.bp), 4) if hasattr(bleu, "bp") else 1.0,
        }

    def compute_sentence(self, prediction: str, reference: str) -> float:
        """Compute sentence-level BLEU score."""
        try:
            score = sacrebleu.sentence_bleu(prediction, [reference], tokenize=self.tokenize)
            return round(float(score.score), 2)
        except Exception:
            score = sacrebleu.sentence_bleu(prediction, [reference], tokenize="13a")
            return round(float(score.score), 2)
