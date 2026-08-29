"""
chrF and chrF++ Evaluation Metric for Morphologically Rich Dravidian Languages.
Character n-gram F-score with word n-gram extension (chrF++).
"""

from typing import Dict, List, Union
import sacrebleu


class ChrFMetric:
    """
    Computes chrF and chrF++ scores using SacreBLEU.
    Particularly informative for agglutinative Indian languages (Tamil, Malayalam).
    """

    def __init__(self, char_order: int = 6, word_order: int = 2, beta: float = 2.0):
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta

    def compute(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute corpus chrF and chrF++ scores.
        """
        if not predictions or not references:
            return {"chrf": 0.0, "chrf_plus_plus": 0.0}

        if isinstance(references[0], str):
            ref_streams = [references]
        elif isinstance(references[0], list):
            num_refs = max(len(r) for r in references)
            ref_streams = []
            for ref_idx in range(num_refs):
                stream = [r[ref_idx] if ref_idx < len(r) else r[0] for r in references]
                ref_streams.append(stream)
        else:
            ref_streams = [references]

        # Standard chrF (character n-grams only, word_order=0)
        chrf = sacrebleu.corpus_chrf(
            predictions,
            ref_streams,
            char_order=self.char_order,
            word_order=0,
            beta=self.beta,
        )

        # chrF++ (character n-grams + word n-grams, word_order=2)
        chrf_pp = sacrebleu.corpus_chrf(
            predictions,
            ref_streams,
            char_order=self.char_order,
            word_order=self.word_order,
            beta=self.beta,
        )

        return {
            "chrf": round(float(chrf.score), 2),
            "chrf_plus_plus": round(float(chrf_pp.score), 2),
        }

    def compute_sentence(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute sentence-level chrF and chrF++."""
        chrf = sacrebleu.sentence_chrf(
            prediction,
            [reference],
            char_order=self.char_order,
            word_order=0,
            beta=self.beta,
        )
        chrf_pp = sacrebleu.sentence_chrf(
            prediction,
            [reference],
            char_order=self.char_order,
            word_order=self.word_order,
            beta=self.beta,
        )
        return {
            "chrf": round(float(chrf.score), 2),
            "chrf_plus_plus": round(float(chrf_pp.score), 2),
        }
