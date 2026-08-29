"""
Master Evaluator Orchestrator for Legal Translation.
Coordinates lexical, semantic, entity, number, terminology, and error classification metrics.
Strictly adheres to empirical integrity: no fabricated scores if references are missing.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import logging

from src.evaluation.bleu import BLEUMetric
from src.evaluation.chrf import ChrFMetric
from src.evaluation.rouge import ROUGEMetric
from src.evaluation.bertscore import BERTScoreMetric
from src.evaluation.entity_metrics import EntityPreservationMetric
from src.evaluation.terminology import LegalTerminologyMetric

logger = logging.getLogger(__name__)


@dataclass
class SentenceEvaluation:
    chunk_id: str
    source_text: str
    candidate_text: str
    reference_text: Optional[str]
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    errors_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    model_name: str
    target_language: str
    total_chunks: int
    has_references: bool
    metrics: Dict[str, Any]
    sentence_metrics: List[Dict[str, Any]]
    error_analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_language": self.target_language,
            "total_chunks": self.total_chunks,
            "has_references": self.has_references,
            "metrics": self.metrics,
            "sentence_metrics": self.sentence_metrics,
            "error_analysis": self.error_analysis,
        }


class Evaluator:
    """
    Central evaluation engine coordinating all automatic and legal domain metrics.
    """

    def __init__(
        self,
        target_lang: str = "ta",
        legal_terms_dict: Optional[Dict[str, Any]] = None,
        bertscore_model: str = "bert-base-multilingual-cased",
        sacrebleu_tokenize: str = "flores200",
        device: Optional[str] = None,
    ):
        from src.analysis.error_analysis import LegalErrorClassifier

        self.target_lang = target_lang
        self.bleu_metric = BLEUMetric(tokenize=sacrebleu_tokenize)
        self.chrf_metric = ChrFMetric()
        self.rouge_metric = ROUGEMetric()
        self.bertscore_metric = BERTScoreMetric(model_type=bertscore_model, device=device)
        self.entity_metric = EntityPreservationMetric()
        self.terminology_metric = LegalTerminologyMetric(legal_terms_dict=legal_terms_dict)
        self.error_classifier = LegalErrorClassifier(legal_terms_dict=legal_terms_dict)

    def evaluate(
        self,
        candidate_texts: List[str],
        source_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        model_name: str = "translation_model",
    ) -> EvaluationReport:
        """
        Run complete evaluation suite across predictions, source, and references.
        """
        num_chunks = len(candidate_texts)
        ids = chunk_ids or [f"chunk_{i:04d}" for i in range(num_chunks)]
        has_refs = reference_texts is not None and len(reference_texts) == num_chunks and any(r.strip() for r in reference_texts)

        metrics: Dict[str, Any] = {}
        sentence_evals: List[Dict[str, Any]] = []

        # 1. Reference-based metrics (if reference is available)
        if has_refs:
            logger.info(f"Computing reference-based metrics (BLEU, chrF++, ROUGE, BERTScore) for {model_name}...")
            # BLEU
            bleu_res = self.bleu_metric.compute(candidate_texts, [[r] for r in reference_texts])
            metrics.update(bleu_res)

            # chrF / chrF++
            chrf_res = self.chrf_metric.compute(candidate_texts, [[r] for r in reference_texts])
            metrics.update(chrf_res)

            # ROUGE
            rouge_res = self.rouge_metric.compute(candidate_texts, reference_texts)
            metrics.update(rouge_res)

            # BERTScore
            bert_res = self.bertscore_metric.compute(candidate_texts, reference_texts, lang=self.target_lang)
            metrics.update(bert_res)
        else:
            logger.info(
                "Reference translation unavailable. Automatic reference-based metrics (BLEU, chrF, BERTScore) cannot be calculated."
            )
            metrics["reference_status"] = "Reference translation unavailable. Automatic reference-based metrics cannot be calculated."

        # 2. Source-to-Candidate Legal Domain Metrics (Always computed)
        logger.info(f"Computing legal entity, number, and terminology metrics for {model_name}...")
        entity_res = self.entity_metric.compute(source_texts, candidate_texts, target_lang=self.target_lang)
        metrics.update(entity_res)

        term_res = self.terminology_metric.compute(source_texts, candidate_texts, target_lang=self.target_lang)
        metrics.update(term_res)

        # 3. Error Analysis & Classification
        error_res = self.error_classifier.analyze(
            source_texts=source_texts,
            candidate_texts=candidate_texts,
            reference_texts=reference_texts if has_refs else None,
            target_lang=self.target_lang,
        )
        metrics["critical_error_rate_pct"] = error_res["critical_error_rate_pct"]
        metrics["critical_errors_count"] = error_res["critical_errors"]

        # 4. Sentence-level breakdown
        for i in range(num_chunks):
            src_i = source_texts[i]
            cand_i = candidate_texts[i]
            ref_i = reference_texts[i] if has_refs else None

            sent_bleu = self.bleu_metric.compute_sentence(cand_i, ref_i) if ref_i else None
            sent_chrf = self.chrf_metric.compute_sentence(cand_i, ref_i)["chrf_plus_plus"] if ref_i else None

            sentence_evals.append(
                SentenceEvaluation(
                    chunk_id=ids[i],
                    source_text=src_i,
                    candidate_text=cand_i,
                    reference_text=ref_i,
                    bleu=sent_bleu,
                    chrf=sent_chrf,
                ).to_dict()
            )

        return EvaluationReport(
            model_name=model_name,
            target_language=self.target_lang,
            total_chunks=num_chunks,
            has_references=has_refs,
            metrics=metrics,
            sentence_metrics=sentence_evals,
            error_analysis=error_res,
        )
