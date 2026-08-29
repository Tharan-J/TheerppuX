"""
Master Pipeline Orchestration for TheerppuX Legal Translation.
Implements TranslationPipeline, EvaluationPipeline, and ExperimentPipeline.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from src.config import PipelineConfig, load_config
from src.document.loader import DocumentLoader
from src.document.models import Document, ProcessedChunk
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.segmenter import TextSegmenter
from src.preprocessing.language_detector import LanguageDetector
from src.translation.factory import ModelFactory
from src.translation.base import TranslationResult
from src.evaluation.evaluator import Evaluator, EvaluationReport
from src.analysis.report import ReportGenerator
from src.analysis.consistency import CrossLanguageConsistencyChecker

logger = logging.getLogger("TheerppuX.Pipeline")


class TranslationPipeline:
    """
    End-to-end pipeline for translating a single legal document with a specified model configuration.
    """

    def __init__(self, config: PipelineConfig, model_name: str = "indictrans2"):
        self.config = config
        self.model_name = model_name
        self.loader = DocumentLoader(
            ocr_enabled=config.document.get("ocr", {}).get("enabled", True),
            scanned_threshold_chars=config.document.get("scanned_threshold_chars_per_page", 50),
        )
        self.cleaner = TextCleaner(
            unicode_norm=config.preprocessing.get("unicode_normalization", "NFC"),
            clean_ocr_artifacts=config.preprocessing.get("clean_ocr_artifacts", True),
            remove_headers_footers=config.preprocessing.get("remove_headers_footers", True),
        )
        self.segmenter = TextSegmenter(
            mode=config.preprocessing.get("segmentation", {}).get("mode", "legal_aware"),
            max_chunk_chars=config.preprocessing.get("max_chunk_length", 600),
            min_chunk_chars=config.preprocessing.get("min_chunk_length", 20),
        )
        self.language_detector = LanguageDetector()
        self.model = ModelFactory.create(model_name, config=config.to_dict())

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full translation pipeline on input document.
        """
        input_path = Path(input_path)
        doc_id = input_path.stem

        logger.info("=" * 60)
        logger.info("STAGE 1 LEGAL TRANSLATION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input        : {input_path.name}")
        logger.info(f"Source       : English")
        logger.info(f"Target       : {self.config.target_lang_name} ({self.config.target_lang})")
        logger.info(f"Model        : {self.model_name}")
        logger.info(f"Device       : {self.config.device_info['device'].upper()} ({self.config.device_info['device_name']})")
        logger.info(f"Precision    : {self.config.device_info['precision']}")

        # 1. Ingestion
        raw_doc = self.loader.load(input_path, document_id=doc_id)
        logger.info(f"Pages        : {raw_doc.total_pages}")
        logger.info(f"Total Words  : {raw_doc.total_words}")

        # 2. Language Validation
        lang_check = self.language_detector.detect(raw_doc.full_text[:2000])
        if not lang_check["is_english"] and lang_check["language"] != "en":
            logger.warning(
                f"Source document may not be in English! Detected: {lang_check['language']} "
                f"(confidence: {lang_check['confidence']})"
            )

        # 3. Cleaning
        cleaned_doc = self.cleaner.clean_document(raw_doc)

        # 4. Legal-Aware Segmentation
        chunks = self.segmenter.segment_document(cleaned_doc)
        logger.info(f"Chunks       : {len(chunks)}")
        logger.info("-" * 60)
        logger.info("TRANSLATION IN PROGRESS...")

        # 5. Translation
        trans_result = self.model.translate_chunks(
            chunks=chunks,
            source_lang="en",
            target_lang=self.config.target_lang,
        )
        logger.info(f"Completed    : {len(trans_result.translations)} / {len(chunks)}")
        logger.info(f"Latency      : {trans_result.latency_seconds:.2f}s ({trans_result.chars_per_second:.1f} chars/sec)")

        # 6. Save structured outputs
        out_base = Path(output_dir) if output_dir else self.config.outputs_dir / doc_id
        saved_paths = self._save_outputs(
            output_dir=out_base,
            doc=cleaned_doc,
            chunks=chunks,
            trans_result=trans_result,
        )
        logger.info("-" * 60)
        logger.info(f"Saved to     : {saved_paths['translation_file']}")
        logger.info("=" * 60)

        return {
            "document_id": doc_id,
            "target_language": self.config.target_lang,
            "model": self.model_name,
            "total_chunks": len(chunks),
            "latency_seconds": trans_result.latency_seconds,
            "saved_paths": saved_paths,
            "translation_result": trans_result.to_dict(),
        }

    def _save_outputs(
        self,
        output_dir: Path,
        doc: Document,
        chunks: List[ProcessedChunk],
        trans_result: TranslationResult,
    ) -> Dict[str, str]:
        """Save outputs in structured reproducible hierarchy."""
        extracted_dir = output_dir / "extracted"
        chunks_dir = output_dir / "chunks"
        trans_dir = output_dir / "translations"

        extracted_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        trans_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save extracted pages JSON and full text
        pages_file = extracted_dir / "pages.json"
        with open(pages_file, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

        full_text_file = extracted_dir / "full_text.txt"
        with open(full_text_file, "w", encoding="utf-8") as f:
            f.write(doc.full_text)

        # 2. Save chunks JSON
        chunks_file = chunks_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

        # 3. Save translation JSON
        trans_filename = f"{self.model_name}_{self.config.target_lang}.json"
        trans_file = trans_dir / trans_filename
        with open(trans_file, "w", encoding="utf-8") as f:
            json.dump(trans_result.to_dict(), f, indent=2, ensure_ascii=False)

        # 4. Save plain translation text
        trans_txt_file = trans_dir / f"{self.model_name}_{self.config.target_lang}.txt"
        with open(trans_txt_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(trans_result.translations))

        # 5. Save experiment metadata
        meta_file = output_dir / "experiment_metadata.json"
        metadata = ReportGenerator.generate_experiment_metadata(
            document_name=doc.document_id,
            source_lang="en",
            target_lang=self.config.target_lang,
            models_evaluated=[self.model_name],
            device_info=self.config.device_info,
            parameters={"model_info": self.model.get_model_info()},
        )
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "output_directory": str(output_dir),
            "pages_file": str(pages_file),
            "chunks_file": str(chunks_file),
            "translation_file": str(trans_file),
            "metadata_file": str(meta_file),
        }


class ExperimentPipeline:
    """
    Orchestrates a comprehensive multi-model benchmark experiment on a legal document:
    P1 (Baseline), P2 (IndicTrans2), P3 (Legal-Aware).
    Evaluates all models, performs error analysis, and records reproducible results.
    """

    def __init__(
        self,
        config: PipelineConfig,
        models: Optional[List[str]] = None,
        reference_path: Optional[Union[str, Path]] = None,
    ):
        self.config = config
        self.models_to_run = models or ["baseline", "indictrans2", "legal_aware"]
        self.reference_path = Path(reference_path) if reference_path else None
        self.evaluator = Evaluator(
            target_lang=config.target_lang,
            legal_terms_dict=config.legal_terms,
            bertscore_model=config.evaluation.get("bertscore_model", "bert-base-multilingual-cased"),
            sacrebleu_tokenize=config.evaluation.get("sacrebleu_tokenize", "flores200"),
            device=config.device_info["device"],
        )

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full multi-model experiment.
        """
        input_path = Path(input_path)
        doc_id = input_path.stem
        out_base = Path(output_dir) if output_dir else self.config.outputs_dir / doc_id

        # 1. Ingestion & Preprocessing (Identical across all models)
        loader = DocumentLoader()
        cleaner = TextCleaner()
        segmenter = TextSegmenter()

        raw_doc = loader.load(input_path, document_id=doc_id)
        cleaned_doc = cleaner.clean_document(raw_doc)
        chunks = segmenter.segment_document(cleaned_doc)
        source_texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        # 2. Check for human reference translations
        references: Optional[List[str]] = None
        ref_file_to_load = self.reference_path
        if not ref_file_to_load:
            lang_dir = "tamil" if self.config.target_lang == "ta" else "malayalam"
            auto_ref = self.config.references_dir / lang_dir / f"{doc_id}.txt"
            if auto_ref.exists():
                ref_file_to_load = auto_ref

        if ref_file_to_load and ref_file_to_load.exists():
            logger.info(f"Loading reference translation from: {ref_file_to_load}")
            try:
                ref_doc = loader.load(ref_file_to_load, document_id=f"{doc_id}_ref")
                ref_cleaned = cleaner.clean_document(ref_doc)
                ref_chunks = segmenter.segment_document(ref_cleaned)
                if len(ref_chunks) == len(chunks):
                    references = [c.text for c in ref_chunks]
                else:
                    # If paragraph split gives exact count
                    raw_paras = [p.strip() for p in ref_cleaned.full_text.split("\n\n") if p.strip()]
                    if len(raw_paras) == len(chunks):
                        references = raw_paras
                    else:
                        # Resample or group reference chunks into exactly len(chunks)
                        logger.info(f"Aligning {len(ref_chunks)} reference chunks to {len(chunks)} source chunks...")
                        ref_full_text = ref_cleaned.full_text
                        # Simple proportional segmentation of reference text
                        total_src_chars = sum(len(c.text) for c in chunks)
                        aligned_refs = []
                        char_pos = 0
                        ref_total_len = len(ref_full_text)

                        for c in chunks:
                            fraction = len(c.text) / max(total_src_chars, 1)
                            take_chars = int(fraction * ref_total_len)
                            chunk_ref = ref_full_text[char_pos : char_pos + take_chars].strip()
                            aligned_refs.append(chunk_ref if chunk_ref else ref_full_text[:50])
                            char_pos += take_chars

                        # Ensure last chunk takes any remaining text
                        if char_pos < ref_total_len and aligned_refs:
                            aligned_refs[-1] = (aligned_refs[-1] + " " + ref_full_text[char_pos:]).strip()

                        references = aligned_refs
            except Exception as e:
                logger.warning(f"Error loading reference translation: {e}")

        # 3. Run all requested models
        model_results: Dict[str, Dict[str, Any]] = {}
        model_eval_reports: Dict[str, EvaluationReport] = {}
        model_translations: Dict[str, List[tuple]] = {}

        for m_name in self.models_to_run:
            logger.info(f"\n>>> Running Model Configuration: {m_name.upper()} <<<")
            model_inst = ModelFactory.create(m_name, config=self.config.to_dict())
            trans_res = model_inst.translate_chunks(chunks, source_lang="en", target_lang=self.config.target_lang)

            # Evaluate
            report = self.evaluator.evaluate(
                candidate_texts=trans_res.translations,
                source_texts=source_texts,
                reference_texts=references,
                chunk_ids=chunk_ids,
                model_name=m_name,
            )

            # Save per-model translations
            trans_dir = out_base / "translations"
            trans_dir.mkdir(parents=True, exist_ok=True)
            with open(trans_dir / f"{m_name}_{self.config.target_lang}.json", "w", encoding="utf-8") as f:
                json.dump(trans_res.to_dict(), f, indent=2, ensure_ascii=False)

            model_results[m_name] = {
                **report.metrics,
                "latency_seconds": round(trans_res.latency_seconds, 2),
                "chars_per_second": round(trans_res.chars_per_second, 1),
            }
            model_eval_reports[m_name] = report
            model_translations[m_name] = [
                (cid, src, tgt) for cid, src, tgt in zip(chunk_ids, source_texts, trans_res.translations)
            ]

        # 4. Save evaluation artifacts
        eval_dir = out_base / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2)

        sentence_metrics_dict = {
            m: r.sentence_metrics for m, r in model_eval_reports.items()
        }
        with open(eval_dir / "sentence_metrics.json", "w", encoding="utf-8") as f:
            json.dump(sentence_metrics_dict, f, indent=2, ensure_ascii=False)

        error_analysis_dict = {
            m: r.error_analysis for m, r in model_eval_reports.items()
        }
        with open(eval_dir / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(error_analysis_dict, f, indent=2, ensure_ascii=False)

        # 5. Create Human Evaluation CSV template
        human_eval_csv = eval_dir / "human_evaluation_template.csv"
        ReportGenerator.create_human_eval_template(
            output_csv_path=human_eval_csv,
            case_id=doc_id,
            language=self.config.target_lang,
            model_translations=model_translations,
        )

        # 6. Save extracted pages & chunks
        (out_base / "extracted").mkdir(parents=True, exist_ok=True)
        with open(out_base / "extracted" / "pages.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_doc.to_dict(), f, indent=2, ensure_ascii=False)

        (out_base / "chunks").mkdir(parents=True, exist_ok=True)
        with open(out_base / "chunks" / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

        # 7. Experiment metadata
        meta_file = out_base / "experiment_metadata.json"
        metadata = ReportGenerator.generate_experiment_metadata(
            document_name=doc_id,
            source_lang="en",
            target_lang=self.config.target_lang,
            models_evaluated=self.models_to_run,
            device_info=self.config.device_info,
            parameters={"total_chunks": len(chunks)},
        )
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 8. Print formatted summary table to console
        summary_str = ReportGenerator.format_terminal_summary(
            doc_id=doc_id,
            target_lang=self.config.target_lang,
            model_results=model_results,
        )
        print("\n" + summary_str + "\n")

        return {
            "document_id": doc_id,
            "target_language": self.config.target_lang,
            "models_evaluated": self.models_to_run,
            "model_results": model_results,
            "output_directory": str(out_base),
            "summary_table": summary_str,
        }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TheerppuX Legal Translation Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input case document path (.pdf, .txt, .docx)")
    parser.add_argument("--target", "-t", required=True, help="Target language code (ta=Tamil, ml=Malayalam)")
    parser.add_argument("--model", "-m", default="indictrans2", help="Translation model configuration")
    parser.add_argument("--output", "-o", default=None, help="Custom output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Hardware compute device")

    args = parser.parse_args()

    overrides = {}
    if args.device != "auto":
        overrides["device"] = args.device

    try:
        config = load_config(target_lang=args.target, overrides=overrides)
        pipeline = TranslationPipeline(config=config, model_name=args.model)
        res = pipeline.run(input_path=args.input, output_dir=args.output)
        print(f"\n[SUCCESS] Completed translation: {res['saved_paths']['translation_file']}")
    except ValueError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

