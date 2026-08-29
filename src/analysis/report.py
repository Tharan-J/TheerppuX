"""
Experiment Reporting and Human Evaluation Analysis Module.
Generates structured metadata, terminal summaries, markdown reports,
and analyzes human evaluation CSVs with inter-rater agreement.
"""

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Handles generation and persistence of experiment reports, metadata,
    and human evaluation CSV templates.
    """

    @staticmethod
    def generate_experiment_metadata(
        document_name: str,
        source_lang: str,
        target_lang: str,
        models_evaluated: List[str],
        device_info: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create structured, reproducible experiment metadata."""
        import platform
        import torch
        import transformers

        return {
            "document": document_name,
            "source_language": source_lang,
            "target_language": target_lang,
            "models_evaluated": models_evaluated,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": device_info.get("device", "unknown"),
            "device_name": device_info.get("device_name", "unknown"),
            "precision": device_info.get("precision", "unknown"),
            "parameters": parameters or {},
            "software_versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "os": platform.platform(),
            },
        }

    @staticmethod
    def format_terminal_summary(
        doc_id: str,
        target_lang: str,
        model_results: Dict[str, Dict[str, Any]],
    ) -> str:
        """Format an aesthetically clean, ASCII terminal summary table."""
        lines = []
        lines.append("=" * 75)
        lines.append(f"  THEERPPUX STAGE 1 — LEGAL TRANSLATION BENCHMARK RESULTS")
        lines.append("=" * 75)
        lines.append(f"  Document : {doc_id}")
        lines.append(f"  Target   : {'Tamil (ta)' if target_lang == 'ta' else 'Malayalam (ml)'}")
        lines.append("-" * 75)

        table_data = []
        headers = [
            "Metric",
            "P1 (Baseline)",
            "P2 (IndicTrans2)",
            "P3 (Legal-Aware)",
        ]

        metrics_to_show = [
            ("BLEU (SacreBLEU)", "bleu"),
            ("chrF++", "chrf_plus_plus"),
            ("ROUGE-L", "rouge_l"),
            ("BERTScore F1", "bertscore_f1"),
            ("Entity F1", "entity_f1"),
            ("Number Accuracy (%)", "number_accuracy"),
            ("Legal Term Accuracy (%)", "legal_term_accuracy"),
            ("Critical Error Rate (%)", "critical_error_rate_pct"),
            ("Latency (sec)", "latency_seconds"),
        ]

        p1_res = model_results.get("baseline", {})
        p2_res = model_results.get("indictrans2", {})
        p3_res = model_results.get("legal_aware", {})

        for label, key in metrics_to_show:
            val1 = p1_res.get(key, "N/A")
            val2 = p2_res.get(key, "N/A")
            val3 = p3_res.get(key, "N/A")
            table_data.append([label, str(val1), str(val2), str(val3)])

        table_str = tabulate(table_data, headers=headers, tablefmt="github")
        lines.append(table_str)
        lines.append("=" * 75)
        return "\n".join(lines)

    @staticmethod
    def create_human_eval_template(
        output_csv_path: Union[str, Path],
        case_id: str,
        language: str,
        model_translations: Dict[str, List[Tuple[str, str, str]]],  # model -> list of (chunk_id, src_text, tgt_text)
    ) -> Path:
        """
        Create a standardized human evaluation CSV template.
        Scale: 1 to 5 for quality metrics.
        """
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "case_id",
            "language",
            "model",
            "chunk_id",
            "source_text",
            "translated_text",
            "factual_accuracy",
            "legal_fidelity",
            "fluency",
            "completeness",
            "terminology_accuracy",
            "severity",
            "comments",
            "annotator_id",
        ]

        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for model_name, items in model_translations.items():
                for chunk_id, src_text, tgt_text in items:
                    writer.writerow(
                        {
                            "case_id": case_id,
                            "language": language,
                            "model": model_name,
                            "chunk_id": chunk_id,
                            "source_text": src_text,
                            "translated_text": tgt_text,
                            "factual_accuracy": "",
                            "legal_fidelity": "",
                            "fluency": "",
                            "completeness": "",
                            "terminology_accuracy": "",
                            "severity": "",
                            "comments": "",
                            "annotator_id": "",
                        }
                    )

        logger.info(f"Created human evaluation template at: {output_csv_path}")
        return output_csv_path

    @staticmethod
    def analyze_human_evaluation_csv(csv_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load completed human evaluation CSV and compute:
        - Mean, Median, Std per dimension
        - Model ranking
        - Inter-rater agreement statistics if multiple annotators are present
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Human evaluation file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        numeric_cols = [
            "factual_accuracy",
            "legal_fidelity",
            "fluency",
            "completeness",
            "terminology_accuracy",
        ]

        # Filter rows with numeric evaluations
        valid_df = df.dropna(subset=numeric_cols).copy()
        for col in numeric_cols:
            valid_df[col] = pd.to_numeric(valid_df[col], errors="coerce")

        valid_df = valid_df.dropna(subset=numeric_cols)

        if valid_df.empty:
            return {"status": "no_completed_evaluations", "message": "CSV contains no evaluated rows."}

        # Compute per-model stats
        stats_by_model = {}
        for model_name, group in valid_df.groupby("model"):
            model_stats = {}
            for col in numeric_cols:
                model_stats[col] = {
                    "mean": round(float(group[col].mean()), 2),
                    "median": round(float(group[col].median()), 2),
                    "std": round(float(group[col].std()), 2),
                }
            stats_by_model[str(model_name)] = model_stats

        # Inter-rater agreement if annotators > 1
        agreement_info = {}
        if "annotator_id" in valid_df.columns and valid_df["annotator_id"].nunique() > 1:
            annotators = valid_df["annotator_id"].dropna().unique()
            agreement_info["num_annotators"] = len(annotators)
            # Calculate correlation across common items
            try:
                pivot = valid_df.pivot_table(index="chunk_id", columns="annotator_id", values="legal_fidelity")
                corr = pivot.corr().values
                np.fill_diagonal(corr, np.nan)
                avg_corr = float(np.nanmean(corr))
                agreement_info["mean_pearson_correlation"] = round(avg_corr, 3)
            except Exception as e:
                agreement_info["error"] = str(e)

        return {
            "total_evaluated_samples": len(valid_df),
            "stats_by_model": stats_by_model,
            "inter_rater_agreement": agreement_info,
        }
