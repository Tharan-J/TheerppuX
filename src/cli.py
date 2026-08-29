"""
Command-Line Interface (CLI) for TheerppuX Legal Translation Pipeline.
Provides commands: `translate`, `experiment`, `evaluate`, and `compare`.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

from src.config import SUPPORTED_TARGET_LANGUAGES, load_config
from src.pipeline import TranslationPipeline, ExperimentPipeline
from src.evaluation.evaluator import Evaluator
from src.analysis.report import ReportGenerator
from src.translation.factory import ModelFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TheerppuX.CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="theerppux",
        description="TheerppuX: Multilingual Indian Legal Document Understanding & Translation Pipeline (Stage 1)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. TRANSLATE command
    trans_parser = subparsers.add_parser("translate", help="Translate a single legal document")
    trans_parser.add_argument("--input", "-i", required=True, help="Path to input case document (.pdf, .txt, .docx)")
    trans_parser.add_argument("--target", "-t", required=True, choices=["ta", "ml"], help="Target language code (ta=Tamil, ml=Malayalam)")
    trans_parser.add_argument("--model", "-m", default="indictrans2", choices=["baseline", "indictrans2", "legal_aware"], help="Translation model configuration")
    trans_parser.add_argument("--output", "-o", default=None, help="Custom output directory")
    trans_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Hardware compute device")

    # 2. EXPERIMENT command
    exp_parser = subparsers.add_parser("experiment", help="Run multi-model benchmark experiment on a legal document")
    exp_parser.add_argument("--input", "-i", required=True, help="Path to input case document (.pdf, .txt, .docx)")
    exp_parser.add_argument("--target", "-t", required=True, choices=["ta", "ml"], help="Target language code (ta=Tamil, ml=Malayalam)")
    exp_parser.add_argument("--models", nargs="+", default=["baseline", "indictrans2", "legal_aware"], help="Model configurations to compare")
    exp_parser.add_argument("--references", "-r", default=None, help="Path to human reference translation file")
    exp_parser.add_argument("--output", "-o", default=None, help="Custom output directory")
    exp_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Hardware compute device")

    # 3. EVALUATE command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing translation outputs against human references")
    eval_parser.add_argument("--predictions", "-p", required=True, help="Path to predictions JSON or text directory")
    eval_parser.add_argument("--references", "-r", required=True, help="Path to reference translation file or directory")
    eval_parser.add_argument("--target", "-t", required=True, choices=["ta", "ml"], help="Target language code (ta=Tamil, ml=Malayalam)")
    eval_parser.add_argument("--source", "-s", default=None, help="Path to original source text (for entity and terminology evaluation)")

    # 4. COMPARE command
    comp_parser = subparsers.add_parser("compare", help="Compare multiple model evaluations across a dataset directory")
    comp_parser.add_argument("--dataset", "-d", required=True, help="Path to dataset directory containing outputs/evaluations")
    comp_parser.add_argument("--target", "-t", required=True, choices=["ta", "ml"], help="Target language code (ta=Tamil, ml=Malayalam)")

    return parser.parse_args()


def handle_translate(args):
    """Execute translate command."""
    overrides = {}
    if args.device != "auto":
        overrides["device"] = args.device

    config = load_config(target_lang=args.target, overrides=overrides)
    pipeline = TranslationPipeline(config=config, model_name=args.model)
    res = pipeline.run(input_path=args.input, output_dir=args.output)
    print(f"\n[SUCCESS] Translation completed and saved to: {res['saved_paths']['translation_file']}")


def handle_experiment(args):
    """Execute experiment command."""
    overrides = {}
    if args.device != "auto":
        overrides["device"] = args.device

    config = load_config(target_lang=args.target, overrides=overrides)
    exp = ExperimentPipeline(
        config=config,
        models=args.models,
        reference_path=args.references,
    )
    res = exp.run(input_path=args.input, output_dir=args.output)
    print(f"\n[SUCCESS] Experiment completed. Artifacts saved in: {res['output_directory']}")


def handle_evaluate(args):
    """Execute evaluate command."""
    pred_path = Path(args.predictions)
    ref_path = Path(args.references)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found at: {pred_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"References not found at: {ref_path}")

    config = load_config(target_lang=args.target)
    evaluator = Evaluator(
        target_lang=args.target,
        legal_terms_dict=config.legal_terms,
    )

    # Load predictions
    if pred_path.suffix == ".json":
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            predictions = data.get("translations", [])
    else:
        with open(pred_path, "r", encoding="utf-8") as f:
            predictions = [l.strip() for l in f.read().split("\n\n") if l.strip()]

    # Load references
    with open(ref_path, "r", encoding="utf-8") as f:
        references = [l.strip() for l in f.read().split("\n\n") if l.strip()]

    # Load source if provided
    source_texts = []
    if args.source and Path(args.source).exists():
        with open(args.source, "r", encoding="utf-8") as f:
            source_texts = [l.strip() for l in f.read().split("\n\n") if l.strip()]
    if not source_texts:
        source_texts = [""] * len(predictions)

    report = evaluator.evaluate(
        candidate_texts=predictions,
        source_texts=source_texts,
        reference_texts=references,
        model_name=pred_path.stem,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS : {pred_path.name}")
    print("=" * 60)
    print(json.dumps(report.metrics, indent=2))
    print("=" * 60)


def handle_compare(args):
    """Execute compare command."""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"\nComparing models in dataset directory: {dataset_path} for target '{args.target}'")
    # Scan for metrics.json files
    metric_files = list(dataset_path.glob("**/metrics.json"))
    if not metric_files:
        print(f"No metrics.json files found under {dataset_path}")
        return

    print(f"Found {len(metric_files)} evaluation metric runs.")


def main():
    # If invoked directly with legacy or simple syntax like:
    # python -m src.pipeline --input ... --target ta
    if len(sys.argv) > 1 and sys.argv[1] not in ["translate", "experiment", "evaluate", "compare", "-h", "--help"]:
        # Forward to translate or experiment
        parser = argparse.ArgumentParser(description="Stage 1 Legal Translation Pipeline")
        parser.add_argument("--input", "-i", required=True)
        parser.add_argument("--target", "-t", required=True, choices=["ta", "ml"])
        parser.add_argument("--model", "-m", default="indictrans2")
        parser.add_argument("--output", "-o", default=None)
        direct_args = parser.parse_args()
        direct_args.device = "auto"
        handle_translate(direct_args)
        return

    args = parse_args()
    if not args.command:
        print("Error: No command specified. Run `python -m src.cli --help` for usage.")
        sys.exit(1)

    if args.command == "translate":
        handle_translate(args)
    elif args.command == "experiment":
        handle_experiment(args)
    elif args.command == "evaluate":
        handle_evaluate(args)
    elif args.command == "compare":
        handle_compare(args)


if __name__ == "__main__":
    main()
