#!/usr/bin/env python3
"""
Dataset Preparation & Verification Utility for TheerppuX.
Scans data/raw/ directories, validates file formats, checks for human references,
and compiles dataset statistics.
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document.loader import DocumentLoader
from src.preprocessing.cleaner import TextCleaner
from tabulate import tabulate


def inspect_dataset(data_dir: Path):
    raw_dir = data_dir / "raw"
    ref_ta_dir = data_dir / "references" / "tamil"
    ref_ml_dir = data_dir / "references" / "malayalam"

    print("=" * 65)
    print("  THEERPPUX DATASET INSPECTOR")
    print("=" * 65)

    if not raw_dir.exists():
        print(f"Raw directory not found: {raw_dir}")
        return

    loader = DocumentLoader(ocr_enabled=False)
    cleaner = TextCleaner()

    files = list(raw_dir.glob("*.*"))
    supported_files = [f for f in files if f.suffix.lower() in [".pdf", ".txt", ".docx"]]

    table_data = []
    headers = ["Case ID", "Format", "Pages", "Words", "Ref (TA)", "Ref (ML)"]

    for file_path in sorted(supported_files):
        doc_id = file_path.stem
        try:
            doc = loader.load(file_path, document_id=doc_id)
            cleaned = cleaner.clean_document(doc)

            has_ta_ref = (ref_ta_dir / f"{doc_id}.txt").exists()
            has_ml_ref = (ref_ml_dir / f"{doc_id}.txt").exists()

            table_data.append(
                [
                    doc_id,
                    doc.file_type.upper(),
                    doc.total_pages,
                    cleaned.total_words,
                    "✓ Available" if has_ta_ref else "✗ Missing",
                    "✓ Available" if has_ml_ref else "✗ Missing",
                ]
            )
        except Exception as e:
            table_data.append([doc_id, file_path.suffix, "Error", str(e), "N/A", "N/A"])

    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 65)
    print(f"Total Case Documents: {len(supported_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TheerppuX Dataset Preparation Utility")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    args = parser.parse_args()

    inspect_dataset(Path(args.data_dir))
