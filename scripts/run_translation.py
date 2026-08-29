#!/usr/bin/env python3
"""
Convenience script to run translation on a legal document.
Usage:
    python scripts/run_translation.py --input data/raw/case_001.pdf --target ta --model indictrans2
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python scripts/run_translation.py --input <path> --target <ta|ml> [--model <baseline|indictrans2|legal_aware>]")
        sys.exit(1)
    # Inject 'translate' subcommand if not provided
    if sys.argv[1] not in ["translate", "-h", "--help"]:
        sys.argv.insert(1, "translate")
    main()
