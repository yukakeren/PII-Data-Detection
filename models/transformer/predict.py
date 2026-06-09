"""CLI entrypoint for transformer-based PII inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import TransformerPIIPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Local transformer PII prediction")
    parser.add_argument("text", help="Text to analyze")
    parser.add_argument(
        "--model-key",
        choices=["distilbert", "deberta"],
        default="distilbert",
        help="Which local model directory to use",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence for non-O predictions",
    )
    args = parser.parse_args()

    pipeline = TransformerPIIPipeline()
    result = pipeline.predict(
        args.text,
        model_key=args.model_key,
        min_confidence=args.min_confidence,
    )

    print("Summary:")
    print(json.dumps(result["summary"], indent=2))
    print("\nEntities:")
    print(json.dumps(result["entities"], indent=2))
    print("\nRedacted Text:")
    print(result["redacted_text"])


if __name__ == "__main__":
    main()
