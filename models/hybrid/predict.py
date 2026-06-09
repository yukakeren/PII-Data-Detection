"""CLI entrypoint for hybrid PII prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hybrid import HybridPIIPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Rule + ML PII detection")
    parser.add_argument("text", help="Text to analyze")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "rules_only", "crf_only"],
        default="hybrid",
        help="Inference mode",
    )
    args = parser.parse_args()

    pipeline = HybridPIIPipeline()
    result = pipeline.predict(args.text, mode=args.mode)

    print("Summary:")
    print(json.dumps(result["summary"], indent=2))
    print("\nEntities:")
    print(json.dumps(result["entities"], indent=2))
    print("\nRedacted Text:")
    print(result["redacted_text"])


if __name__ == "__main__":
    main()
