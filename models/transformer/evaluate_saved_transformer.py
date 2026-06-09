"""Evaluate a saved transformer token-classification model without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader
from src.evaluate import save_metrics
from models.transformer.train_transformer import (
    DEFAULT_OUTPUT_DIRS,
    DEFAULT_BASE_MODELS,
    evaluate_docs,
    load_label_maps,
    load_optional_dependencies,
    maybe_limit_docs,
    chunk_docs,
    normalize_docs,
    save_json,
    save_prediction_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved local transformer model")
    parser.add_argument(
        "--model-key",
        choices=["distilbert", "deberta"],
        default="distilbert",
        help="Model family used for naming defaults",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing the fine-tuned model; defaults to the standard output dir",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Prefix for saved predictions and metrics; defaults to <model-key>_manual_eval",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for evaluation tokenization",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Word-level chunk size used to prepare evaluation docs",
    )
    parser.add_argument("--max-val-chunks", type=int, default=None)
    parser.add_argument("--max-test-chunks", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = load_optional_dependencies()
    torch_module = deps["torch"]
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_OUTPUT_DIRS[args.model_key]
    artifact_prefix = args.artifact_prefix or f"{args.model_key}_manual_eval"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    loader = DataLoader()
    val_docs = maybe_limit_docs(
        chunk_docs(
            normalize_docs(loader.load_raw_json(str(REPO_ROOT / "data" / "processed" / "val.json"))),
            args.chunk_size,
        ),
        args.max_val_chunks,
    )
    test_docs = maybe_limit_docs(
        chunk_docs(
            normalize_docs(loader.load_raw_json(str(REPO_ROOT / "data" / "processed" / "test_internal.json"))),
            args.chunk_size,
        ),
        args.max_test_chunks,
    )

    label_to_id, id_to_label, _ = load_label_maps(REPO_ROOT / "configs" / "label_schema.json")
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_dir)
    model = deps["AutoModelForTokenClassification"].from_pretrained(
        model_dir,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    if not args.force_cpu and torch_module.cuda.is_available():
        model.to("cuda")

    val_metrics, _ = evaluate_docs(
        model=model,
        tokenizer=tokenizer,
        docs=val_docs,
        id_to_label=id_to_label,
        torch_module=torch_module,
        max_length=args.max_length,
    )
    test_metrics, test_rows = evaluate_docs(
        model=model,
        tokenizer=tokenizer,
        docs=test_docs,
        id_to_label=id_to_label,
        torch_module=torch_module,
        max_length=args.max_length,
    )

    metrics_payload = {
        "model": artifact_prefix,
        "base_model": DEFAULT_BASE_MODELS[args.model_key],
        "total_tokens": test_metrics["total_tokens"],
        "token_level": test_metrics["token_level"],
        "entity_level": test_metrics["entity_level"],
        "validation_metrics": val_metrics,
        "evaluation": {
            "model_dir": str(model_dir),
            "device": "cpu" if args.force_cpu or not torch_module.cuda.is_available() else torch_module.cuda.get_device_name(0),
            "max_length": args.max_length,
            "chunk_size": args.chunk_size,
            "val_chunks": len(val_docs),
            "test_chunks": len(test_docs),
        },
    }

    predictions_path = REPO_ROOT / "results" / "predictions" / f"{artifact_prefix}_predictions.csv"
    metrics_path = REPO_ROOT / "results" / "metrics" / f"{artifact_prefix}_metrics.json"
    summary_path = model_dir / "manual_eval_summary.json"

    save_prediction_rows(test_rows, predictions_path)
    save_metrics(metrics_payload, str(metrics_path))
    save_json(
        {
            "model_dir": str(model_dir),
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "artifact_prefix": artifact_prefix,
        },
        summary_path,
    )

    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
