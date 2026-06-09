"""Stable DeBERTa training script."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader
from src.evaluate import save_metrics
from models.transformer.train_transformer import (
    DEFAULT_BASE_MODELS,
    build_dataset_class,
    build_loss_weights,
    chunk_docs,
    compute_metrics_factory,
    compute_warmup_steps,
    evaluate_docs,
    load_label_maps,
    load_optional_dependencies,
    maybe_limit_docs,
    normalize_docs,
    save_json,
    save_prediction_rows,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "transformer" / "deberta-pii"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable DeBERTa training")
    parser.add_argument("--model-key", default="deberta", help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default=DEFAULT_BASE_MODELS["deberta"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact-prefix", default="deberta_stable")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss-mode",
        choices=["default", "balanced", "sqrt_balanced"],
        default="default",
    )
    parser.add_argument("--max-train-chunks", type=int, default=None)
    parser.add_argument("--max-val-chunks", type=int, default=None)
    parser.add_argument("--max-test-chunks", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    return parser.parse_args()


def build_stable_trainer_class(base_trainer, torch_module):
    class StableWeightedTrainer(base_trainer):
        def __init__(self, *args, loss_weights=None, disable_autocast=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_weights = loss_weights
            self.disable_autocast = disable_autocast

        def autocast_smart_context_manager(self, cache_enabled: bool | None = True):
            if self.disable_autocast:
                return contextlib.nullcontext()
            return super().autocast_smart_context_manager(cache_enabled=cache_enabled)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if self.loss_weights is None:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss_fct = torch_module.nn.CrossEntropyLoss(
                weight=self.loss_weights.to(device=logits.device, dtype=logits.dtype),
                ignore_index=-100,
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    return StableWeightedTrainer


def main() -> None:
    args = parse_args()
    deps = load_optional_dependencies()
    torch_module = deps["torch"]
    use_cpu = args.force_cpu or not torch_module.cuda.is_available()
    use_bf16 = bool(args.use_bf16 and not use_cpu and torch_module.cuda.is_bf16_supported())
    device_name = "cpu" if use_cpu else torch_module.cuda.get_device_name(0)

    train_path = REPO_ROOT / "data" / "processed" / "train.json"
    val_path = REPO_ROOT / "data" / "processed" / "val.json"
    test_path = REPO_ROOT / "data" / "processed" / "test_internal.json"
    missing = [str(path) for path in [train_path, val_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))

    loader = DataLoader()
    train_docs = maybe_limit_docs(
        chunk_docs(normalize_docs(loader.load_raw_json(str(train_path))), args.chunk_size),
        args.max_train_chunks,
    )
    val_docs = maybe_limit_docs(
        chunk_docs(normalize_docs(loader.load_raw_json(str(val_path))), args.chunk_size),
        args.max_val_chunks,
    )
    test_docs = maybe_limit_docs(
        chunk_docs(normalize_docs(loader.load_raw_json(str(test_path))), args.chunk_size),
        args.max_test_chunks,
    )

    label_to_id, id_to_label, label_list = load_label_maps(REPO_ROOT / "configs" / "label_schema.json")
    TokenDataset = build_dataset_class(torch_module)
    TrainerClass = build_stable_trainer_class(deps["Trainer"], torch_module)

    tokenizer = deps["AutoTokenizer"].from_pretrained(args.model_name, use_fast=False)
    model = deps["AutoModelForTokenClassification"].from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    if use_bf16:
        model = model.to(dtype=torch_module.bfloat16)
    else:
        model = model.to(dtype=torch_module.float32)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_dataset = TokenDataset(train_docs, tokenizer, label_to_id, args.max_length)
    val_dataset = TokenDataset(val_docs, tokenizer, label_to_id, args.max_length)
    loss_weights = build_loss_weights(
        docs=train_docs,
        label_list=label_list,
        torch_module=torch_module,
        loss_mode=args.loss_mode,
    )
    warmup_steps = compute_warmup_steps(
        train_examples=len(train_dataset),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
    )

    if not use_cpu:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        report_to="none",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=use_cpu,
        bf16=use_bf16,
        fp16=False,
        tf32=(False if use_cpu else True),
        gradient_checkpointing=True,
        do_train=True,
        do_eval=True,
    )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=deps["DataCollatorForTokenClassification"](tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics_factory(id_to_label),
        loss_weights=loss_weights,
        disable_autocast=(not use_cpu and not use_bf16),
    )

    print(
        "Training configuration:",
        json.dumps(
            {
                "model_key": "deberta",
                "base_model": args.model_name,
                "device": device_name,
                "train_chunks": len(train_docs),
                "val_chunks": len(val_docs),
                "test_chunks": len(test_docs),
                "max_length": args.max_length,
                "chunk_size": args.chunk_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "loss_mode": args.loss_mode,
                "warmup_steps": warmup_steps,
                "optimizer": "adamw_torch",
                "max_grad_norm": args.max_grad_norm,
                "bf16": use_bf16,
                "autocast_disabled": bool(not use_cpu and not use_bf16),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    trained_model = trainer.model
    val_metrics, _ = evaluate_docs(trained_model, tokenizer, val_docs, id_to_label, torch_module, args.max_length)
    test_metrics, test_rows = evaluate_docs(trained_model, tokenizer, test_docs, id_to_label, torch_module, args.max_length)

    metrics_payload = {
        "model": args.artifact_prefix,
        "base_model": args.model_name,
        "total_tokens": test_metrics["total_tokens"],
        "token_level": test_metrics["token_level"],
        "entity_level": test_metrics["entity_level"],
        "validation_metrics": val_metrics,
        "training": {
            "device": device_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "chunk_size": args.chunk_size,
            "loss_mode": args.loss_mode,
            "warmup_steps": warmup_steps,
            "train_chunks": len(train_docs),
            "val_chunks": len(val_docs),
            "test_chunks": len(test_docs),
            "seed": args.seed,
            "optim": "adamw_torch",
            "max_grad_norm": args.max_grad_norm,
            "bf16": use_bf16,
            "train_metrics": train_result.metrics,
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
        },
    }

    predictions_path = REPO_ROOT / "results" / "predictions" / f"{args.artifact_prefix}_predictions.csv"
    metrics_path = REPO_ROOT / "results" / "metrics" / f"{args.artifact_prefix}_metrics.json"
    summary_path = output_dir / "run_summary.json"

    save_prediction_rows(test_rows, predictions_path)
    save_metrics(metrics_payload, str(metrics_path))
    save_json(
        {
            "model_key": "deberta",
            "base_model": args.model_name,
            "device": device_name,
            "output_dir": str(output_dir),
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "training_args": vars(args),
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
        },
        summary_path,
    )

    print(f"Model saved to: {output_dir}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
