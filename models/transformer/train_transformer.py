"""Training script for transformer-based PII token classification."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader
from src.evaluate import Evaluator, save_metrics


DEFAULT_BASE_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-v3-small",
}

DEFAULT_OUTPUT_DIRS = {
    "distilbert": REPO_ROOT / "models" / "transformer" / "distilbert-pii",
    "deberta": REPO_ROOT / "models" / "transformer" / "deberta-pii",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer for PII token classification")
    parser.add_argument(
        "--model-key",
        choices=["distilbert", "deberta"],
        default="distilbert",
        help="Model family to fine-tune",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Hugging Face base checkpoint override",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train/eval batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Maximum number of word-level tokens per training chunk",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for scheduler",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Number of training steps between logs",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints kept on disk",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["default", "balanced", "sqrt_balanced"],
        default="sqrt_balanced",
        help="Loss reweighting strategy to handle severe label imbalance",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint path for resuming training",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU training even if CUDA is available",
    )
    parser.add_argument(
        "--max-train-chunks",
        type=int,
        default=None,
        help="Optional cap on number of train chunks for faster experiments",
    )
    parser.add_argument(
        "--max-val-chunks",
        type=int,
        default=None,
        help="Optional cap on number of validation chunks for faster experiments",
    )
    parser.add_argument(
        "--max-test-chunks",
        type=int,
        default=None,
        help="Optional cap on number of test chunks for faster experiments",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Prefix for saved prediction and metrics files; defaults to model key",
    )
    return parser.parse_args()


def load_optional_dependencies():
    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Training transformer membutuhkan dependency tambahan. "
            "Install dulu `torch`, `transformers`, dan `accelerate`."
        ) from exc

    return {
        "torch": torch,
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForTokenClassification": DataCollatorForTokenClassification,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def load_label_maps(config_path: Path) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    with config_path.open() as handle:
        schema = json.load(handle)
    ordered = sorted(schema.items(), key=lambda item: item[1])
    label_to_id = {label: idx for label, idx in ordered}
    id_to_label = {idx: label for label, idx in ordered}
    label_list = [label for label, _ in ordered]
    return label_to_id, id_to_label, label_list


def normalize_docs(docs: Sequence[Dict]) -> List[Dict]:
    normalized = []
    for doc in docs:
        tokens = []
        labels = []
        for token, label in zip(doc["tokens"], doc["labels"]):
            if token.strip() == "":
                continue
            tokens.append(token)
            labels.append(label)
        normalized.append(
            {
                "document": doc["document"],
                "tokens": tokens,
                "labels": labels,
            }
        )
    return normalized


def chunk_docs(docs: Sequence[Dict], chunk_size: int) -> List[Dict]:
    chunked = []
    for doc in docs:
        tokens = doc["tokens"]
        labels = doc["labels"]

        for start in range(0, len(tokens), chunk_size):
            end = start + chunk_size
            chunked.append(
                {
                    "document": doc["document"],
                    "tokens": tokens[start:end],
                    "labels": labels[start:end],
                }
            )

    return chunked


def maybe_limit_docs(docs: Sequence[Dict], max_docs: int | None) -> List[Dict]:
    if max_docs is None:
        return list(docs)
    return list(docs[:max_docs])


def build_loss_weights(
    docs: Sequence[Dict],
    label_list: Sequence[str],
    torch_module,
    loss_mode: str,
):
    if loss_mode == "default":
        return None

    counts = Counter(
        label
        for doc in docs
        for label in doc["labels"]
    )
    o_count = max(counts.get("O", 1), 1)
    weights = []

    for label in label_list:
        if label == "O":
            weight = 1.0
        else:
            label_count = max(counts.get(label, 1), 1)
            ratio = o_count / label_count
            if loss_mode == "balanced":
                weight = min(ratio, 50.0)
            else:
                weight = min(ratio ** 0.5, 10.0)
        weights.append(weight)

    mean_weight = sum(weights) / max(len(weights), 1)
    normalized = [weight / mean_weight for weight in weights]
    return torch_module.tensor(normalized, dtype=torch_module.float32)


def build_dataset_class(torch_module):
    class TokenDataset(torch_module.utils.data.Dataset):
        def __init__(self, docs, tokenizer, label_to_id, max_length):
            self.docs = docs
            self.tokenizer = tokenizer
            self.label_to_id = label_to_id
            self.max_length = max_length

        def __len__(self):
            return len(self.docs)

        def __getitem__(self, idx):
            doc = self.docs[idx]
            encoding = self.tokenizer(
                doc["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

            word_ids = encoding.word_ids()
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[doc["labels"][word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            item = {
                key: torch_module.tensor(value)
                for key, value in encoding.items()
            }
            item["labels"] = torch_module.tensor(label_ids)
            return item

    return TokenDataset


def build_trainer_class(base_trainer, torch_module):
    class WeightedTokenTrainer(base_trainer):
        def __init__(self, *args, loss_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_weights = loss_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = {
                key: value
                for key, value in inputs.items()
                if key != "labels"
            }
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss_fct = torch_module.nn.CrossEntropyLoss(
                weight=self.loss_weights.to(logits.device) if self.loss_weights is not None else None,
                ignore_index=-100,
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTokenTrainer


def compute_metrics_factory(id_to_label: Dict[int, str]):
    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        pred_ids = predictions.argmax(axis=-1)
        y_true = []
        y_pred = []

        for pred_row, label_row in zip(pred_ids, labels):
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                y_true.append(id_to_label[int(label_id)])
                y_pred.append(id_to_label[int(pred_id)])

        metrics = Evaluator.evaluate(y_true, y_pred)
        return {
            "token_precision": metrics["token_level"]["precision"],
            "token_recall": metrics["token_level"]["recall"],
            "token_f1": metrics["token_level"]["f1"],
            "entity_precision": metrics["entity_level"]["precision"],
            "entity_recall": metrics["entity_level"]["recall"],
            "entity_f1": metrics["entity_level"]["f1"],
        }

    return compute_metrics


def get_model_device(model):
    return next(model.parameters()).device


def predict_document_labels(model, tokenizer, tokens, id_to_label, torch_module, max_length):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    word_ids = encoding.word_ids(batch_index=0)
    device = get_model_device(model)
    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch_module.no_grad():
        outputs = model(**encoding)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    labels = ["O"] * len(tokens)
    seen = set()

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        labels[word_idx] = id_to_label[int(pred_ids[token_idx])]
        seen.add(word_idx)

    return labels


def evaluate_docs(model, tokenizer, docs, id_to_label, torch_module, max_length):
    model.eval()
    y_true = []
    y_pred = []
    rows = []

    for doc in docs:
        pred_labels = predict_document_labels(
            model=model,
            tokenizer=tokenizer,
            tokens=doc["tokens"],
            id_to_label=id_to_label,
            torch_module=torch_module,
            max_length=max_length,
        )

        y_true.extend(doc["labels"])
        y_pred.extend(pred_labels)

        for token, true_label, pred_label in zip(doc["tokens"], doc["labels"], pred_labels):
            rows.append(
                {
                    "document_id": doc["document"],
                    "token": token,
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
            )

    metrics = Evaluator.evaluate(y_true, y_pred)
    metrics["total_tokens"] = len(y_true)
    return metrics, rows


def save_prediction_rows(rows: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["document_id", "token", "true_label", "pred_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def compute_warmup_steps(
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    warmup_ratio: float,
) -> int:
    updates_per_epoch = max(
        math.ceil(train_examples / max(batch_size * gradient_accumulation_steps, 1)),
        1,
    )
    total_steps = max(updates_per_epoch * epochs, 1)
    return int(total_steps * warmup_ratio)


def main() -> None:
    args = parse_args()
    deps = load_optional_dependencies()

    model_key = args.model_key
    model_name = args.model_name or DEFAULT_BASE_MODELS[model_key]
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIRS[model_key]

    train_path = REPO_ROOT / "data" / "processed" / "train.json"
    val_path = REPO_ROOT / "data" / "processed" / "val.json"
    test_path = REPO_ROOT / "data" / "processed" / "test_internal.json"

    missing = [str(path) for path in [train_path, val_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "File dataset belum lengkap. Pastikan path ini tersedia: "
            + ", ".join(missing)
        )

    loader = DataLoader()
    train_docs = maybe_limit_docs(
        chunk_docs(
        normalize_docs(loader.load_raw_json(str(train_path))),
        args.chunk_size,
        ),
        args.max_train_chunks,
    )
    val_docs = maybe_limit_docs(
        chunk_docs(
        normalize_docs(loader.load_raw_json(str(val_path))),
        args.chunk_size,
        ),
        args.max_val_chunks,
    )
    test_docs = maybe_limit_docs(
        chunk_docs(
        normalize_docs(loader.load_raw_json(str(test_path))),
        args.chunk_size,
        ),
        args.max_test_chunks,
    )

    label_to_id, id_to_label, label_list = load_label_maps(
        REPO_ROOT / "configs" / "label_schema.json"
    )

    torch_module = deps["torch"]
    use_cpu = args.force_cpu or not torch_module.cuda.is_available()
    device_name = "cpu" if use_cpu else torch_module.cuda.get_device_name(0)
    TokenDataset = build_dataset_class(torch_module)
    TrainerClass = build_trainer_class(deps["Trainer"], torch_module)
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    model = deps["AutoModelForTokenClassification"].from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        report_to="none",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=use_cpu,
        do_train=True,
        do_eval=True,
    )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=deps["DataCollatorForTokenClassification"](tokenizer),
        compute_metrics=compute_metrics_factory(id_to_label),
        loss_weights=loss_weights,
    )

    print(
        "Training configuration:",
        json.dumps(
            {
                "model_key": model_key,
                "base_model": model_name,
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
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    trained_model = trainer.model

    val_metrics, _ = evaluate_docs(
        trained_model,
        tokenizer,
        val_docs,
        id_to_label,
        torch_module,
        args.max_length,
    )
    test_metrics, test_rows = evaluate_docs(
        trained_model,
        tokenizer,
        test_docs,
        id_to_label,
        torch_module,
        args.max_length,
    )

    model_alias = args.artifact_prefix or model_key
    metrics_payload = {
        "model": model_alias,
        "base_model": model_name,
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
            "train_metrics": train_result.metrics,
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
        },
    }

    predictions_path = REPO_ROOT / "results" / "predictions" / f"{model_alias}_predictions.csv"
    metrics_path = REPO_ROOT / "results" / "metrics" / f"{model_alias}_metrics.json"
    run_summary_path = output_dir / "run_summary.json"

    save_prediction_rows(test_rows, predictions_path)
    save_metrics(metrics_payload, str(metrics_path))
    save_json(
        {
            "model_key": model_key,
            "base_model": model_name,
            "device": device_name,
            "output_dir": str(output_dir),
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "training_args": vars(args),
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
        },
        run_summary_path,
    )

    print(f"Model saved to: {output_dir}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Run summary saved to: {run_summary_path}")


if __name__ == "__main__":
    main()
