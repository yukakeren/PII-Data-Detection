"""Launch long-running transformer training jobs with logging and metadata."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "models" / "transformer" / "train_transformer.py"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "logs" / "transformer"

PRESETS = {
    "distilbert_full": {
        "model_key": "distilbert",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "max_length": 256,
        "chunk_size": 128,
        "gradient_accumulation_steps": 1,
        "loss_mode": "sqrt_balanced",
    },
    "deberta_full": {
        "model_key": "deberta",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "max_length": 256,
        "chunk_size": 128,
        "gradient_accumulation_steps": 2,
        "loss_mode": "sqrt_balanced",
    },
    "distilbert_full_cpu": {
        "model_key": "distilbert",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "max_length": 256,
        "chunk_size": 128,
        "gradient_accumulation_steps": 2,
        "loss_mode": "sqrt_balanced",
        "force_cpu": True,
    },
    "deberta_full_cpu": {
        "model_key": "deberta",
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 3e-5,
        "max_length": 256,
        "chunk_size": 128,
        "gradient_accumulation_steps": 4,
        "loss_mode": "sqrt_balanced",
        "force_cpu": True,
    },
    "distilbert_smoke": {
        "model_key": "distilbert",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "max_length": 128,
        "chunk_size": 128,
        "gradient_accumulation_steps": 1,
        "loss_mode": "sqrt_balanced",
        "max_train_chunks": 32,
        "max_val_chunks": 32,
        "max_test_chunks": 32,
        "force_cpu": True,
    },
    "deberta_smoke": {
        "model_key": "deberta",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "max_length": 128,
        "chunk_size": 128,
        "gradient_accumulation_steps": 1,
        "loss_mode": "sqrt_balanced",
        "max_train_chunks": 16,
        "max_val_chunks": 16,
        "max_test_chunks": 16,
        "force_cpu": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch transformer training job")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="distilbert_full",
        help="Training preset",
    )
    parser.add_argument(
        "--model-key",
        choices=["distilbert", "deberta"],
        default=None,
        help="Optional model override",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name used for logs and artifact prefix",
    )
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON),
        help="Python executable used to start training",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory to save logs and metadata",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional model output directory override",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Optional predictions/metrics prefix override",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run training in the background",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the final command and metadata path without launching",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss-mode", choices=["default", "balanced", "sqrt_balanced"], default=None)
    parser.add_argument("--max-train-chunks", type=int, default=None)
    parser.add_argument("--max-val-chunks", type=int, default=None)
    parser.add_argument("--max-test-chunks", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = dict(PRESETS[args.preset])

    overrides = {
        "model_key": args.model_key,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "chunk_size": args.chunk_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "loss_mode": args.loss_mode,
        "max_train_chunks": args.max_train_chunks,
        "max_val_chunks": args.max_val_chunks,
        "max_test_chunks": args.max_test_chunks,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.force_cpu:
        config["force_cpu"] = True

    return config


def build_run_name(model_key: str, preset: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_key}_{preset}_{timestamp}"


def build_command(args: argparse.Namespace, config: dict, run_name: str) -> list[str]:
    command = [
        args.python_bin,
        str(TRAIN_SCRIPT),
        "--model-key",
        config["model_key"],
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--max-length",
        str(config["max_length"]),
        "--chunk-size",
        str(config["chunk_size"]),
        "--gradient-accumulation-steps",
        str(config["gradient_accumulation_steps"]),
        "--loss-mode",
        config["loss_mode"],
        "--artifact-prefix",
        args.artifact_prefix or run_name,
    ]

    optional_pairs = {
        "weight_decay": "--weight-decay",
        "warmup_ratio": "--warmup-ratio",
        "logging_steps": "--logging-steps",
        "save_total_limit": "--save-total-limit",
        "seed": "--seed",
    }

    for key, flag in optional_pairs.items():
        value = config.get(key)
        if value is not None:
            command.extend([flag, str(value)])

    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])

    for key in ["max_train_chunks", "max_val_chunks", "max_test_chunks", "resume_from_checkpoint"]:
        value = config.get(key)
        if value is not None:
            flag = "--" + key.replace("_", "-")
            command.extend([flag, str(value)])

    if config.get("force_cpu"):
        command.append("--force-cpu")

    return command


def write_metadata(metadata_path: Path, payload: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    config = merge_config(args)
    run_name = build_run_name(config["model_key"], args.preset, args.run_name)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    metadata_path = log_dir / f"{run_name}.json"
    command = build_command(args, config, run_name)
    command_text = shlex.join(command)

    metadata = {
        "run_name": run_name,
        "preset": args.preset,
        "log_path": str(log_path),
        "command": command,
        "command_text": command_text,
        "cwd": str(REPO_ROOT),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": args.output_dir,
        "artifact_prefix": args.artifact_prefix or run_name,
        "status": "dry_run" if args.dry_run else "prepared",
    }

    if args.dry_run:
        write_metadata(metadata_path, metadata)
        print(f"Dry run command:\n{command_text}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Planned log file: {log_path}")
        return

    if args.detach:
        with log_path.open("ab") as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        metadata["status"] = "running"
        metadata["pid"] = process.pid
        write_metadata(metadata_path, metadata)
        print(f"Training started in background with PID {process.pid}")
        print(f"Log file: {log_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Tail command: tail -f {shlex.quote(str(log_path))}")
        return

    with log_path.open("ab") as handle:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    metadata["status"] = "finished" if process.returncode == 0 else "failed"
    metadata["returncode"] = process.returncode
    write_metadata(metadata_path, metadata)
    print(f"Training finished with exit code {process.returncode}")
    print(f"Log file: {log_path}")
    print(f"Metadata: {metadata_path}")
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
