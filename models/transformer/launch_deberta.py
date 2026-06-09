"""Dedicated launcher for DeBERTa training."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "models" / "transformer" / "train_deberta.py"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "logs" / "transformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DeBERTa training job")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--artifact-prefix", default=None)
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--loss-mode", choices=["default", "balanced", "sqrt_balanced"], default="default")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-chunks", type=int, default=None)
    parser.add_argument("--max-val-chunks", type=int, default=None)
    parser.add_argument("--max-test-chunks", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    return parser.parse_args()


def build_run_name(explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"deberta_{timestamp}"


def build_command(args: argparse.Namespace, run_name: str) -> list[str]:
    command = [
        args.python_bin,
        str(TRAIN_SCRIPT),
        "--model-key",
        "deberta",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--chunk-size",
        str(args.chunk_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--loss-mode",
        args.loss_mode,
        "--artifact-prefix",
        args.artifact_prefix or run_name,
        "--save-total-limit",
        str(args.save_total_limit),
        "--max-grad-norm",
        str(args.max_grad_norm),
    ]

    optional_pairs = {
        "weight_decay": "--weight-decay",
        "warmup_ratio": "--warmup-ratio",
        "logging_steps": "--logging-steps",
        "seed": "--seed",
        "max_train_chunks": "--max-train-chunks",
        "max_val_chunks": "--max-val-chunks",
        "max_test_chunks": "--max-test-chunks",
        "resume_from_checkpoint": "--resume-from-checkpoint",
    }

    for key, flag in optional_pairs.items():
        value = getattr(args, key)
        if value is not None:
            command.extend([flag, str(value)])

    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.force_cpu:
        command.append("--force-cpu")
    if args.use_bf16:
        command.append("--use-bf16")

    return command


def write_metadata(metadata_path: Path, payload: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    run_name = build_run_name(args.run_name)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    metadata_path = log_dir / f"{run_name}.json"
    command = build_command(args, run_name)
    command_text = shlex.join(command)

    metadata = {
        "run_name": run_name,
        "preset": "deberta",
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
