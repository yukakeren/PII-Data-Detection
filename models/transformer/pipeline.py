"""Inference pipeline for local transformer token-classification models."""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hybrid.pipeline import REDACTION_LABELS, TOKEN_PATTERN


@dataclass
class TokenSpan:
    text: str
    start: int
    end: int
    is_space: bool = False


@dataclass
class Entity:
    entity_type: str
    text: str
    start: int
    end: int
    source: str


class TransformerPIIPipeline:
    """Load fine-tuned local DistilBERT/DeBERTa models for inference."""

    DEFAULT_MODEL_DIRS = {
        "distilbert": REPO_ROOT / "models" / "transformer" / "distilbert-pii",
        "deberta": REPO_ROOT / "models" / "transformer" / "deberta-pii",
    }

    def __init__(self, model_dirs: Optional[Dict[str, str]] = None, chunk_size: int = 128):
        self.model_dirs = {
            key: Path(value)
            for key, value in (model_dirs or self.DEFAULT_MODEL_DIRS).items()
        }
        self.chunk_size = chunk_size
        self._loaded = {}
        self._status = {}

    def _load_backend(self, model_key: str):
        if model_key in self._loaded:
            return self._loaded[model_key]

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except Exception as exc:
            self._status[model_key] = f"missing dependency: {exc}"
            self._loaded[model_key] = None
            return None

        model_dir = self.model_dirs.get(model_key)
        if model_dir is None:
            self._status[model_key] = f"unknown model key: {model_key}"
            self._loaded[model_key] = None
            return None

        if not model_dir.exists():
            self._status[model_key] = f"model directory not found: {model_dir}"
            self._loaded[model_key] = None
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            model.eval()
        except Exception as exc:
            self._status[model_key] = f"failed to load model: {exc}"
            self._loaded[model_key] = None
            return None

        backend = {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": model,
        }
        self._status[model_key] = "loaded"
        self._loaded[model_key] = backend
        return backend

    def predict(
        self,
        text: str,
        model_key: str = "distilbert",
        min_confidence: float = 0.0,
    ) -> Dict:
        token_spans = self.tokenize_with_spans(text)
        content_tokens = [token for token in token_spans if not token.is_space]
        backend = self._load_backend(model_key)

        if not content_tokens:
            return self._empty_result(text, model_key, "empty input")

        if backend is None:
            return self._empty_result(text, model_key, self._status.get(model_key, "unavailable"))

        torch = backend["torch"]
        tokenizer = backend["tokenizer"]
        model = backend["model"]
        labels, sources, confidences = self._predict_chunked(
            tokens=content_tokens,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            model_key=model_key,
            min_confidence=min_confidence,
        )

        entities = self.labels_to_entities(text, content_tokens, labels, sources)
        token_rows = self.build_token_rows(content_tokens, labels, sources, confidences)

        return {
            "tokens": token_rows,
            "entities": [asdict(entity) for entity in entities],
            "redacted_text": self.redact_text(text, entities),
            "prediction_csv": self.rows_to_csv(token_rows),
            "summary": {
                "model_key": model_key,
                "transformer_available": True,
                "status": self._status[model_key],
                "entity_count": len(entities),
                "chunk_count": max(1, (len(content_tokens) + self.chunk_size - 1) // self.chunk_size),
            },
        }

    def _empty_result(self, text: str, model_key: str, status: str) -> Dict:
        return {
            "tokens": [],
            "entities": [],
            "redacted_text": text,
            "prediction_csv": self.rows_to_csv([]),
            "summary": {
                "model_key": model_key,
                "transformer_available": False,
                "status": status,
                "entity_count": 0,
            },
        }

    def tokenize_with_spans(self, text: str) -> List[TokenSpan]:
        tokens: List[TokenSpan] = []
        for match in TOKEN_PATTERN.finditer(text):
            token_text = match.group(0)
            tokens.append(
                TokenSpan(
                    text=token_text,
                    start=match.start(),
                    end=match.end(),
                    is_space=token_text.isspace(),
                )
            )
        return tokens

    def _predict_chunked(
        self,
        tokens: Sequence[TokenSpan],
        tokenizer,
        model,
        torch_module,
        model_key: str,
        min_confidence: float,
    ) -> tuple[List[str], List[str], List[float]]:
        labels: List[str] = []
        sources: List[str] = []
        confidences: List[float] = []

        for start in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[start:start + self.chunk_size]
            chunk_labels, chunk_sources, chunk_confidences = self._predict_single_chunk(
                tokens=chunk_tokens,
                tokenizer=tokenizer,
                model=model,
                torch_module=torch_module,
                model_key=model_key,
                min_confidence=min_confidence,
            )
            labels.extend(chunk_labels)
            sources.extend(chunk_sources)
            confidences.extend(chunk_confidences)

        return labels, sources, confidences

    def _predict_single_chunk(
        self,
        tokens: Sequence[TokenSpan],
        tokenizer,
        model,
        torch_module,
        model_key: str,
        min_confidence: float,
    ) -> tuple[List[str], List[str], List[float]]:
        words = [token.text for token in tokens]
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )

        with torch_module.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0]
        probabilities = torch_module.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).tolist()
        word_ids = encoding.word_ids(batch_index=0)
        id_to_label = getattr(model.config, "id2label", {})

        labels = ["O"] * len(words)
        sources = [""] * len(words)
        confidences = [0.0] * len(words)
        seen_word_ids = set()

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen_word_ids:
                continue

            label_id = pred_ids[token_idx]
            label = id_to_label.get(label_id, "O")
            confidence = float(probabilities[token_idx][label_id].item())
            if label != "O" and confidence < min_confidence:
                label = "O"

            labels[word_idx] = label
            sources[word_idx] = f"ml:{model_key}"
            confidences[word_idx] = confidence
            seen_word_ids.add(word_idx)

        return labels, sources, confidences

    def labels_to_entities(
        self,
        text: str,
        tokens: Sequence[TokenSpan],
        labels: Sequence[str],
        sources: Sequence[str],
    ) -> List[Entity]:
        entities: List[Entity] = []
        current_type = None
        current_start = 0
        current_end = 0
        current_source = ""

        for token, label, source in zip(tokens, labels, sources):
            if label == "O":
                if current_type is not None:
                    entities.append(
                        Entity(
                            entity_type=current_type,
                            text=text[current_start:current_end],
                            start=current_start,
                            end=current_end,
                            source=current_source,
                        )
                    )
                    current_type = None
                continue

            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or entity_type != current_type:
                if current_type is not None:
                    entities.append(
                        Entity(
                            entity_type=current_type,
                            text=text[current_start:current_end],
                            start=current_start,
                            end=current_end,
                            source=current_source,
                        )
                    )
                current_type = entity_type
                current_start = token.start
                current_end = token.end
                current_source = source
            else:
                current_end = token.end

        if current_type is not None:
            entities.append(
                Entity(
                    entity_type=current_type,
                    text=text[current_start:current_end],
                    start=current_start,
                    end=current_end,
                    source=current_source,
                )
            )

        return entities

    def redact_text(self, text: str, entities: Sequence[Entity]) -> str:
        if not entities:
            return text

        redacted = text
        for entity in sorted(entities, key=lambda item: item.start, reverse=True):
            replacement = REDACTION_LABELS.get(entity.entity_type, "[PII]")
            redacted = redacted[:entity.start] + replacement + redacted[entity.end:]
        return redacted

    def build_token_rows(
        self,
        tokens: Sequence[TokenSpan],
        labels: Sequence[str],
        sources: Sequence[str],
        confidences: Sequence[float],
    ) -> List[Dict]:
        rows = []
        for token, label, source, confidence in zip(tokens, labels, sources, confidences):
            rows.append(
                {
                    "token": token.text,
                    "pred_label": label,
                    "source": source or "none",
                    "confidence": round(confidence, 4),
                    "start": token.start,
                    "end": token.end,
                }
            )
        return rows

    def rows_to_csv(self, rows: Sequence[Dict], document_id: int = 0) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["document_id", "token", "true_label", "pred_label"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "document_id": document_id,
                    "token": row["token"],
                    "true_label": "",
                    "pred_label": row["pred_label"],
                }
            )
        return output.getvalue()
