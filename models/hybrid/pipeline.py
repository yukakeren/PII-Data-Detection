"""Hybrid Rule + ML pipeline for PII detection and anonymization."""

from __future__ import annotations

import csv
import io
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.crf.feature_extraction import sentence_features


TOKEN_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"@?[A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*|"
    r"\d+|[^\w\s]|\s+",
    re.UNICODE,
)

EMAIL_REGEX = re.compile(
    r"(?<!\w)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)"
)
URL_REGEX = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
PHONE_REGEX = re.compile(r"(?<!\w)\+?\d[\d()\-\s]{6,}\d(?!\w)")
ID_REGEX = re.compile(r"(?<!\w)\d{8,16}(?!\w)")

NAME_PREFIXES = [
    ("my", "name", "is"),
    ("i", "am"),
    ("nama", "saya"),
    ("saya",),
    ("this", "is"),
]

USERNAME_PREFIXES = [
    ("username",),
    ("user", "name"),
    ("user",),
    ("akun",),
    ("handle",),
    ("instagram",),
    ("ig",),
]

ADDRESS_PREFIXES = [
    ("address",),
    ("alamat",),
]

STRUCTURED_TYPES = {
    "EMAIL",
    "URL_PERSONAL",
    "PHONE_NUM",
    "ID_NUM",
    "USERNAME",
}

REDACTION_LABELS = {
    "NAME_STUDENT": "[NAME_STUDENT]",
    "EMAIL": "[EMAIL]",
    "USERNAME": "[USERNAME]",
    "ID_NUM": "[ID_NUM]",
    "PHONE_NUM": "[PHONE_NUM]",
    "URL_PERSONAL": "[URL_PERSONAL]",
    "STREET_ADDRESS": "[STREET_ADDRESS]",
}


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


class HybridPIIPipeline:
    """Inference pipeline that combines regex rules and CRF predictions."""

    def __init__(self, crf_model_path: Optional[str] = None):
        self.crf_model_path = Path(crf_model_path) if crf_model_path else REPO_ROOT / "models" / "crf" / "crf_model.pkl"
        self._crf_model = None
        self._crf_error = None
        self._load_crf_model()

    @property
    def crf_available(self) -> bool:
        return self._crf_model is not None

    def _load_crf_model(self) -> None:
        if not self.crf_model_path.exists():
            self._crf_error = f"Model not found: {self.crf_model_path.name}"
            return

        try:
            import joblib
        except Exception as exc:  # pragma: no cover - import error depends on env
            self._crf_error = f"joblib unavailable: {exc}"
            return

        try:
            self._crf_model = joblib.load(self.crf_model_path)
        except Exception as exc:  # pragma: no cover - model load depends on env
            self._crf_error = f"failed to load CRF model: {exc}"

    def predict(self, text: str, mode: str = "hybrid") -> Dict:
        token_spans = self.tokenize_with_spans(text)
        content_tokens = [token for token in token_spans if not token.is_space]
        content_texts = [token.text for token in content_tokens]

        ml_labels = self._predict_with_crf(content_texts) if mode in {"hybrid", "crf_only"} else ["O"] * len(content_tokens)
        rule_labels, rule_sources = self._predict_with_rules(text, content_tokens) if mode in {"hybrid", "rules_only", "crf_only"} else (["O"] * len(content_tokens), [""] * len(content_tokens))

        combined_labels, combined_sources = self._merge_predictions(
            ml_labels=ml_labels,
            rule_labels=rule_labels,
            rule_sources=rule_sources,
            mode=mode,
        )

        entities = self.labels_to_entities(text, content_tokens, combined_labels, combined_sources)
        redacted_text = self.redact_text(text, entities)
        token_rows = self.build_token_rows(content_tokens, combined_labels, combined_sources)

        return {
            "tokens": token_rows,
            "entities": [asdict(entity) for entity in entities],
            "redacted_text": redacted_text,
            "prediction_csv": self.rows_to_csv(token_rows),
            "summary": {
                "mode": mode,
                "crf_available": self.crf_available,
                "crf_status": "loaded" if self.crf_available else self._crf_error,
                "entity_count": len(entities),
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

    def _predict_with_crf(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []

        if not self.crf_available:
            return ["O"] * len(tokens)

        try:
            features = [sentence_features(list(tokens))]
            predictions = self._crf_model.predict(features)
        except Exception:
            return ["O"] * len(tokens)

        if not predictions:
            return ["O"] * len(tokens)

        return list(predictions[0])

    def _predict_with_rules(self, text: str, tokens: Sequence[TokenSpan]) -> Tuple[List[str], List[str]]:
        labels = ["O"] * len(tokens)
        sources = [""] * len(tokens)

        self._apply_regex_matches(text, tokens, labels, sources, EMAIL_REGEX, "EMAIL", "rule:email")
        self._apply_regex_matches(text, tokens, labels, sources, URL_REGEX, "URL_PERSONAL", "rule:url")
        self._apply_regex_matches(text, tokens, labels, sources, PHONE_REGEX, "PHONE_NUM", "rule:phone")
        self._apply_regex_matches(text, tokens, labels, sources, ID_REGEX, "ID_NUM", "rule:id")
        self._apply_username_rules(tokens, labels, sources)
        self._apply_name_rules(tokens, labels, sources)
        self._apply_address_rules(tokens, labels, sources)

        return labels, sources

    def _merge_predictions(
        self,
        ml_labels: Sequence[str],
        rule_labels: Sequence[str],
        rule_sources: Sequence[str],
        mode: str,
    ) -> Tuple[List[str], List[str]]:
        if mode == "rules_only":
            return list(rule_labels), list(rule_sources)

        if mode == "crf_only" and self.crf_available:
            return list(ml_labels), ["ml:crf" if label != "O" else "" for label in ml_labels]

        if mode == "crf_only":
            return list(rule_labels), list(rule_sources)

        combined_labels = list(ml_labels)
        combined_sources = ["ml:crf" if label != "O" else "" for label in ml_labels]

        for idx, rule_label in enumerate(rule_labels):
            if rule_label == "O":
                continue

            rule_type = rule_label.split("-", 1)[1]
            if combined_labels[idx] == "O" or rule_type in STRUCTURED_TYPES:
                combined_labels[idx] = rule_label
                combined_sources[idx] = rule_sources[idx]

        return combined_labels, combined_sources

    def _apply_regex_matches(
        self,
        text: str,
        tokens: Sequence[TokenSpan],
        labels: List[str],
        sources: List[str],
        pattern: re.Pattern,
        entity_type: str,
        source: str,
    ) -> None:
        for match in pattern.finditer(text):
            if entity_type == "PHONE_NUM":
                digits = re.sub(r"\D", "", match.group(0))
                if len(digits) < 8:
                    continue

            touched = [
                idx
                for idx, token in enumerate(tokens)
                if token.start < match.end() and token.end > match.start()
            ]

            self._set_entity_labels(labels, sources, touched, entity_type, source)

    def _apply_username_rules(
        self,
        tokens: Sequence[TokenSpan],
        labels: List[str],
        sources: List[str],
    ) -> None:
        lowers = [token.text.lower() for token in tokens]

        for idx, token in enumerate(tokens):
            if labels[idx] != "O":
                continue

            token_text = token.text
            if token_text.startswith("@") and len(token_text) > 2 and "@" not in token_text[1:]:
                self._set_entity_labels(labels, sources, [idx], "USERNAME", "rule:username_handle")
                continue

            for prefix in USERNAME_PREFIXES:
                prefix_len = len(prefix)
                if tuple(lowers[idx:idx + prefix_len]) != prefix:
                    continue

                target_idx = self._skip_fillers(tokens, idx + prefix_len)
                if target_idx is None or labels[target_idx] != "O":
                    continue

                candidate = tokens[target_idx].text
                if (
                    candidate.isalpha()
                    or candidate.isdigit()
                    or "@" in candidate
                    or "." in candidate
                    or "/" in candidate
                ):
                    continue

                if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{1,31}", candidate):
                    self._set_entity_labels(labels, sources, [target_idx], "USERNAME", "rule:username_keyword")
                break

    def _apply_name_rules(
        self,
        tokens: Sequence[TokenSpan],
        labels: List[str],
        sources: List[str],
    ) -> None:
        lowers = [token.text.lower() for token in tokens]

        for idx in range(len(tokens)):
            for prefix in NAME_PREFIXES:
                prefix_len = len(prefix)
                if tuple(lowers[idx:idx + prefix_len]) != prefix:
                    continue

                name_indices: List[int] = []
                cursor = idx + prefix_len
                while cursor < len(tokens) and len(name_indices) < 3:
                    candidate = tokens[cursor].text
                    if not self._looks_like_name_token(candidate, allow_lower=prefix == ("nama", "saya")):
                        break
                    if labels[cursor] != "O":
                        break
                    name_indices.append(cursor)
                    cursor += 1

                if name_indices:
                    self._set_entity_labels(labels, sources, name_indices, "NAME_STUDENT", "rule:name_context")
                break

    def _apply_address_rules(
        self,
        tokens: Sequence[TokenSpan],
        labels: List[str],
        sources: List[str],
    ) -> None:
        lowers = [token.text.lower() for token in tokens]

        for idx in range(len(tokens)):
            for prefix in ADDRESS_PREFIXES:
                prefix_len = len(prefix)
                if tuple(lowers[idx:idx + prefix_len]) != prefix:
                    continue

                cursor = self._skip_fillers(tokens, idx + prefix_len)
                if cursor is None:
                    break

                address_indices: List[int] = []
                while cursor < len(tokens) and len(address_indices) < 8:
                    candidate = tokens[cursor].text
                    if candidate in {".", "!", "?"}:
                        break
                    if labels[cursor] == "O":
                        address_indices.append(cursor)
                    cursor += 1

                if address_indices:
                    self._set_entity_labels(labels, sources, address_indices, "STREET_ADDRESS", "rule:address_context")
                break

    def _skip_fillers(self, tokens: Sequence[TokenSpan], start_idx: int) -> Optional[int]:
        fillers = {":", "-", "is", "adalah", "nya", "saya"}
        idx = start_idx
        while idx < len(tokens) and tokens[idx].text.lower() in fillers:
            idx += 1
        return idx if idx < len(tokens) else None

    def _looks_like_name_token(self, token: str, allow_lower: bool = False) -> bool:
        if not token or not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", token):
            return False
        if token.lower() in {"and", "or", "my", "name", "email", "is", "dan"}:
            return False
        return token[0].isupper() or allow_lower

    def _set_entity_labels(
        self,
        labels: List[str],
        sources: List[str],
        indices: Sequence[int],
        entity_type: str,
        source: str,
    ) -> None:
        if not indices:
            return

        for offset, idx in enumerate(indices):
            labels[idx] = f"{'B' if offset == 0 else 'I'}-{entity_type}"
            sources[idx] = source

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
                current_source = source or current_source
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
    ) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for token, label, source in zip(tokens, labels, sources):
            rows.append(
                {
                    "token": token.text,
                    "pred_label": label,
                    "source": source or "none",
                    "start": token.start,
                    "end": token.end,
                }
            )
        return rows

    def rows_to_csv(self, rows: Sequence[Dict[str, str]], document_id: int = 0) -> str:
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
