"""Experimental local Transformers NLI adapter; never wired into production.

The chosen checkpoint is a multilingual general-domain NLI model. It is a
research adapter only until evaluated on this project's labeled legal set.
Weights are loaded only from an existing local cache and CPU is the default.
"""

from __future__ import annotations

import time
import math
from typing import Any, Sequence

from backend.answer.schemas import Claim
from backend.evidence.schemas import Evidence
from backend.verifier.scoring import GroundingAssessment, ScorerUnavailableError


EXPERIMENTAL_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def _normalize_relation(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    value = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "entailment": "supported",
        "entailed": "supported",
        "supported": "supported",
        "contradiction": "contradicted",
        "contradicted": "contradicted",
        "neutral": "unsupported",
        "not_entailment": "unsupported",
        "non_entailment": "unsupported",
        "unsupported": "unsupported",
        "partially_supported": "partially_supported",
        "partial_support": "partially_supported",
    }
    return aliases.get(value)


class TransformersNliGroundingScorer:
    """Map a three-way sequence-classification NLI model to grounding labels.

    Each citation is evaluated independently with Evidence content as premise
    and the claim as hypothesis. The adapter never sees uncited Evidence. A
    neutral result is unsupported (not evidence of partial entailment).
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_id: str = EXPERIMENTAL_MODEL_ID,
        device: str = "cpu",
        max_evidence_items: int = 8,
    ) -> None:
        if tokenizer is None or model is None:
            raise ScorerUnavailableError(
                "NLI model or tokenizer is not available",
                reason_code="NLI_MODEL_NOT_READY",
            )
        if max_evidence_items < 1:
            raise ValueError("max_evidence_items must be at least 1")
        config = getattr(model, "config", None)
        raw_labels = getattr(config, "id2label", None)
        if not isinstance(raw_labels, dict) or not raw_labels:
            raise ScorerUnavailableError(
                "NLI checkpoint has no id2label mapping",
                reason_code="NLI_LABEL_SCHEMA_UNAVAILABLE",
            )
        id_to_relation = {}
        for raw_id, label in raw_labels.items():
            try:
                label_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            relation = _normalize_relation(label)
            if relation:
                id_to_relation[label_id] = relation
        if not {"supported", "unsupported", "contradicted"}.issubset(
            set(id_to_relation.values())
        ):
            raise ScorerUnavailableError(
                "NLI checkpoint must expose entailment, neutral, and contradiction labels",
                reason_code="NLI_LABEL_SCHEMA_UNSUPPORTED",
            )

        self.tokenizer = tokenizer
        self.model = model
        self.model_id = model_id
        self.device = device
        self.max_evidence_items = max_evidence_items
        self.max_input_tokens = getattr(config, "max_position_embeddings", None)
        self.id_to_relation = id_to_relation
        try:
            if hasattr(self.model, "to"):
                self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()
        except Exception as exc:
            raise ScorerUnavailableError(
                "NLI model could not be initialized on the selected device",
                reason_code="NLI_MODEL_INITIALIZATION_FAILED",
            ) from exc

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = EXPERIMENTAL_MODEL_ID,
        *,
        device: str = "cpu",
        revision: str | None = None,
        local_files_only: bool = True,
        max_evidence_items: int = 8,
    ) -> "TransformersNliGroundingScorer":
        """Load only on explicit call; cache-only by default, with no network fetch."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            load_args = {
                "revision": revision,
                "local_files_only": local_files_only,
            }
            tokenizer = AutoTokenizer.from_pretrained(model_path, **load_args)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, **load_args
            )
        except Exception as exc:
            raise ScorerUnavailableError(
                "NLI checkpoint is not available in the local cache or could not load",
                reason_code="NLI_MODEL_UNAVAILABLE_OR_LOAD_FAILED",
            ) from exc
        return cls(
            tokenizer,
            model,
            model_id=model_path,
            device=device,
            max_evidence_items=max_evidence_items,
        )

    def evaluate(
        self,
        claim: Claim,
        cited_evidence: Sequence[Evidence],
        *,
        timeout_seconds: float,
    ) -> GroundingAssessment:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and greater than zero")
        if not claim.claim.strip() or not cited_evidence:
            raise ScorerUnavailableError(
                "NLI input has no claim or cited Evidence",
                reason_code="NLI_INPUT_MISSING",
            )
        expected_ids = set(claim.evidence_ids)
        received_ids = [item.evidence_id for item in cited_evidence]
        if (
            not expected_ids
            or any(not isinstance(item, str) or not item for item in received_ids)
            or set(received_ids) != expected_ids
            or len(received_ids) != len(set(received_ids))
        ):
            raise ScorerUnavailableError(
                "Grounding adapter input does not match the claim's citation IDs",
                reason_code="NLI_CITATION_EVIDENCE_MISMATCH",
            )
        if len(cited_evidence) > self.max_evidence_items:
            raise ScorerUnavailableError(
                "Claim cites more Evidence items than the experimental adapter permits",
                reason_code="NLI_EVIDENCE_LIMIT_EXCEEDED",
            )
        if any(
            not isinstance(item.content, str) or not item.content.strip()
            for item in cited_evidence
        ):
            raise ScorerUnavailableError(
                "Cited Evidence contains no text",
                reason_code="NLI_EVIDENCE_TEXT_MISSING",
            )

        started = time.monotonic()
        try:
            import torch

            premises = [item.content for item in cited_evidence]
            hypotheses = [claim.claim] * len(cited_evidence)
            encoded = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise ScorerUnavailableError(
                    "Tokenizer returned no input ids",
                    reason_code="NLI_TOKENIZER_INVALID_OUTPUT",
                )
            input_length = int(input_ids.shape[-1])
            if self.max_input_tokens and input_length > self.max_input_tokens:
                raise ScorerUnavailableError(
                    "NLI pair exceeds model input limit; truncation is disabled",
                    reason_code="NLI_INPUT_EXCEEDS_MODEL_LIMIT",
                )
            encoded = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                output = self.model(**encoded)
            logits = getattr(output, "logits", None)
            if logits is None or len(logits.shape) != 2:
                raise ScorerUnavailableError(
                    "NLI model returned malformed logits",
                    reason_code="NLI_MODEL_INVALID_OUTPUT",
                )
            if logits.shape[0] != len(cited_evidence) or logits.shape[1] != len(self.id_to_relation):
                raise ScorerUnavailableError(
                    "NLI model logits do not match its label schema",
                    reason_code="NLI_MODEL_INVALID_OUTPUT",
                )
            predicted_ids = torch.argmax(logits, dim=1).tolist()
        except ScorerUnavailableError:
            raise
        except Exception as exc:
            raise ScorerUnavailableError(
                "NLI inference failed",
                reason_code="NLI_INFERENCE_FAILED",
            ) from exc

        # The local synchronous Transformers call cannot be safely interrupted
        # mid-kernel. Late results are still discarded and never pass the gate.
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("NLI inference exceeded its deadline")

        relations = [self.id_to_relation.get(int(label_id)) for label_id in predicted_ids]
        if any(relation is None for relation in relations):
            raise ScorerUnavailableError(
                "NLI model selected a label outside the configured schema",
                reason_code="NLI_MODEL_INVALID_LABEL",
            )

        if "contradicted" in relations:
            status = "contradicted"
            reason = "NLI_CONTRADICTION"
        elif "supported" in relations:
            status = "supported"
            reason = "NLI_ENTAILMENT"
        elif "partially_supported" in relations:
            status = "partially_supported"
            reason = "NLI_PARTIAL_SUPPORT"
        else:
            status = "unsupported"
            reason = "NLI_NEUTRAL_NOT_ENTAILED"

        return GroundingAssessment(status, reason, self.model_id)
