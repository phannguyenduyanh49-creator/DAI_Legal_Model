from typing import Iterable, Optional

from backend.catalog.models import Procedure

from .normalizer import procedure_to_evidence
from .schemas import Evidence, EvidenceBundle


class EvidenceBuilder:

    def __init__(
        self,
        min_score: float = 0.3,
        max_evidences: int = 10,
    ):
        self.min_score = min_score
        self.max_evidences = max_evidences

    # ============================================================
    # PUBLIC API - CATALOG
    # ============================================================

    def build_from_catalog(
        self,
        procedures: Iterable[Procedure],
        query: Optional[str] = None,
    ) -> EvidenceBundle:

        evidences = []

        for procedure in procedures:
            evidence = procedure_to_evidence(procedure)
            evidences.append(evidence)

        return self._build_bundle(
            evidences=evidences,
            query=query,
            sources=["direct_catalog"],
        )

    # ============================================================
    # PUBLIC API - UNIFIED
    # ============================================================

    def build(
        self,
        catalog_evidences: Optional[Iterable[Evidence]] = None,
        rag_evidences: Optional[Iterable[Evidence]] = None,
        mcp_evidences: Optional[Iterable[Evidence]] = None,
        query: Optional[str] = None,
    ) -> EvidenceBundle:

        evidences = []

        if catalog_evidences:
            evidences.extend(catalog_evidences)

        if rag_evidences:
            evidences.extend(rag_evidences)

        if mcp_evidences:
            evidences.extend(mcp_evidences)

        return self._build_bundle(
            evidences=evidences,
            query=query,
        )

    # ============================================================
    # INTERNAL BUILD
    # ============================================================

    def _build_bundle(
        self,
        evidences: Iterable[Evidence],
        query: Optional[str] = None,
        sources: Optional[list[str]] = None,
    ) -> EvidenceBundle:

        evidences = list(evidences)

        # Reject missing IDs and conflicting copies before ordinary deduping.
        evidences, integrity_issues = self._validate_and_deduplicate(evidences)

        # 2. Ranking
        evidences = self._rank(evidences)

        # 3. Filtering
        evidences = self._filter(evidences)

        # 4. Limit
        evidences = evidences[: self.max_evidences]

        detected_sources = self._get_sources(
            evidences
        )

        if sources:
            detected_sources = list(
                dict.fromkeys(
                    sources + detected_sources
                )
            )

        return EvidenceBundle(
            query=query,
            evidences=evidences,
            total=len(evidences),
            metadata={
                "sources": detected_sources,
                "deduplicated": True,
                "integrity_issues": integrity_issues,
                "integrity_excluded_count": len(integrity_issues),
                "ranking": "source_score",
                "min_score": self.min_score,
                "max_evidences": self.max_evidences,
            },
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    def _validate_and_deduplicate(
        self,
        evidences: list[Evidence],
    ) -> tuple[list[Evidence], list[str]]:
        """Drop missing IDs and copies with conflicting source provenance."""
        by_id: dict[str, Evidence] = {}
        conflicting_ids: set[str] = set()
        issues: list[str] = []
        for evidence in evidences:
            evidence_id = str(evidence.evidence_id or "").strip()
            if not evidence_id:
                issues.append("EVIDENCE_ID_MISSING")
                continue
            if evidence_id in conflicting_ids:
                continue
            if evidence.source_type == "domain_upload":
                metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
                domain_id = str(metadata.get("domain_id") or "").strip()
                document_id = str(metadata.get("document_id") or "").strip()
                source_id = str(evidence.source_id or "").strip()
                if not domain_id or not document_id or not source_id:
                    missing = [
                        name for name, value in (
                            ("domain_id", domain_id),
                            ("document_id", document_id),
                            ("source_id", source_id),
                        ) if not value
                    ]
                    if evidence_id in by_id or evidence_id in conflicting_ids:
                        by_id.pop(evidence_id, None)
                        conflicting_ids.add(evidence_id)
                        issues.append(f"EVIDENCE_ID_PROVENANCE_CONFLICT:{evidence_id}")
                    else:
                        # Keep the ID poisoned for this request so a later
                        # duplicate cannot replace the invalid provenance.
                        conflicting_ids.add(evidence_id)
                        issues.append(
                            "EVIDENCE_SOURCE_IDENTITY_MISSING:"
                            + ",".join(missing)
                        )
                    continue
                if (
                    source_id != document_id
                    or not evidence_id.startswith(
                        f"domain:{domain_id}:{document_id}:"
                    )
                ):
                    if evidence_id in by_id or evidence_id in conflicting_ids:
                        by_id.pop(evidence_id, None)
                        conflicting_ids.add(evidence_id)
                        issues.append(f"EVIDENCE_ID_PROVENANCE_CONFLICT:{evidence_id}")
                    else:
                        conflicting_ids.add(evidence_id)
                        issues.append(
                            f"EVIDENCE_SOURCE_IDENTITY_CONFLICT:{evidence_id}"
                        )
                    continue
            if evidence_id in conflicting_ids:
                continue
            prior = by_id.get(evidence_id)
            if prior is None:
                by_id[evidence_id] = evidence
                continue
            identity_fields = (
                "source_type", "source_id", "source_url", "title", "location", "content"
            )
            identity_conflict = any(
                getattr(prior, field_name) != getattr(evidence, field_name)
                for field_name in identity_fields
            )
            prior_metadata = prior.metadata if isinstance(prior.metadata, dict) else {}
            current_metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
            for field_name in (
                "domain_id", "document_id", "record_id", "chunk_index",
                "point_id", "sheet_name", "row_number",
            ):
                if (
                    prior_metadata.get(field_name) is not None
                    and current_metadata.get(field_name) is not None
                    and prior_metadata[field_name] != current_metadata[field_name]
                ):
                    identity_conflict = True
            if identity_conflict:
                by_id.pop(evidence_id, None)
                conflicting_ids.add(evidence_id)
                issues.append(f"EVIDENCE_ID_PROVENANCE_CONFLICT:{evidence_id}")
            elif (evidence.score or 0.0) > (prior.score or 0.0):
                by_id[evidence_id] = evidence

        return list(by_id.values()), issues

    def _deduplicate(
        self,
        evidences: list[Evidence],
    ) -> list[Evidence]:

        seen = set()
        unique = []

        for evidence in evidences:

            # Ưu tiên evidence_id
            key = evidence.evidence_id

            if key in seen:
                continue

            seen.add(key)
            unique.append(evidence)

        return unique

    # ============================================================
    # RANKING
    # ============================================================

    def _rank(
        self,
        evidences: list[Evidence],
    ) -> list[Evidence]:

        return sorted(
            evidences,
            key=lambda evidence: (
                evidence.score
                if evidence.score is not None
                else 0.0
            ),
            reverse=True,
        )

    # ============================================================
    # FILTERING
    # ============================================================

    def _filter(
        self,
        evidences: list[Evidence],
    ) -> list[Evidence]:

        filtered = []

        for evidence in evidences:

            score = (
                evidence.score
                if evidence.score is not None
                else 0.0
            )

            # Loại evidence quá yếu
            if score < self.min_score:
                continue

            # Loại evidence không có nội dung
            if not evidence.content.strip():
                continue

            filtered.append(evidence)

        return filtered

    # ============================================================
    # SOURCE DETECTION
    # ============================================================

    def _get_sources(
        self,
        evidences: Iterable[Evidence],
    ) -> list[str]:

        sources = []

        for evidence in evidences:

            if evidence.source_type not in sources:
                sources.append(
                    evidence.source_type
                )

        return sources
