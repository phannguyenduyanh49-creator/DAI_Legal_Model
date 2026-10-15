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

        # 1. Deduplicate
        evidences = self._deduplicate(evidences)

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
                "ranking": "source_score",
                "min_score": self.min_score,
                "max_evidences": self.max_evidences,
            },
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

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