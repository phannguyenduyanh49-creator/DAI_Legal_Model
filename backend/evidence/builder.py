from typing import Iterable, Optional

from backend.catalog.models import Procedure

from .normalizer import procedure_to_evidence
from .schemas import Evidence, EvidenceBundle


class EvidenceBuilder:
    """
    Build a typed EvidenceBundle from catalog, local RAG, and MCP results.

    Compatibility notes:
    - The public constructor keeps the existing `max_evidences` parameter.
    - `build()` always returns EvidenceBundle, never a plain list.
    """

    def __init__(
        self,
        min_score: float = 0.3,
        max_evidences: int = 10,
        **kwargs,
    ):
        # Accept the singular spelling too, for compatibility with older callers.
        if "max_evidence" in kwargs and kwargs["max_evidence"] is not None:
            max_evidences = kwargs["max_evidence"]

        self.min_score = float(min_score)
        self.max_evidences = max(1, int(max_evidences))

    @staticmethod
    def _as_evidence_list(items) -> list[Evidence]:
        """Normalize optional iterable inputs without splitting strings/dicts."""
        if items is None:
            return []
        if isinstance(items, Evidence):
            return [items]
        if isinstance(items, dict):
            try:
                return [Evidence(**items)]
            except Exception:
                return []
        if isinstance(items, (str, bytes)):
            return []

        try:
            values = list(items)
        except TypeError:
            values = [items]

        result = []
        for item in values:
            if isinstance(item, Evidence):
                result.append(item)
            elif isinstance(item, dict):
                try:
                    result.append(Evidence(**item))
                except Exception:
                    continue
        return result

    # ============================================================
    # PUBLIC API - CATALOG
    # ============================================================

    def build_from_catalog(
        self,
        procedures: Iterable[Procedure],
        query: Optional[str] = None,
    ) -> EvidenceBundle:
        evidences = []

        for procedure in procedures or []:
            try:
                evidence = procedure_to_evidence(procedure)
            except Exception:
                continue
            if evidence is not None:
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
        **kwargs,
    ) -> EvidenceBundle:
        """
        Merge evidence sources and always return EvidenceBundle.

        The keyword aliases allow older call sites to pass `catalog`, `rag`,
        or `mcp` without changing the return type.
        """
        if catalog_evidences is None:
            catalog_evidences = kwargs.get("catalog")
        if rag_evidences is None:
            rag_evidences = kwargs.get("rag")
        if mcp_evidences is None:
            mcp_evidences = kwargs.get("mcp")

        evidences = []
        evidences.extend(self._as_evidence_list(catalog_evidences))
        evidences.extend(self._as_evidence_list(rag_evidences))
        evidences.extend(self._as_evidence_list(mcp_evidences))

        return self._build_bundle(evidences=evidences, query=query)

    # ============================================================
    # INTERNAL BUILD
    # ============================================================

    def _build_bundle(
        self,
        evidences: Iterable[Evidence],
        query: Optional[str] = None,
        sources: Optional[list[str]] = None,
    ) -> EvidenceBundle:
        evidences = self._deduplicate(list(evidences))
        evidences = self._rank(evidences)
        evidences = self._filter(evidences)
        evidences = evidences[: self.max_evidences]

        detected_sources = self._get_sources(evidences)
        if sources:
            detected_sources = list(dict.fromkeys(sources + detected_sources))

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

    @staticmethod
    def _deduplicate(evidences: list[Evidence]) -> list[Evidence]:
        seen = set()
        unique = []

        for evidence in evidences:
            evidence_id = getattr(evidence, "evidence_id", None)
            if not evidence_id:
                continue
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            unique.append(evidence)

        return unique

    # ============================================================
    # RANKING
    # ============================================================

    @staticmethod
    def _rank(evidences: list[Evidence]) -> list[Evidence]:
        return sorted(
            evidences,
            key=lambda evidence: (
                float(evidence.score)
                if getattr(evidence, "score", None) is not None
                else 0.0
            ),
            reverse=True,
        )

    # ============================================================
    # FILTERING
    # ============================================================

    def _filter(self, evidences: list[Evidence]) -> list[Evidence]:
        filtered = []

        for evidence in evidences:
            content = str(getattr(evidence, "content", "") or "").strip()
            if not content:
                continue

            source_type = str(getattr(evidence, "source_type", "") or "").lower()
            source_url = str(getattr(evidence, "source_url", "") or "")
            score_value = getattr(evidence, "score", None)

            try:
                score = float(score_value) if score_value is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0

            # Keep evidence returned by the official National Public Service
            # Portal even if its score is below the local RAG threshold.
            official_mcp = (
                source_type == "mcp"
                and source_url.startswith("https://dichvucong.gov.vn")
            )
            if score < self.min_score and not official_mcp:
                continue

            filtered.append(evidence)

        return filtered

    # ============================================================
    # SOURCE DETECTION
    # ============================================================

    @staticmethod
    def _get_sources(evidences: Iterable[Evidence]) -> list[str]:
        sources = []
        for evidence in evidences:
            source_type = getattr(evidence, "source_type", None)
            if source_type and source_type not in sources:
                sources.append(source_type)
        return sources
