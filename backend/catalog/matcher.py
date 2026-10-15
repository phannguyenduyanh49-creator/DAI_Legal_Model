from .schemas import CatalogQuery, CatalogResult
from .repository import CatalogRepository


class CatalogMatcher:

    def __init__(self):
        self.repository = CatalogRepository()

    def match(
        self,
        query: CatalogQuery,
    ) -> CatalogResult:

        if query.domain != "legal":
            return CatalogResult(
                matched=False,
                match_type="unsupported_domain",
                confidence=0.0,
                reason="Domain not supported by this catalog",
            )

        record = None
        match_type = "none"

        if query.formality_id:
            record = self.repository.find_by_formality_id(
                query.formality_id
            )
            match_type = "exact_formality_id"

        elif query.code:
            record = self.repository.find_by_code(
                query.code
            )
            match_type = "exact_code"

        elif query.procedure_name:
            record = self.repository.find_by_name(
                query.procedure_name
            )
            match_type = "exact_name"

        else:
            return CatalogResult(
                matched=False,
                match_type="missing_lookup_key",
                confidence=0.0,
                reason="formality_id, code, or procedure_name is required",
            )

        if record is None:
            return CatalogResult(
                matched=False,
                match_type="none",
                confidence=0.0,
                reason="No exact catalog record found",
            )

        return CatalogResult(
            matched=True,
            match_type=match_type,
            confidence=1.0,
            record=record,
        )