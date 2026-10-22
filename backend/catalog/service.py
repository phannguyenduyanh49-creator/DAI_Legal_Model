from .matcher import CatalogMatcher
from .schemas import CatalogQuery, CatalogResult
from sqlalchemy.exc import OperationalError


class CatalogService:

    def __init__(self):
        self.matcher = CatalogMatcher()

    def lookup(
        self,
        query: CatalogQuery,
    ) -> CatalogResult:

        return self.matcher.match(query)

    def lookup_optional(
        self,
        query: CatalogQuery,
    ) -> tuple[CatalogResult | None, str]:
        """Return an explicit unavailable state for database connectivity errors.

        Catalog is an optional evidence source; callers may continue with other
        retrieval sources when PostgreSQL is unavailable. Unexpected errors
        still propagate so programming and data errors are not hidden.
        """
        try:
            result = self.lookup(query)
        except OperationalError:
            return None, "unavailable"
        if result is None:
            return None, "unknown"
        return result, "completed"
