from .matcher import CatalogMatcher
from .schemas import CatalogQuery, CatalogResult


class CatalogService:

    def __init__(self):
        self.matcher = CatalogMatcher()

    def lookup(
        self,
        query: CatalogQuery,
    ) -> CatalogResult:

        return self.matcher.match(query)