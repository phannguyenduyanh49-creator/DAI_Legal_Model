"""Parse explicit Direct Catalog procedure codes from user queries."""

from __future__ import annotations

import re

from backend.catalog.schemas import CatalogQuery


# Active Direct Catalog procedure codes use the N.NNNNNN shape. Only use an
# exact-code lookup when the question contains exactly one such standalone
# token; otherwise preserve the existing natural-language title lookup.
_PROCEDURE_CODE = re.compile(r"(?<![A-Za-z0-9])([0-9]\.[0-9]{6})(?![A-Za-z0-9])")


def catalog_query_from_text(
    question: str,
    *,
    domain: str = "legal",
    intent: str = "procedure_lookup",
) -> CatalogQuery:
    matches = list(dict.fromkeys(_PROCEDURE_CODE.findall(question or "")))
    if len(matches) == 1:
        return CatalogQuery(domain=domain, intent=intent, code=matches[0])
    return CatalogQuery(
        domain=domain,
        intent=intent,
        procedure_name=question,
    )
