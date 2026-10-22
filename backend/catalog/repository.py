from itertools import combinations
from typing import Optional

from sqlalchemy import and_, func, or_, select

from .database import SessionLocal
from .models import Procedure
from .schemas import CatalogRecord, CatalogSource


_VIETNAMESE_ACCENTS = {
    **{char: "a" for char in "àáảãạăằắẳẵặâầấẩẫậ"},
    **{char: "e" for char in "èéẻẽẹêềếểễệ"},
    **{char: "i" for char in "ìíỉĩị"},
    **{char: "o" for char in "òóỏõọôồốổỗộơờớởỡợ"},
    **{char: "u" for char in "ùúủũụưừứửữự"},
    **{char: "y" for char in "ỳýỷỹỵ"},
    "đ": "d",
}
_ACCENT_SOURCE = "".join(_VIETNAMESE_ACCENTS)
_ACCENT_TARGET = "".join(_VIETNAMESE_ACCENTS.values())


class CatalogRepository:

    @staticmethod
    def _to_catalog_record(row: Procedure) -> CatalogRecord:
        record_id = str(row.formality_id)
        return CatalogRecord(
            record_id=record_id,
            code=str(row.code or ""),
            procedure_name=str(row.procedure_name or ""),
            target_type=row.target_type,
            category_name=row.category_name,
            description=row.description,
            execution_steps=row.execution_steps,
            execution_methods=row.execution_methods,
            profile_components=row.profile_components,
            requirements_conditions=row.requirements_conditions,
            fees=row.fees,
            legal_basis=row.legal_basis,
            results=row.results,
            target_objects=row.target_objects,
            department_promulgate=row.department_promulgate,
            executing_agencies=row.executing_agencies,
            coordinating_agencies=row.coordinating_agencies,
            source=CatalogSource(
                source_id=record_id,
                source_type=str(row.source or "postgres_catalog"),
                source_url=row.source_url,
            ),
        )

    def find_by_formality_id(
        self,
        formality_id: str,
    ) -> Optional[CatalogRecord]:

        with SessionLocal() as session:
            stmt = (
                select(Procedure)
                .where(
                    Procedure.formality_id == formality_id,
                    Procedure.status == "active",
                )
                .limit(1)
            )

            row = session.execute(stmt).scalar_one_or_none()
            return self._to_catalog_record(row) if row is not None else None

    def find_by_code(
        self,
        code: str,
    ) -> Optional[CatalogRecord]:

        with SessionLocal() as session:
            stmt = (
                select(Procedure)
                .where(
                    Procedure.code == code,
                    Procedure.status == "active",
                )
                .limit(1)
            )

            row = session.execute(stmt).scalar_one_or_none()
            return self._to_catalog_record(row) if row is not None else None

    def find_by_name(
        self,
        procedure_name: str,
    ) -> Optional[CatalogRecord]:

        with SessionLocal() as session:
            stmt = (
                select(Procedure)
                .where(
                    Procedure.procedure_name == procedure_name,
                    Procedure.status == "active",
                )
                .limit(1)
            )

            row = session.execute(stmt).scalar_one_or_none()
            return self._to_catalog_record(row) if row is not None else None

    def find_name_candidates(self, tokens: set[str]) -> list[CatalogRecord]:
        """Find active procedures whose titles contain multiple query terms.

        Accent folding is done in SQL with PostgreSQL ``translate`` so both
        accented and unaccented Vietnamese queries can find the same titles.
        This is candidate generation only; the matcher scores and filters them.
        """
        if not tokens:
            return []
        normalized_name = func.translate(
            func.lower(Procedure.procedure_name), _ACCENT_SOURCE, _ACCENT_TARGET
        )
        title_conditions = [normalized_name.contains(token) for token in sorted(tokens)]
        # Require at least two independent title terms during candidate
        # generation so a common single word does not pull most rows into Python.
        pair_conditions = [
            and_(left, right)
            for left, right in combinations(title_conditions, 2)
        ]
        stmt = (
            select(Procedure)
            .where(
                Procedure.status == "active",
                or_(*pair_conditions),
            )
        )
        with SessionLocal() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._to_catalog_record(row) for row in rows]
