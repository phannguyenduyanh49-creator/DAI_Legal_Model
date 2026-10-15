from typing import Optional

from sqlalchemy import select

from .database import SessionLocal
from .models import Procedure


class CatalogRepository:

    def find_by_formality_id(
        self,
        formality_id: str,
    ) -> Optional[Procedure]:

        with SessionLocal() as session:
            stmt = (
                select(Procedure)
                .where(
                    Procedure.formality_id == formality_id,
                    Procedure.status == "active",
                )
                .limit(1)
            )

            return session.execute(stmt).scalar_one_or_none()

    def find_by_code(
        self,
        code: str,
    ) -> Optional[Procedure]:

        with SessionLocal() as session:
            stmt = (
                select(Procedure)
                .where(
                    Procedure.code == code,
                    Procedure.status == "active",
                )
                .limit(1)
            )

            return session.execute(stmt).scalar_one_or_none()

    def find_by_name(
        self,
        procedure_name: str,
    ) -> Optional[Procedure]:

        with SessionLocal() as session:
            stmt = (
                select(Procedure)
                .where(
                    Procedure.procedure_name == procedure_name,
                    Procedure.status == "active",
                )
                .limit(1)
            )

            return session.execute(stmt).scalar_one_or_none()