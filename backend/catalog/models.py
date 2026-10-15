from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, Text, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Procedure(Base):
    __tablename__ = "procedures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identity
    formality_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    code: Mapped[str | None] = mapped_column(String(50), index=True)
    procedure_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # Classification
    target_type: Mapped[str | None] = mapped_column(String(100))
    category_name: Mapped[str | None] = mapped_column(Text)
    department_promulgate: Mapped[str | None] = mapped_column(Text)

    # Procedure information
    description: Mapped[str | None] = mapped_column(Text)
    execution_steps: Mapped[str | None] = mapped_column(Text)
    execution_methods: Mapped[str | None] = mapped_column(Text)
    profile_components: Mapped[str | None] = mapped_column(Text)
    requirements_conditions: Mapped[str | None] = mapped_column(Text)

    # Result / legal information
    fees: Mapped[str | None] = mapped_column(Text)
    legal_basis: Mapped[str | None] = mapped_column(Text)
    results: Mapped[str | None] = mapped_column(Text)

    # Actors
    target_objects: Mapped[str | None] = mapped_column(Text)
    executing_agencies: Mapped[str | None] = mapped_column(Text)
    coordinating_agencies: Mapped[str | None] = mapped_column(Text)

    # Dataset flags
    is_ministry: Mapped[bool] = mapped_column(Boolean, default=False)
    is_province: Mapped[bool] = mapped_column(Boolean, default=False)
    is_ward: Mapped[bool] = mapped_column(Boolean, default=False)
    is_vertical: Mapped[bool] = mapped_column(Boolean, default=False)
    is_full_process: Mapped[bool] = mapped_column(Boolean, default=False)

    # Provenance
    source: Mapped[str | None] = mapped_column(String(255))
    source_url: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(128), index=True)
    scraped_at: Mapped[str | None] = mapped_column(String(100))

    # Catalog status
    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )