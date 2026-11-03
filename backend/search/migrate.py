"""Apply the search-store schema explicitly; this module never runs on API startup."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.search.repository import create_search_database_engine


MIGRATION = Path(__file__).with_name("migrations") / "0001_domain_search.up.sql"


def apply_migration() -> None:
    statements = [part.strip() for part in MIGRATION.read_text(encoding="utf-8").split(";")]
    engine = create_search_database_engine()
    try:
        with engine.begin() as connection:
            for statement in statements:
                if statement:
                    connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PostgreSQL domain-search tables")
    parser.add_argument("--apply", action="store_true", help="Apply the migration")
    args = parser.parse_args()
    if not args.apply:
        print(f"Dry run only. Would apply {MIGRATION.name}; pass --apply to execute.")
        return
    apply_migration()
    print(f"Applied {MIGRATION.name}")


if __name__ == "__main__":
    main()
