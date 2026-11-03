"""Read-only source-record viewer for human relevance annotation."""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import text

from backend.search.repository import create_search_database_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-key", required=True)
    args = parser.parse_args()
    engine = create_search_database_engine()
    try:
        with engine.connect() as connection:
            row = connection.execute(text("""
                SELECT document_key, domain_id, source_id, document_id, record_id,
                       title, content, source_type, metadata
                FROM domain_search_documents WHERE document_key = :document_key
            """), {"document_key": args.document_key}).mappings().first()
    finally:
        engine.dispose()
    if row is None:
        raise SystemExit("No PostgreSQL search record exists for that document key.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(dict(row), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
