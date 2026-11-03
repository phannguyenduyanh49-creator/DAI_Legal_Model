"""Retry PostgreSQL-to-Elasticsearch outbox events (one pass by default)."""

from __future__ import annotations

import argparse
import time

from backend.search.service import DomainSearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize pending search index events")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--loop", action="store_true", help="Continue polling for pending events")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if not 1 <= args.limit <= 5000:
        parser.error("--limit must be between 1 and 5000")
    if args.interval < 0.5:
        parser.error("--interval must be at least 0.5 seconds")
    service = DomainSearchService()
    while True:
        try:
            indexed = service.sync_pending(args.limit)
            print(f"indexed_document_count={len(indexed)}")
        except Exception as exc:
            # Avoid logging connection strings, response bodies, or credentials.
            print(f"outbox_sync_failed exception_type={type(exc).__name__}")
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
