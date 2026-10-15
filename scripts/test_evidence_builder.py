import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.catalog.repository import CatalogRepository
from backend.evidence.builder import EvidenceBuilder


def main():

    repository = CatalogRepository()


    procedure = repository.find_by_code(
        "1.002421"
    )

    if procedure is None:
        print("Procedure not found")
        return


    builder = EvidenceBuilder(
        min_score=0.3,
        max_evidences=10,
    )

    bundle = builder.build_from_catalog(
        procedures=[procedure],
        query="Thủ tục cho phép mang tiền vào tỉnh nội địa",
    )


    print()
    print("=" * 60)
    print("EVIDENCE BUILDER TEST")
    print("=" * 60)

    print(f"Query       : {bundle.query}")
    print(f"Total       : {bundle.total}")
    print()

    print("Metadata:")

    for key, value in bundle.metadata.items():
        print(f"  {key}: {value}")

    print()
    print("Evidence:")

    for index, evidence in enumerate(
        bundle.evidences,
        start=1,
    ):

        print("-" * 60)

        print(f"Rank        : {index}")
        print(f"Evidence ID : {evidence.evidence_id}")
        print(f"Source      : {evidence.source_type}")
        print(f"Title       : {evidence.title}")
        print(f"Score       : {evidence.score}")
        print(f"URL         : {evidence.source_url}")

        print()

        print(
            "Completeness ratio :",
            evidence.metadata.get(
                "completeness_ratio"
            ),
        )

        print(
            "Completeness status:",
            evidence.metadata.get(
                "completeness_status"
            ),
        )

        print()
        print("Fields:")

        for field_name, available in (
            evidence.completeness.items()
        ):

            symbol = "✓" if available else "✗"

            print(
                f"  {symbol} {field_name}"
            )

        print()
        print("Full Content:")
        print(evidence.content)

    print()
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()