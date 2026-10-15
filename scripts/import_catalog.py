import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk

from backend.catalog.database import SessionLocal
from backend.catalog.models import Procedure


DATASET_PATH = PROJECT_ROOT / "data" / "dichvucong_procedures"


def import_catalog():
    print("Loading dataset...")

    dataset = load_from_disk(str(DATASET_PATH))

    print(f"Dataset loaded: {len(dataset)} records")

    inserted = 0
    updated = 0
    skipped = 0

    with SessionLocal() as session:

        for index, row in enumerate(dataset, start=1):

            formality_id = row.get("formality_id")

            # Skip invalid records
            if not formality_id:
                skipped += 1
                continue

            # Check whether procedure already exists
            procedure = (
                session.query(Procedure)
                .filter(
                    Procedure.formality_id == formality_id
                )
                .first()
            )

            # Create new record
            if procedure is None:

                procedure = Procedure(
                    formality_id=formality_id
                )

                session.add(procedure)

                inserted += 1

            # Existing record
            else:
                updated += 1

            # --------------------------------
            # Basic information
            # --------------------------------

            procedure.code = row.get("code")

            procedure.procedure_name = (
                row.get("procedure_name") or ""
            )

            procedure.target_type = (
                row.get("target_type")
            )

            procedure.category_name = (
                row.get("category_name")
            )

            procedure.department_promulgate = (
                row.get("department_promulgate")
            )

            # --------------------------------
            # Procedure information
            # --------------------------------

            procedure.description = (
                row.get("description")
            )

            procedure.execution_steps = (
                row.get("execution_steps")
            )

            procedure.execution_methods = (
                row.get("execution_methods")
            )

            procedure.profile_components = (
                row.get("profile_components")
            )

            procedure.requirements_conditions = (
                row.get("requirements_conditions")
            )

            # --------------------------------
            # Legal information
            # --------------------------------

            procedure.fees = (
                row.get("fees")
            )

            procedure.legal_basis = (
                row.get("legal_basis")
            )

            procedure.results = (
                row.get("results")
            )

            # --------------------------------
            # Actors
            # --------------------------------

            procedure.target_objects = (
                row.get("target_objects")
            )

            procedure.executing_agencies = (
                row.get("executing_agencies")
            )

            procedure.coordinating_agencies = (
                row.get("coordinating_agencies")
            )

            # --------------------------------
            # Dataset flags
            # --------------------------------

            procedure.is_ministry = bool(
                row.get("is_ministry", False)
            )

            procedure.is_province = bool(
                row.get("is_province", False)
            )

            procedure.is_ward = bool(
                row.get("is_ward", False)
            )

            procedure.is_vertical = bool(
                row.get("is_vertical", False)
            )

            procedure.is_full_process = bool(
                row.get("is_full_process", False)
            )

            # --------------------------------
            # Source / provenance
            # --------------------------------

            procedure.source = (
                row.get("source")
            )

            procedure.source_url = (
                row.get("source_url")
            )

            procedure.content_hash = (
                row.get("content_hash")
            )

            procedure.scraped_at = (
                row.get("scraped_at")
            )

            # --------------------------------
            # Catalog status
            # --------------------------------

            procedure.status = "active"

            # Commit every 500 records
            if index % 500 == 0:
                session.commit()

                print(
                    f"Processed {index}/{len(dataset)} records..."
                )

        # Final commit
        session.commit()

    print()
    print("=" * 50)
    print("IMPORT COMPLETED")
    print("=" * 50)

    print(f"Total records : {len(dataset)}")
    print(f"Inserted      : {inserted}")
    print(f"Updated       : {updated}")
    print(f"Skipped       : {skipped}")
    print("=" * 50)


if __name__ == "__main__":
    import_catalog()