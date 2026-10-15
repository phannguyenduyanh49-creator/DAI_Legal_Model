from backend.catalog.models import Procedure
from backend.catalog.schemas import CatalogRecord
from .schemas import Evidence


def procedure_to_evidence(
    procedure: Procedure,
) -> Evidence:

    # ============================================================
    # STRUCTURED FIELDS
    # ============================================================

    fields = {
        "formality_id": procedure.formality_id,
        "code": procedure.code,
        "procedure_name": procedure.procedure_name,
        "target_type": procedure.target_type,
        "category_name": procedure.category_name,
        "department_promulgate": (
            procedure.department_promulgate
        ),
        "description": procedure.description,
        "execution_steps": procedure.execution_steps,
        "execution_methods": procedure.execution_methods,
        "profile_components": (
            procedure.profile_components
        ),
        "requirements_conditions": (
            procedure.requirements_conditions
        ),
        "fees": procedure.fees,
        "legal_basis": procedure.legal_basis,
        "results": procedure.results,
        "target_objects": procedure.target_objects,
        "executing_agencies": (
            procedure.executing_agencies
        ),
        "coordinating_agencies": (
            procedure.coordinating_agencies
        ),
    }

    # ============================================================
    # COMPLETENESS
    # ============================================================

    completeness = {}

    for field_name, value in fields.items():

        if value is None:
            completeness[field_name] = False
            continue

        if isinstance(value, str):
            completeness[field_name] = bool(
                value.strip()
            )
            continue

        completeness[field_name] = True

    # ============================================================
    # HUMAN-READABLE CONTENT
    # ============================================================

    content_parts = []

    sections = [
        ("Mã thủ tục", procedure.code),
        ("Tên thủ tục", procedure.procedure_name),
        ("Loại đối tượng", procedure.target_type),
        ("Lĩnh vực", procedure.category_name),
        (
            "Cơ quan ban hành",
            procedure.department_promulgate,
        ),
        ("Mô tả", procedure.description),
        (
            "Đối tượng thực hiện",
            procedure.target_objects,
        ),
        (
            "Trình tự thực hiện",
            procedure.execution_steps,
        ),
        (
            "Cách thức thực hiện",
            procedure.execution_methods,
        ),
        (
            "Thành phần hồ sơ",
            procedure.profile_components,
        ),
        (
            "Yêu cầu, điều kiện",
            procedure.requirements_conditions,
        ),
        ("Phí, lệ phí", procedure.fees),
        ("Kết quả thực hiện", procedure.results),
        (
            "Cơ quan thực hiện",
            procedure.executing_agencies,
        ),
        (
            "Cơ quan phối hợp",
            procedure.coordinating_agencies,
        ),
        (
            "Căn cứ pháp lý",
            procedure.legal_basis,
        ),
    ]

    for label, value in sections:

        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()

        if not value:
            continue

        content_parts.append(
            f"{label}:\n{value}"
        )

    content = "\n\n".join(content_parts)

    # ============================================================
    # COMPLETENESS SUMMARY
    # ============================================================

    required_fields = [
        "procedure_name",
        "code",
        "target_objects",
        "execution_steps",
        "execution_methods",
        "profile_components",
        "requirements_conditions",
        "fees",
        "results",
        "executing_agencies",
        "legal_basis",
    ]

    available_required = sum(
        completeness.get(field_name, False)
        for field_name in required_fields
    )

    completeness_ratio = (
        available_required / len(required_fields)
        if required_fields
        else 0.0
    )

    completeness_status = (
        "complete"
        if completeness_ratio >= 0.8
        else "partial"
    )

    # ============================================================
    # BUILD EVIDENCE
    # ============================================================

    return Evidence(
        evidence_id=(
            f"catalog:{procedure.formality_id}"
        ),
        source_type="direct_catalog",

        source_id=procedure.formality_id,

        source_url=procedure.source_url,

        title=procedure.procedure_name,

        content=content,

        score=1.0,

        fields=fields,

        completeness=completeness,

        metadata={
            "code": procedure.code,
            "category_name": procedure.category_name,
            "target_type": procedure.target_type,
            "department_promulgate": (
                procedure.department_promulgate
            ),
            "fees": procedure.fees,
            "legal_basis": procedure.legal_basis,
            "executing_agencies": (
                procedure.executing_agencies
            ),
            "completeness_ratio": (
                round(completeness_ratio, 3)
            ),
            "completeness_status": (
                completeness_status
            ),
        },
    )
def catalog_record_to_evidence(
    record: CatalogRecord,
) -> Evidence:

    fields = {
        "formality_id": record.record_id,
        "code": record.code,
        "procedure_name": record.procedure_name,
        "target_type": record.target_type,
        "category_name": record.category_name,
        "department_promulgate": record.department_promulgate,
        "description": record.description,
        "execution_steps": record.execution_steps,
        "execution_methods": record.execution_methods,
        "profile_components": record.profile_components,
        "requirements_conditions": record.requirements_conditions,
        "fees": record.fees,
        "legal_basis": record.legal_basis,
        "results": record.results,
        "target_objects": record.target_objects,
        "executing_agencies": record.executing_agencies,
        "coordinating_agencies": record.coordinating_agencies,
    }

    completeness = {}

    for field_name, value in fields.items():
        if value is None:
            completeness[field_name] = False
        elif isinstance(value, str):
            completeness[field_name] = bool(value.strip())
        else:
            completeness[field_name] = True

    sections = [
        ("Mã thủ tục", record.code),
        ("Tên thủ tục", record.procedure_name),
        ("Loại đối tượng", record.target_type),
        ("Lĩnh vực", record.category_name),
        ("Cơ quan ban hành", record.department_promulgate),
        ("Mô tả", record.description),
        ("Đối tượng thực hiện", record.target_objects),
        ("Trình tự thực hiện", record.execution_steps),
        ("Cách thức thực hiện", record.execution_methods),
        ("Thành phần hồ sơ", record.profile_components),
        ("Yêu cầu, điều kiện", record.requirements_conditions),
        ("Phí, lệ phí", record.fees),
        ("Kết quả thực hiện", record.results),
        ("Cơ quan thực hiện", record.executing_agencies),
        ("Cơ quan phối hợp", record.coordinating_agencies),
        ("Căn cứ pháp lý", record.legal_basis),
    ]

    content_parts = []

    for label, value in sections:
        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()

        if value:
            content_parts.append(
                f"{label}:\n{value}"
            )

    content = "\n\n".join(content_parts)

    required_fields = [
        "procedure_name",
        "code",
        "target_objects",
        "execution_steps",
        "execution_methods",
        "profile_components",
        "requirements_conditions",
        "fees",
        "results",
        "executing_agencies",
        "legal_basis",
    ]

    available_required = sum(
        completeness.get(field_name, False)
        for field_name in required_fields
    )

    completeness_ratio = (
        available_required / len(required_fields)
        if required_fields
        else 0.0
    )

    completeness_status = (
        "complete"
        if completeness_ratio >= 0.8
        else "partial"
    )

    source_url = None

    if record.source:
        source_url = record.source.source_url

    return Evidence(
        evidence_id=f"catalog:{record.record_id}",
        source_type="direct_catalog",
        source_id=record.record_id,
        source_url=source_url,
        title=record.procedure_name,
        content=content,
        score=1.0,
        fields=fields,
        completeness=completeness,
        metadata={
            "code": record.code,
            "category_name": record.category_name,
            "target_type": record.target_type,
            "department_promulgate": record.department_promulgate,
            "fees": record.fees,
            "legal_basis": record.legal_basis,
            "executing_agencies": record.executing_agencies,
            "completeness_ratio": round(
                completeness_ratio,
                3,
            ),
            "completeness_status": completeness_status,
        },
    )