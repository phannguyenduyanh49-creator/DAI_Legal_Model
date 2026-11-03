"""Domain-scoped document ingestion into Qdrant or the PostgreSQL/Elasticsearch store."""

from __future__ import annotations

import re
import uuid
import hashlib
import time
from io import BytesIO
from pathlib import PurePath
from typing import Any, Callable
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from qdrant_client.http import models

from backend.domain_registry import DomainRegistryService, registry_service

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".md", ".xlsx"}
MAX_XLSX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_XLSX_SHEETS = 20
MAX_XLSX_ROWS_PER_SHEET = 50_000
MAX_XLSX_COLUMNS = 200
MAX_XLSX_CELLS = 200_000
MAX_XLSX_RECORD_CHARS = 8_192


class DomainDocumentError(ValueError):
    pass


def domain_document_collection(domain_id: str) -> str:
    normalized = str(domain_id or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", normalized):
        raise DomainDocumentError("Invalid domain identifier")
    return f"domain_documents_{normalized}"


def domain_upload_payload_matches(payload: dict[str, Any], domain_id: str) -> bool:
    """Fail closed when an upload vector's payload escapes its domain collection."""
    if str(payload.get("source_type") or "") != "domain_upload":
        return True
    return str(payload.get("domain_id") or "").strip().lower() == str(
        domain_id or ""
    ).strip().lower()


def retrieval_collections_for_domain(
    domain_id: str,
    collection_exists: Callable[[str], bool],
    registry: DomainRegistryService = registry_service,
) -> list[str]:
    config = registry.get_domain_config(domain_id)
    if not config.enabled:
        return []
    candidates = []
    if config.knowledge_collection:
        candidates.append(config.knowledge_collection)
    candidates.append(domain_document_collection(config.domain_id))
    return [
        collection
        for collection in dict.fromkeys(candidates)
        if collection_exists(collection)
    ]


def chunk_document(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """Split text by approximate character budget while preserving offsets."""
    size = max(256, min(int(chunk_size) * 4, 8192))
    overlap = max(0, min(int(chunk_overlap) * 4, size // 2))
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            chunks.append((start, end, value))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def excel_row_records(text: str) -> list[dict[str, Any]]:
    """Recover coherent sheet rows from ``excel_to_text`` output."""
    records = []
    for section in re.split(r"\n\n(?=\[Sheet: )", text.strip()):
        match = re.match(r"\[Sheet: (.*?) \| Row: (\d+)\]\n", section)
        if not match:
            continue
        fields = {}
        for line in section[match.end():].splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() and value.strip():
                fields[key.strip()] = value.strip()
        if fields:
            records.append({
                "sheet_name": match.group(1),
                "row_number": int(match.group(2)),
                "fields": fields,
                "text": section,
            })
    return records


def _normalized_header(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", value.lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _is_instruction_sheet(sheet_name: str) -> bool:
    normalized = _normalized_header(sheet_name).replace("_", " ")
    return bool(re.search(
        r"(?:^| )(?:readme|read me|instructions?|guide|huong dan|ghi chu|metadata|about|gioi thieu|muc luc)(?: |$)",
        normalized,
    ))


def _sheet_domain_from_name(
    sheet_name: str,
    registry: DomainRegistryService,
) -> str | None:
    normalized_name = _normalized_header(sheet_name)
    for domain_id in registry.list_all_domains():
        config = registry.get_domain_config(domain_id)
        aliases = {_normalized_header(alias) for alias in config.worksheet_aliases}
        if normalized_name in aliases:
            return domain_id
    return None


def _sheet_domain_from_headers(
    records: list[dict[str, Any]],
    registry: DomainRegistryService,
) -> str | None:
    headers = {
        _normalized_header(str(key))
        for record in records
        for key in (record.get("fields") or {})
    }
    matched_domains = {
        domain_id
        for domain_id in registry.list_all_domains()
        if any(
            {_normalized_header(field) for field in signature} <= headers
            for signature in registry.get_domain_config(domain_id).worksheet_header_signatures
        )
    }
    if len(matched_domains) > 1:
        raise DomainDocumentError("Ambiguous Excel worksheet headers map to multiple domains")
    return next(iter(matched_domains), None)


def map_excel_records_to_domains(
    records: list[dict[str, Any]],
    selected_domain: str,
    registry: DomainRegistryService = registry_service,
) -> list[dict[str, Any]]:
    """Map named/schema-recognized sheets; retain selected-domain fallback only for legacy single-domain files."""
    selected = registry.get_domain_config(selected_domain)
    records_by_sheet: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_sheet.setdefault(str(record.get("sheet_name") or ""), []).append(record)

    normal_records = {
        name: rows for name, rows in records_by_sheet.items()
        if not _is_instruction_sheet(name)
    }
    if not normal_records:
        raise DomainDocumentError("Excel has no indexable records after instruction sheets are excluded")

    sheet_domains: dict[str, str | None] = {}
    for name, rows in normal_records.items():
        name_domain = _sheet_domain_from_name(name, registry)
        header_domain = _sheet_domain_from_headers(rows, registry)
        if name_domain and header_domain and name_domain != header_domain:
            raise DomainDocumentError(
                f"Excel worksheet '{name}' name conflicts with its domain-specific headers"
            )
        sheet_domains[name] = name_domain or header_domain

    mapped_domains = {value for value in sheet_domains.values() if value}
    has_ambiguous_sheets = any(value is None for value in sheet_domains.values())
    if mapped_domains and has_ambiguous_sheets:
        names = ", ".join(name for name, value in sheet_domains.items() if value is None)
        raise DomainDocumentError(
            f"Ambiguous Excel worksheet domain mapping for: {names}"
        )
    if mapped_domains and selected.domain_id not in mapped_domains:
        raise DomainDocumentError(
            f"Selected domain '{selected.domain_id}' does not match the workbook's mapped domains"
        )

    mapped_records = []
    for name, rows in normal_records.items():
        target_domain = sheet_domains[name] or selected.domain_id
        try:
            target_config = registry.get_domain_config(target_domain)
        except KeyError as exc:
            raise DomainDocumentError(f"Worksheet '{name}' maps to an unknown domain") from exc
        if (
            not target_config.enabled
            or not target_config.upload_enabled
            or ".xlsx" not in target_config.supported_extensions
        ):
            raise DomainDocumentError(
                f"Worksheet '{name}' maps to a domain that does not allow document upload"
            )
        for record in rows:
            mapped_records.append({**record, "domain_id": target_domain})
    return mapped_records


def decode_document(
    filename: str,
    content: bytes,
    *,
    allowed_extensions: set[str] | None = None,
) -> tuple[str, str]:
    safe_name = PurePath(str(filename or "")).name.strip()
    extension = PurePath(safe_name).suffix.lower()
    supported_extensions = allowed_extensions or ALLOWED_EXTENSIONS

    if extension not in supported_extensions:
        raise DomainDocumentError(
            "Chỉ hỗ trợ file .txt, .md và .xlsx."
        )

    if not content:
        raise DomainDocumentError("File rỗng.")

    if len(content) > MAX_DOCUMENT_BYTES:
        raise DomainDocumentError("File vượt quá giới hạn 2 MiB.")

    if extension == ".xlsx":
        text = excel_to_text(content)
    else:
        try:
            text = content.decode("utf-8-sig").replace("\x00", "").strip()
        except UnicodeDecodeError as exc:
            raise DomainDocumentError(
                "File văn bản phải sử dụng UTF-8."
            ) from exc

    if not text:
        raise DomainDocumentError(
            "Không tìm thấy nội dung có thể sử dụng."
        )

    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise DomainDocumentError(
            "Nội dung sau khi đọc vượt quá giới hạn 2 MiB."
        )

    return safe_name, text

def excel_to_text(content: bytes) -> str:
    """Convert bounded .xlsx sheets to text without loading the workbook into RAM."""
    try:
        with ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            expanded_bytes = sum(member.file_size for member in members)
            if (
                len(members) > 2_000
                or expanded_bytes > MAX_XLSX_UNCOMPRESSED_BYTES
                or any(
                    member.file_size > 0
                    and member.file_size / max(member.compress_size, 1) > 200
                    for member in members
                )
            ):
                raise DomainDocumentError("File Excel vượt quá giới hạn xử lý an toàn.")
    except BadZipFile as exc:
        raise DomainDocumentError("Không đọc được file Excel .xlsx hợp lệ.") from exc

    try:
        workbook = load_workbook(
            BytesIO(content),
            read_only=True,
            data_only=True,
        )
    except Exception as exc:
        raise DomainDocumentError(
            "Không đọc được file Excel .xlsx hợp lệ."
        ) from exc

    sections = []
    total_cells = 0
    total_characters = 0

    try:
        if len(workbook.worksheets) > MAX_XLSX_SHEETS:
            raise DomainDocumentError("File Excel có quá nhiều sheet.")
        for sheet in workbook.worksheets:
            if _is_instruction_sheet(sheet.title):
                continue
            if (
                sheet.max_row is not None
                and sheet.max_row > MAX_XLSX_ROWS_PER_SHEET
            ) or (
                sheet.max_column is not None
                and sheet.max_column > MAX_XLSX_COLUMNS
            ):
                raise DomainDocumentError("Sheet Excel vượt giới hạn số dòng hoặc cột.")
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows, None)

            if not headers:
                continue

            headers = [
                str(value).strip() if value is not None
                else f"column_{i + 1}"
                for i, value in enumerate(headers)
            ]

            for row_number, row in enumerate(rows, start=2):
                if row_number > MAX_XLSX_ROWS_PER_SHEET:
                    raise DomainDocumentError("Sheet Excel vượt giới hạn số dòng.")
                total_cells += len(row)
                if total_cells > MAX_XLSX_CELLS:
                    raise DomainDocumentError("File Excel vượt giới hạn số ô.")
                fields = []

                for i, value in enumerate(row):
                    if value is None or not str(value).strip():
                        continue

                    key = (
                        headers[i]
                        if i < len(headers)
                        else f"column_{i + 1}"
                    )
                    fields.append(f"{key}: {str(value).strip()}")

                if fields:
                    section = (
                        f"[Sheet: {sheet.title} | Row: {row_number}]\n"
                        + "\n".join(fields)
                    )
                    total_characters += len(section)
                    if total_characters > MAX_DOCUMENT_BYTES:
                        raise DomainDocumentError("Nội dung Excel sau khi đọc vượt quá giới hạn 2 MiB.")
                    sections.append(section)
    finally:
        workbook.close()

    if not sections:
        raise DomainDocumentError(
            "File Excel không có dữ liệu để lập chỉ mục."
        )

    return "\n\n".join(sections)

class DomainDocumentService:
    def __init__(
        self,
        qdrant: Any,
        embedder: Any,
        registry: DomainRegistryService = registry_service,
        search_store: Any | None = None,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.registry = registry
        self.search_store = search_store

    def upload(self, domain_id: str, filename: str, content: bytes) -> dict[str, Any]:
        try:
            config = self.registry.get_domain_config(domain_id)
        except KeyError as exc:
            raise DomainDocumentError("Unknown domain") from exc
        if not config.enabled:
            raise DomainDocumentError("Domain is disabled")
        if not config.upload_enabled:
            raise DomainDocumentError("Document upload is not enabled for this domain")
        title, text = decode_document(
            filename,
            content,
            allowed_extensions=set(config.supported_extensions),
        )
        chunk_size = config.chunking_strategy.chunk_size
        chunk_overlap = config.chunking_strategy.chunk_overlap
        indexed_chunks = []
        is_xlsx = PurePath(title).suffix.lower() == ".xlsx"
        row_records = excel_row_records(text) if is_xlsx else []
        has_explicit_domain_mapping = any(
            _sheet_domain_from_name(str(record.get("sheet_name") or ""), self.registry) is not None
            or _sheet_domain_from_headers([record], self.registry) is not None
            for record in row_records
            if not _is_instruction_sheet(str(record.get("sheet_name") or ""))
        )

        # Legal .xlsx files keep their historical generic chunking unless they
        # contain a worksheet explicitly mapped to another domain, which fails
        # closed in map_excel_records_to_domains below.
        use_row_records = is_xlsx and config.domain_id not in {"legal_ai", "administrative_law"}
        if is_xlsx and has_explicit_domain_mapping:
            row_records = map_excel_records_to_domains(row_records, config.domain_id, self.registry)
            use_row_records = True
        elif use_row_records:
            row_records = map_excel_records_to_domains(row_records, config.domain_id, self.registry)

        if use_row_records:
            for record in row_records:
                if len(record["text"]) > MAX_XLSX_RECORD_CHARS:
                    raise DomainDocumentError(
                        "Một bản ghi trong Excel quá dài để lập chỉ mục thành Evidence độc lập."
                    )
                fields = record["fields"]
                normalized_fields = {
                    _normalized_header(key): value
                    for key, value in fields.items()
                }
                record_name = next((
                    value for key, value in normalized_fields.items()
                    if key in {
                        "name", "title", "product", "school", "course",
                        "facility", "destination", "ten", "ten_co_so",
                        "ten_khoa_hoc", "ten_don_vi_chuong_trinh", "ten_san_pham",
                        "ten_dia_diem", "ten_diem_den", "ten_dia_danh",
                        "destination_name",
                    }
                ), "")
                indexed_chunks.append({
                    "domain_id": record["domain_id"],
                    "start": 0,
                    "end": len(record["text"]),
                    "text": record["text"],
                    "sheet_name": record["sheet_name"],
                    "row_number": record["row_number"],
                    "record_fields": normalized_fields,
                    "record_id": normalized_fields.get("record_id"),
                    "record_name": record_name,
                })
        else:
            indexed_chunks = [
                {"domain_id": config.domain_id, "start": start, "end": end, "text": chunk}
                for start, end, chunk in chunk_document(
                    text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            ]
        if not indexed_chunks:
            raise DomainDocumentError("File không có bản ghi để lập chỉ mục.")
        if self.search_store is None:
            embedding_started_at = time.perf_counter()
            vectors = self.embedder.encode(
                [chunk["text"] for chunk in indexed_chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            print(
                "[MODEL INFERENCE]",
                "model_name=BAAI/bge-m3",
                "selected_device=", str(getattr(self.embedder, "device", "unknown")),
                "status=completed",
                "input_count=", len(indexed_chunks),
                "duration_ms=", round((time.perf_counter() - embedding_started_at) * 1000, 2),
                "operation=document_embedding",
                flush=True,
            )
        else:
            vectors = [None] * len(indexed_chunks)
        if len(vectors) != len(indexed_chunks):
            raise RuntimeError("Document embedding returned an incomplete vector set")

        content_digest = hashlib.sha256(content).hexdigest()
        chunks_by_domain: dict[str, list[tuple[dict[str, Any], Any]]] = {}
        for chunk_record, vector in zip(indexed_chunks, vectors):
            chunks_by_domain.setdefault(chunk_record["domain_id"], []).append((chunk_record, vector))

        domain_results = []
        document_ids = {}
        collections = {}
        points_by_domain = {}
        payloads_by_domain = {}
        for target_domain, domain_chunks in chunks_by_domain.items():
            collection = domain_document_collection(target_domain)
            deterministic_ids = self.search_store is not None or (
                is_xlsx and target_domain not in {"legal_ai", "administrative_law"}
            )
            document_id = (
                str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"domain-upload:{target_domain}:{title.casefold()}:{content_digest}",
                ))
                if deterministic_ids
                else str(uuid.uuid4())
            )
            document_ids[target_domain] = document_id
            collections[target_domain] = (
                self.search_store.elasticsearch.index_name
                if self.search_store is not None
                else collection
            )
            if self.search_store is None and not self.qdrant.collection_exists(collection):
                self.qdrant.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=int(self.embedder.get_sentence_embedding_dimension()),
                        distance=models.Distance.COSINE,
                    ),
                )

            points = []
            payload_rows = []
            for index, (chunk_record, vector) in enumerate(domain_chunks):
                start = chunk_record["start"]
                end = chunk_record["end"]
                chunk = chunk_record["text"]
                evidence_id = f"domain:{target_domain}:{document_id}:{index + 1}"
                payload = {
                    "domain_id": target_domain,
                    "document_id": document_id,
                    "evidence_id": evidence_id,
                    "source_type": "domain_upload",
                    "source_id": document_id,
                    "title": title,
                    "procedure_name": title,
                    "location": (
                        f"Sheet {chunk_record['sheet_name']} · row {chunk_record['row_number']}"
                        if "record_fields" in chunk_record
                        else f"Characters {start + 1}-{end}"
                    ),
                    "chunk_index": index + 1,
                    "text": chunk,
                    "full_text": chunk,
                }
                if "record_fields" in chunk_record:
                    fields = chunk_record["record_fields"]
                    payload.update({
                        "sheet_name": chunk_record["sheet_name"],
                        "row_number": chunk_record["row_number"],
                        "record_fields": fields,
                        "record_name": chunk_record["record_name"],
                    })
                    if chunk_record.get("record_id"):
                        payload["record_id"] = chunk_record["record_id"]
                    if target_domain == "tourism":
                        payload.update({
                            "tourism_region": next((value for key, value in fields.items() if key in {"khu_vuc", "region"}), ""),
                            "tourism_province_city": next((value for key, value in fields.items() if key in {"tinh_thanh_pho", "tinh_thanh", "province_city", "city"}), ""),
                            "tourism_visit_duration": next((value for key, value in fields.items() if key in {"thoi_gian_tham_quan_goi_y", "thoi_gian_tham_quan", "visit_duration", "recommended_visit_duration"}), ""),
                        })
                point_id = (
                    str(uuid.uuid5(uuid.UUID(document_id), str(index + 1)))
                    if deterministic_ids
                    else str(uuid.uuid4())
                )
                payload_rows.append((str(point_id), payload))
                if self.search_store is None:
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                        payload=payload,
                    ))
            points_by_domain[target_domain] = points
            payloads_by_domain[target_domain] = payload_rows

        if self.search_store is not None:
            # A single PostgreSQL transaction stores every mapped domain before
            # the outbox is synchronized to Elasticsearch.
            self.search_store.index_documents([
                row for payload_rows in payloads_by_domain.values() for row in payload_rows
            ])
            for target_domain, payload_rows in payloads_by_domain.items():
                domain_results.append({
                    "domain_id": target_domain,
                    "document_id": document_ids[target_domain],
                    "collection": collections[target_domain],
                    "chunk_count": len(payload_rows),
                })

        else:
            # Validate/create every target collection before the first point write.
            # Qdrant has no cross-collection transaction; a later failure can still
            # leave a partial multi-domain upload, which is surfaced to the caller.
            for target_domain, points in points_by_domain.items():
                upsert_result = self.qdrant.upsert(
                    collection_name=collections[target_domain],
                    points=points,
                    wait=True,
                )
                upsert_status = getattr(upsert_result, "status", None)
                upsert_status = getattr(upsert_status, "value", upsert_status)
                if upsert_status != "completed":
                    raise RuntimeError(
                        f"Qdrant did not confirm completed indexing for domain '{target_domain}'"
                    )
                domain_results.append({
                    "domain_id": target_domain,
                    "document_id": document_ids[target_domain],
                    "collection": collections[target_domain],
                    "chunk_count": len(points),
                })

        for target_domain, points in points_by_domain.items() if self.search_store is None else []:
            collection = collections[target_domain]
            document_id = document_ids[target_domain]
            if target_domain == "tourism" and is_xlsx:
                replacement_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="domain_id",
                            match=models.MatchValue(value=target_domain),
                        ),
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value="domain_upload"),
                        ),
                        models.FieldCondition(
                            key="title",
                            match=models.MatchValue(value=title),
                        ),
                    ],
                    must_not=[models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )],
                )
                cleanup_result = self.qdrant.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=replacement_filter),
                    wait=True,
                )
                cleanup_status = getattr(cleanup_result, "status", None)
                cleanup_status = getattr(cleanup_status, "value", cleanup_status)
                if cleanup_status != "completed":
                    raise RuntimeError("Qdrant did not confirm replacement of the prior tourism workbook")
        selected_result = next(
            (item for item in domain_results if item["domain_id"] == config.domain_id),
            None,
        )
        return {
            "status": "indexed",
            "domain_id": config.domain_id,
            "document_id": selected_result["document_id"] if selected_result else None,
            "title": title,
            "chunk_count": selected_result["chunk_count"] if selected_result else 0,
            "total_chunk_count": sum(item["chunk_count"] for item in domain_results),
            "indexed_domains": domain_results,
        }
