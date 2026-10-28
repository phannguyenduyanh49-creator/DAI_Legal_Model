import os
import sys
import time
import logging
import json
import re
import uuid
import httpx
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from backend.answer.diagnostics import classify_upstream_exception
from backend.answer.reference_policy import (
    classify_business_outcome,
    reference_answer_failure_reason,
)

from backend import answer


from backend.query_analysis import QueryAnalysisService, AnalysisResult
from backend.orchestrator import DomainRouterPlanner, RoutingPlanResult
from backend.domain_registry import registry_service, domain_router
from backend.policy.middleware import PolicyEnforcementMiddleware
# ==============================================================================
# CẤU HÌNH HỆ THỐNG LOGGING & ENCODING UTF-8 (HỖ TRỢ WINDOWS CONSOLE)
# ==============================================================================
def configure_utf8_stdio() -> None:
    """Ép mã hóa console sang UTF-8 để hiển thị log tiếng Việt không bị lỗi font trên Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass

configure_utf8_stdio()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("DAI_Legal_Core_API_Gateway")


@dataclass
class LegacyPipelineResult:
    answer: str
    pipeline: Dict[str, Any] = field(default_factory=dict)
    verifier_status: str = "not_reported"
    verifier_scope: str = "not_reported"
    session_cookie_headers: List[str] = field(default_factory=list)


UNVERIFIED_ANSWER_FALLBACK = (
    "Chua the dua ra cau tra loi phap ly du dieu kien "
    "kiem tra trich dan Evidence."
)


def _verification_fallback_message(reason_code: str | None) -> str:
    """Return safe, user-facing copy aligned with the structured failure code."""
    code = reason_code or ""
    if code == "EVIDENCE_IDS_MISSING":
        return (
            "Hiện chưa có Evidence phù hợp để kiểm chứng câu trả lời; "
            "hệ thống chưa tạo câu trả lời pháp lý."
        )
    if code.startswith(("MODEL_API_", "UPSTREAM_")):
        return "Không thể hoàn tất sinh câu trả lời do lỗi dịch vụ mô hình. Vui lòng thử lại sau."
    if code.startswith("CITATION_"):
        return "Câu trả lời đã được giữ lại vì citation Evidence thiếu, sai định dạng hoặc không hợp lệ."
    if code.startswith("CLAIM_GROUNDING_"):
        return "Câu trả lời đã được giữ lại vì hệ thống chưa xác minh được các nhận định dựa trên Evidence."
    if code.startswith("REFERENCE_ANSWER_ESSENTIAL_FACTS"):
        return (
            "Evidence hiá»‡n cÃ³ chÆ°a Ä‘á»§ cÃ¡c thÃ´ng tin thiáº¿t yáº¿u Ä‘á»ƒ "
            "hiá»ƒn thá»‹ cÃ¢u tráº£ lá»i an toÃ n; cÃ¢u tráº£ lá»i Ä‘Ã£ Ä‘Æ°á»£c giá»¯ láº¡i."
        )
    if code.startswith("VERIFICATION_CONFIDENCE_"):
        return "Câu trả lời đã được giữ lại vì chưa đạt điều kiện kiểm chứng độ tin cậy."
    return UNVERIFIED_ANSWER_FALLBACK


def _generation_token_limit() -> int:
    try:
        value = int(os.getenv("MODEL_MAX_NEW_TOKENS", "256"))
    except (TypeError, ValueError):
        value = 256
    return max(1, min(value, 512))


def _valid_session_id(value: str | None) -> str | None:
    """Forward only Core-generated UUID session identifiers to the Core API."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def _citation_link_failure_reason(result: "LegacyPipelineResult") -> str | None:
    """Validate Evidence IDs, cited IDs, and citation text/telemetry consistency."""
    pipeline = result.pipeline
    evidence_ids = pipeline.get("evidence_ids")
    cited_ids = pipeline.get("cited_evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return "EVIDENCE_IDS_MISSING"
    if not isinstance(cited_ids, list) or not cited_ids:
        return "CITATION_TELEMETRY_MISMATCH"
    if any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
        return "EVIDENCE_IDS_INVALID"
    if len(set(evidence_ids)) != len(evidence_ids):
        return "EVIDENCE_IDS_INVALID"
    if any(not isinstance(item, str) or not item.strip() for item in cited_ids):
        return "EVIDENCE_ID_INVALID"
    allowed = {item for item in evidence_ids if isinstance(item, str) and item}
    cited = {item for item in cited_ids if isinstance(item, str) and item}
    if not allowed:
        return "EVIDENCE_IDS_MISSING"
    if not cited:
        return "CITATION_TELEMETRY_MISMATCH"
    if not cited.issubset(allowed):
        return "EVIDENCE_ID_NOT_ALLOWED"
    answer_citations = {
        item.strip()
        for item in re.findall(
            r"\[Evidence ID:\s*([^\]]+)\]", result.answer, flags=re.IGNORECASE
        )
        if item.strip()
    }
    if not answer_citations or answer_citations != cited:
        return "CITATION_TELEMETRY_MISMATCH"
    return None


def _citation_integrity_failure_reason(
    result: "LegacyPipelineResult",
) -> str | None:
    """Return a stable diagnostic reason while preserving fail-closed checks."""
    pipeline = result.pipeline
    if result.verifier_status == "not_required":
        if pipeline.get("response_status") == "waiting_permission":
            return None
        return "VERIFIER_NOT_PASSED"
    evidence_ids = pipeline.get("evidence_ids")
    allowed_evidence_ids = {
        item for item in evidence_ids
        if isinstance(item, str) and item.strip()
    } if isinstance(evidence_ids, list) else set()
    if (
        not allowed_evidence_ids
        and pipeline.get("failure_reason_code") in {
            "EVIDENCE_IDS_MISSING", "CITATION_MISSING",
        }
        and pipeline.get("failure_stage") in {
            "retrieval_pipeline", "evidence_builder", "prompt_assembly", "verifier",
        }
    ):
        return "EVIDENCE_IDS_MISSING"
    if pipeline.get("citation_verification") in ("failed", "error"):
        return "CITATION_VERIFICATION_NOT_PASSED"
    if result.verifier_status not in ("passed", "not_verified"):
        return "VERIFIER_NOT_PASSED"
    link_failure_reason = _citation_link_failure_reason(result)
    if link_failure_reason:
        return link_failure_reason

    confidence_status = pipeline.get("confidence_status")
    if pipeline.get("verification_threshold") is not None and confidence_status != "passed":
        return (
            "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
            if confidence_status == "not_verified"
            else "VERIFICATION_CONFIDENCE_NOT_PASSED"
        )
    grounding_status = pipeline.get("claim_grounding_status")
    if grounding_status == "failed":
        return "CLAIM_GROUNDING_NOT_PASSED"
    if result.verifier_status not in ("passed", "not_verified"):
        return "VERIFIER_NOT_PASSED"
    if grounding_status != "passed":
        return (
            "CLAIM_GROUNDING_NOT_EVALUATED"
            if grounding_status in (None, "not_evaluated")
            else "CLAIM_GROUNDING_NOT_PASSED"
        )
    if pipeline.get("verification_status") != "passed":
        return "VERIFICATION_NOT_PASSED"
    if result.verifier_status != "passed":
        return "VERIFIER_NOT_PASSED"
    if pipeline.get("citation_verification") != "passed":
        return "CITATION_VERIFICATION_NOT_PASSED"
    return None


def _has_valid_citation_integrity(result: "LegacyPipelineResult") -> bool:
    """Compatibility wrapper for callers that only need the pass/fail value."""
    return _citation_integrity_failure_reason(result) is None


def _business_outcome(
    result: "LegacyPipelineResult",
    answer_is_verified: bool,
    answer_is_unverified_reference: bool = False,
) -> str:
    """Compatibility wrapper around the shared business-outcome policy."""
    return classify_business_outcome(
        result.pipeline,
        answer_is_verified=answer_is_verified,
        answer_is_unverified_reference=answer_is_unverified_reference,
    )


# ==============================================================================
# PYDANTIC v2 SCHEMAS & VALIDATION
# ==============================================================================
class UserContext(BaseModel):
    user_id: str = Field(..., description="Mã định danh người dùng hệ thống", json_schema_extra={"example": "lawyer_expert_01"})
    role: str = Field(default="legal_consultant", description="Vai trò người dùng", json_schema_extra={"example": "admin"})
    session_id: Optional[str] = Field(default=None, description="Mã phiên làm việc hiện tại của user")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=4000, description="Câu hỏi hoặc vấn đề pháp lý cần tư vấn", json_schema_extra={"example": "Thủ tục thành lập công ty cổ phần và điều kiện về vốn điều lệ theo luật mới?"})
    domain_hint: Optional[str] = Field(default=None, description="Gợi ý phân vùng luật (civil_law, enterprise_law, criminal_law, administrative_law, labor_law, land_law)", json_schema_extra={"example": "enterprise_law"})
    user_context: Optional[UserContext] = Field(default=None, description="Thông tin bối cảnh người dùng thực hiện request")
    enable_hyde: bool = Field(default=True, description="Kích hoạt cơ chế sinh tài liệu giả định HyDE để tăng độ chính xác tìm kiếm")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="Nhiệt độ sinh văn bản cho LLM backend")
    allow_external: bool = Field(
        default=False,
        description="Chỉ cho phép tra cứu nguồn ngoài khi người dùng chủ động đồng ý",
    )

    @field_validator("query")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Câu truy vấn không được để trống hoặc chứa toàn khoảng trắng.")
        return v

class CoreAPIResponse(BaseModel):
    request_id: str = Field(..., description="Mã vết duy nhất của request (Request Trace ID)")
    timestamp: str = Field(..., description="Thời gian ghi nhận phản hồi (UTC ISO format)")
    processing_time_ms: float = Field(..., description="Tổng thời gian xử lý của Gateway (tính bằng mili-giây)")
    status: str = Field(..., description="Trạng thái thực thi yêu cầu (SUCCESS / FAILED)")
    outcome: str = Field(..., description="Business outcome independent of HTTP status")
    query_analysis: AnalysisResult = Field(..., description="Kết quả phân tích từ module Query Analysis")
    routing_and_planning: RoutingPlanResult = Field(..., description="Kết quả phân vùng domain và kế hoạch thực thi từ Orchestrator")
    next_step: str = Field(
        ...,
        description="Giai đoạn xử lý đã thực hiện",
    )
    domain_policy: Dict[str, Any] = Field(
        default_factory=dict,
        description="Policy được chọn theo domain",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Câu trả lời từ pipeline xử lý phía backend",
    )
    pipeline_execution: Dict[str, Any] = Field(
        default_factory=dict,
        description="Trạng thái các bước pipeline đã được gọi",
    )


# ==============================================================================
# CORE API GATEWAY CLASS
# ==============================================================================
class CoreAPIGateway:
    """
    Lớp điều phối chính quản lý ứng dụng FastAPI Gateway.
    Đóng vai trò là trung tâm nhận request, kiểm tra bảo mật middleware,
    đăng ký các route endpoint và tích hợp QueryAnalysis + Orchestrator.
    """
    def __init__(self):
        logger.info("[Module 1 - CoreAPIGateway] Đang khởi tạo ứng dụng FastAPI và cấu hình Middleware...")

        self.app = FastAPI(
            title="DAI Legal Model - Enterprise Core Gateway API",
            description=(
                "Hệ thống Backend chuẩn sản xuất phục vụ trợ lý AI pháp lý.\n"
                "Tích hợp các Module:\n"
                "1. API Gateway & Backend API (FastAPI)\n"
                "2. Orchestrator (Domain Router + Agent Planner)\n"
                "3. Query Analysis & Clarification (HyDE & Advanced Rewriting)"
            ),
            version="2.5.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Cấu hình CORS Middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                origin.strip()
                for origin in os.getenv(
                    "CORS_ALLOWED_ORIGINS",
                    ",".join(
                        f"http://{host}:{port}"
                        for host in ("127.0.0.1", "localhost")
                        for port in (5173, 5174, 5175)
                    ),
                ).split(",")
                if origin.strip()
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Khởi tạo các Service nghiệp vụ cốt lõi

        self.query_analyzer = QueryAnalysisService()
        self.orchestrator = DomainRouterPlanner()
        self.policy_middleware = PolicyEnforcementMiddleware()

        # Core API hiện tại vẫn chịu trách nhiệm Retrieval/Evidence/Prompt/Qwen
        self.legacy_api_url = os.getenv(
            "LEGACY_CORE_API_URL",
            "http://127.0.0.1:8000/api/v1/chat/completions",
        )

        # Đăng ký các route domain/policy từ Domain Registry của Huy
        self.app.include_router(domain_router)

        # Đăng ký các Endpoint API
        self._register_middleware_and_routes()



    async def _call_legacy_pipeline(
        self,
        query: str,
        allow_external: bool,
        system_instruction: str,
        request_id: str,
        policy_domain: str,
        allowed_mcp_tools: list[str],
        session_id: str | None = None,
    ) -> LegacyPipelineResult:
        """Gọi Core API cũ để tái sử dụng pipeline Retrieval/Evidence/Prompt/Qwen."""
        payload = {
            "prompt": query,
            "allow_external": allow_external,
            "system_instruction": system_instruction,
            "policy_domain": policy_domain,
            "mcp_allowed_tools": allowed_mcp_tools,
            "max_new_tokens": _generation_token_limit(),
        }

        timeout_seconds = 390.0
        started_at = time.perf_counter()
        logger.info(
            "[PIPELINE STAGE] request_id=%s stage=legacy_api status=started timeout_ms=%.0f allow_external=%s prompt_chars=%d",
            request_id,
            timeout_seconds * 1000,
            allow_external,
            len(query),
        )
        try:
            request_headers = {"X-Request-ID": request_id}
            if session_id:
                request_headers["Cookie"] = f"session_id={session_id}"
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds)
            ) as client:
                response = await client.post(
                    self.legacy_api_url,
                    json=payload,
                    headers=request_headers,
                )
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "[PIPELINE STAGE] request_id=%s stage=legacy_api status=responded upstream_status_code=%s elapsed_ms=%.2f content_type=%s response_chars=%d",
                request_id,
                response.status_code,
                elapsed_ms,
                response.headers.get("content-type", ""),
                len(response.text),
            )
            response.raise_for_status()

        except httpx.HTTPError as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            reason_code = classify_upstream_exception(exc)
            upstream_status_code = (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError)
                else None
            )
            exception_type = type(exc).__name__
            logger.error(
                "[PIPELINE STAGE] request_id=%s stage=legacy_api status=failed reason_code=%s exception_type=%s upstream_status_code=%s elapsed_ms=%.2f",
                request_id,
                reason_code,
                exception_type,
                upstream_status_code,
                elapsed_ms,
            )
            failure_headers = {
                "X-Request-ID": request_id,
                "X-Failure-Stage": "legacy_api",
                "X-Failure-Reason-Code": reason_code,
                "X-Exception-Type": exception_type,
            }
            if upstream_status_code is not None:
                failure_headers["X-Upstream-Status-Code"] = str(
                    upstream_status_code
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Không thể gọi pipeline Retrieval/Evidence/Qwen hiện có. "
                    "Vui lòng cung cấp mã yêu cầu cho nhóm hỗ trợ."
                ),
                headers=failure_headers,
            ) from exc

        # Core API cũ có thể trả về Server-Sent Events (SSE).
        # Gom các delta thành một câu trả lời hoàn chỉnh.
        answer_parts: list[str] = []
        pipeline_metadata: Dict[str, Any] = {}
        verifier_status = "not_reported"
        verifier_scope = "not_reported"

        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue

            raw = line[len("data:"):].strip()

            try:
                event_data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event_data, dict):
                continue

            delta = event_data.get("delta")
            if isinstance(delta, str):
                answer_parts.append(delta)
            if isinstance(event_data.get("pipeline"), dict):
                pipeline_metadata = event_data["pipeline"]
            if event_data.get("outcome") in {
                "answered", "answered_unverified", "needs_clarification",
                "withheld", "verification_failed",
            }:
                pipeline_metadata["response_outcome"] = event_data["outcome"]
            if event_data.get("status") == "waiting_permission":
                pipeline_metadata["response_status"] = "waiting_permission"
            if isinstance(event_data.get("verifier"), str):
                verifier_status = event_data["verifier"]
            if isinstance(event_data.get("verifier_scope"), str):
                verifier_scope = event_data["verifier_scope"]

        answer = "".join(answer_parts).strip()

        # Hỗ trợ trường hợp Core API trả JSON thông thường thay vì SSE.
        if not answer:
            try:
                result = response.json()
                if isinstance(result, dict):
                    answer_value = result.get("answer")
                    if isinstance(answer_value, str):
                        answer = answer_value.strip()
                    if isinstance(result.get("pipeline"), dict):
                        pipeline_metadata = result["pipeline"]
                    if result.get("outcome") in {
                        "answered", "answered_unverified", "needs_clarification",
                        "withheld", "verification_failed",
                    }:
                        pipeline_metadata["response_outcome"] = result["outcome"]
                    if result.get("status") == "waiting_permission":
                        pipeline_metadata["response_status"] = "waiting_permission"
                    if isinstance(result.get("verifier"), str):
                        verifier_status = result["verifier"]
                    if isinstance(result.get("verifier_scope"), str):
                        verifier_scope = result["verifier_scope"]
            except ValueError:
                pass

        if not answer:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Core API đã phản hồi nhưng không trích xuất được "
                    "nội dung câu trả lời."
                ),
                headers={
                    "X-Request-ID": request_id,
                    "X-Failure-Stage": "legacy_api_response_parse",
                    "X-Failure-Reason-Code": "UNKNOWN_PIPELINE_FAILURE",
                },
            )

        get_all_headers = getattr(response.headers, "get_list", None)
        upstream_set_cookies = (
            get_all_headers("set-cookie")
            if callable(get_all_headers)
            else [response.headers.get("set-cookie", "")]
        )
        return LegacyPipelineResult(
            answer=answer,
            pipeline=pipeline_metadata,
            verifier_status=verifier_status,
            verifier_scope=verifier_scope,
            session_cookie_headers=[
                value
                for value in upstream_set_cookies
                if value
                if value.partition("=")[0].strip() == "session_id"
            ],
        )

    def _register_middleware_and_routes(self):
        @self.app.post("/api/v2/domains/{domain_id}/documents", tags=["Domain Documents"])
        async def upload_domain_document(domain_id: str, file: UploadFile = File(...)):
            try:
                registry_service.get_domain_config(domain_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown domain") from exc

            content = await file.read(2 * 1024 * 1024 + 1)
            filename = file.filename or "document.txt"
            content_type = file.content_type or "application/octet-stream"
            await file.close()
            legacy_base_url = self.legacy_api_url.removesuffix(
                "/api/v1/chat/completions"
            )
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                    upstream = await client.post(
                        f"{legacy_base_url}/api/v1/domains/{domain_id}/documents",
                        files={"file": (filename, content, content_type)},
                    )
            except httpx.HTTPError as exc:
                logger.error(
                    "[DOMAIN DOCUMENT UPLOAD] domain_id=%s status=failed exception_type=%s",
                    domain_id,
                    type(exc).__name__,
                )
                raise HTTPException(
                    status_code=502,
                    detail="Domain document service is unavailable",
                ) from exc
            if not upstream.is_success:
                try:
                    detail = upstream.json().get("detail", "Upload rejected")
                except (ValueError, AttributeError):
                    detail = "Upload rejected"
                raise HTTPException(status_code=upstream.status_code, detail=detail)
            return upstream.json()

        # Middleware đo lường hiệu năng thực thi (Latency Measurement)
        @self.app.middleware("http")
        async def add_process_latency_header(request: Request, call_next):
            start_time = time.perf_counter()
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            response.headers["X-Gateway-Latency-MS"] = str(duration_ms)
            return response

        # Endpoint kiểm tra sức khỏe hệ thống (Health Check)
        @self.app.get("/health", tags=["System Diagnostics"])
        async def health_check():
            return {
                "status": "HEALTHY",
                "service_name": "DAI-Legal-Model Core Gateway API",
                "environment": "Production-Ready Local (Drive E)",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Endpoint xử lý câu hỏi pháp lý chính thức qua 3 Module đầu tiên
        @self.app.post("/api/v2/legal-query", response_model=CoreAPIResponse, tags=["Core Legal Processing Pipeline"])
        async def process_legal_consultation_query(
            request_payload: QueryRequest,
            request: Request,
            response: Response,
        ):
            start_time = time.perf_counter()
            request_id = f"trace_legal_{uuid.uuid4().hex}"
            pipeline_execution = {
                "request_id": request_id,
                "stage_durations_ms": {},
                "query_analysis": "not_started",
                "orchestrator": "not_started",
                "domain_policy": "not_started",
                "legacy_api": "not_started",
                "catalog_lookup": "not_observed",
                "retrieval": "not_started",
                "reranking": "not_started",
                "evidence_builder": "not_started",
                "enough_data": "not_evaluated",
                "required_fields": [],
                "missing_fields": [],
                "external_fallback": "not_evaluated",
                "prompt_assembly": "not_started",
                "model_api": "not_started",
                "retrieval_evidence_prompt_qwen": "not_observed",
                "verifier": "not_integrated",
                "external_search_allowed": request_payload.allow_external,
            }

            logger.info(f"\n==================================================")
            logger.info(f"[API GATEWAY] NHẬN YÊU CẦU MỚI - Request ID: {request_id}")
            logger.info(
                "[PIPELINE STAGE] request_id=%s stage=request status=received query_chars=%d",
                request_id,
                len(request_payload.query),
            )
            if request_payload.domain_hint:
                logger.info("[PIPELINE STAGE] request_id=%s stage=domain_hint status=provided", request_id)
            logger.info(f"Cơ chế HyDE: {'Đang bật' if request_payload.enable_hyde else 'Đang tắt'}")
            logger.info(f"==================================================")

            try:
                if request_payload.domain_hint:
                    try:
                        self.policy_middleware.registry_domain_for(
                            request_payload.domain_hint
                        )
                    except KeyError as exc:
                        raise HTTPException(
                            status_code=422,
                            detail="Requested domain is not registered",
                        ) from exc

                # BƯỚC 1: Gọi Module Query Analysis & Clarification (Thay thế Synonym Mapping)
                stage_started_at = time.perf_counter()
                analysis_result = self.query_analyzer.analyze(
                    query=request_payload.query,
                    enable_hyde=request_payload.enable_hyde,
                    request_id=request_id,
                )
                pipeline_execution["stage_durations_ms"]["query_analysis"] = round(
                    (time.perf_counter() - stage_started_at) * 1000, 2
                )
                pipeline_execution["query_analysis"] = "completed"
                logger.info(
                    "[PIPELINE STAGE] request_id=%s stage=query_analysis status=completed duration_ms=%.2f",
                    request_id,
                    pipeline_execution["stage_durations_ms"]["query_analysis"],
                )

                # BƯỚC 2: Gọi Module Orchestrator (Domain Router + Agent Planner)
                stage_started_at = time.perf_counter()
                routing_plan_result = self.orchestrator.process(
                    query_analysis=analysis_result,
                    domain_hint=request_payload.domain_hint
                )
                pipeline_execution["stage_durations_ms"]["orchestrator"] = round(
                    (time.perf_counter() - stage_started_at) * 1000, 2
                )
                pipeline_execution["orchestrator"] = "completed"

                if routing_plan_result.clarification_required:
                    pipeline_execution.update({
                        "domain_policy": "not_started",
                        "failure_stage": "domain_routing",
                        "failure_reason_code": "AMBIGUOUS_DOMAIN_SIGNALS",
                        "response_outcome": "needs_clarification",
                        "answer_withheld": True,
                        "model_api": "not_started",
                    })
                    pipeline_execution.setdefault("stage_status", {}).update({
                        "domain_policy": "not_started",
                        "retrieval": "not_started",
                        "evidence_builder": "not_started",
                        "prompt_assembly": "not_started",
                        "model_api": "not_started",
                    })
                    total_processing_ms = round(
                        (time.perf_counter() - start_time) * 1000, 2
                    )
                    return CoreAPIResponse(
                        request_id=request_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        processing_time_ms=total_processing_ms,
                        status="FAILED",
                        outcome="needs_clarification",
                        query_analysis=analysis_result,
                        routing_and_planning=routing_plan_result,
                        next_step="CLARIFICATION_REQUIRED",
                        domain_policy={"status": "not_selected"},
                        answer=routing_plan_result.clarification_question,
                        pipeline_execution=pipeline_execution,
                    )

                # BƯỚC 3: Đọc Domain Registry và áp dụng Policy
                stage_started_at = time.perf_counter()
                selected_domain = routing_plan_result.selected_domain
                enforced_policy = self.policy_middleware.apply(selected_domain)
                domain_config = enforced_policy.domain_config
                registry_domain_id = domain_config.domain_id
                pipeline_execution["domain_policy"] = "resolved"
                pipeline_execution["stage_durations_ms"]["domain_policy"] = round(
                    (time.perf_counter() - stage_started_at) * 1000, 2
                )

                policy_mode = domain_config.policy.mode
                persona_instruction = (
                    f"Reply tone: {domain_config.persona.tone}. "
                    f"Language style: {domain_config.persona.language_style}. "
                    f"{domain_config.persona.system_prompt_addition}"
                )

                system_instruction = (
                    f"Chế độ chính sách: {policy_mode}. "
                    f"{persona_instruction} "
                )

                if policy_mode == "STRICT":
                    system_instruction += (
                        "Trả lời dựa trên bằng chứng và nguồn được cung cấp. "
                        "Không tự tạo điều luật, số hiệu văn bản hoặc dữ kiện. "
                        "Nếu dữ liệu chưa đủ, hãy nêu rõ giới hạn."
                    )
                else:
                    system_instruction += (
                        "Trả lời rõ ràng, thân thiện và dễ hiểu. "
                        "Phân biệt thông tin đã xác minh với thông tin chưa chắc chắn."
                    )
                if domain_config.policy.risk_alert_enabled:
                    system_instruction += (
                        " Khi thông tin có thể ảnh hưởng đến quyền, nghĩa vụ hoặc "
                        "thời hạn pháp lý, nhắc người dùng xác minh với văn bản "
                        "và cơ quan có thẩm quyền; nêu rõ nếu bằng chứng hiện có chưa đủ."
                    )

                # BƯỚC 4: Gọi pipeline Retrieval -> Evidence -> Prompt -> Qwen
                logger.info(
                    "[API GATEWAY] Gọi pipeline xử lý với domain=%s, policy=%s",
                    registry_domain_id,
                    policy_mode,
                )

                stage_started_at = time.perf_counter()
                legacy_result = await self._call_legacy_pipeline(
                    query=request_payload.query,
                    allow_external=request_payload.allow_external,
                    system_instruction=system_instruction,
                    request_id=request_id,
                    policy_domain=registry_domain_id,
                    allowed_mcp_tools=list(enforced_policy.allowed_mcp_tools),
                    session_id=_valid_session_id(request.cookies.get("session_id")),
                )
                for cookie_header in legacy_result.session_cookie_headers:
                    response.headers.append("set-cookie", cookie_header)
                pipeline_execution["stage_durations_ms"]["legacy_api"] = round(
                    (time.perf_counter() - stage_started_at) * 1000, 2
                )
                pipeline_execution["legacy_api"] = "completed"
                legacy_pipeline = legacy_result.pipeline
                for key in (
                    "catalog_lookup",
                    "retrieval",
                    "retrieval_collection_status",
                    "reranking",
                    "evidence_builder",
                    "enough_data",
                    "answer_data_sufficiency",
                    "required_fields",
                    "missing_fields",
                    "answer_required_fields",
                    "answer_missing_fields",
                    "external_fallback",
                    "mcp_status",
                    "mcp_raw_result_count",
                    "mcp_mapped_evidence_count",
                    "prompt_assembly",
                    "model_api",
                    "citation_verification",
                    "confidence_status",
                    "confidence_score",
                    "confidence_scorer_status",
                    "confidence_reason",
                    "confidence_reason_code",
                    "verification_threshold",
                    "claim_grounding_status",
                    "grounding_scorer_status",
                    "grounding_reason",
                    "grounding_reason_code",
                    "verification_outcome",
                    "claim_verification_results",
                    "verification_status",
                    "citation_attempts",
                    "citation_attempt_diagnostics",
                    "raw_answer_debug_enabled",
                    "evidence_count",
                    "evidence_ids",
                    "evidence_sources",
                    "cited_evidence_ids",
                    "model_processing_time_ms",
                    "model_api_elapsed_ms",
                    "stage_durations_ms",
                    "retrieved_count",
                    "reranked_count",
                    "failure_stage",
                    "failure_reason_code",
                    "exception_type",
                    "cause_exception_type",
                    "failure_component",
                    "upstream_status_code",
                    "verifier_reason_counts",
                    "stage_status",
                    "response_status",
                ):
                    if key in legacy_pipeline:
                        if key == "stage_durations_ms":
                            pipeline_execution[key].update(legacy_pipeline[key])
                        else:
                            pipeline_execution[key] = legacy_pipeline[key]
                pipeline_execution.setdefault(
                    "verification_threshold",
                    domain_config.policy.verification_threshold,
                )
                pipeline_execution.setdefault("confidence_status", "not_verified")
                pipeline_execution.setdefault(
                    "claim_grounding_status", "not_evaluated"
                )
                pipeline_execution.setdefault("verification_status", "not_verified")
                pipeline_execution["verifier"] = legacy_result.verifier_status
                pipeline_execution["verifier_scope"] = legacy_result.verifier_scope
                pipeline_execution["verifier_status"] = legacy_result.verifier_status
                pipeline_execution["retrieval_evidence_prompt_qwen"] = (
                    "completed"
                    if all(
                        legacy_pipeline.get(key) == "completed"
                        for key in (
                            "retrieval",
                            "reranking",
                            "evidence_builder",
                            "prompt_assembly",
                            "model_api",
                        )
                    )
                    else "partial_or_unknown"
                )

                plan_status = {
                    "Qdrant_Vector_Sparse_Dense_Retrieval": str(
                        legacy_pipeline.get("retrieval", "UNKNOWN")
                    ).upper(),
                    "BGE_Cross_Encoder_Reranking": str(
                        legacy_pipeline.get("reranking", "UNKNOWN")
                    ).upper(),
                    "Self_RAG_Evidence_Verification": "NOT_INTEGRATED",
                    "Qwen_Legal_Generation_With_Citations": (
                        "FAILED"
                        if legacy_pipeline.get("citation_verification") in ("failed", "error")
                        or legacy_result.verifier_status != "passed"
                        or legacy_pipeline.get("model_api") != "completed"
                        else str(legacy_pipeline.get("model_api", "UNKNOWN")).upper()
                    ),
                }
                for step in routing_plan_result.execution_plan:
                    action = step.get("action")
                    if action in plan_status:
                        step["status"] = plan_status[action]

                citation_check_started_at = time.perf_counter()
                failure_reason_code = _citation_integrity_failure_reason(
                    legacy_result
                )
                pipeline_execution.setdefault("stage_durations_ms", {})[
                    "gateway_citation_validation"
                ] = round((time.perf_counter() - citation_check_started_at) * 1000, 2)
                waiting_permission = (
                    legacy_pipeline.get("response_status") == "waiting_permission"
                )
                citation_link_failure = _citation_link_failure_reason(legacy_result)
                pipeline_execution.setdefault("stage_status", {})[
                    "gateway_citation_validation"
                ] = (
                    "not_required" if waiting_permission
                    else "failed" if citation_link_failure
                    else "passed"
                )
                answer_is_verified = failure_reason_code is None and not waiting_permission
                pipeline_execution["gateway_citation_failure_reason_code"] = (
                    failure_reason_code
                )
                reference_failure_reason = (
                    None
                    if answer_is_verified
                    else reference_answer_failure_reason(
                        legacy_result.answer,
                        legacy_pipeline,
                        legacy_result.verifier_status,
                    )
                )
                answer_is_unverified_reference = (
                    not answer_is_verified and reference_failure_reason is None
                )
                pipeline_execution["core_response_outcome"] = legacy_pipeline.get(
                    "response_outcome"
                )
                pipeline_execution["reference_answer_reason_code"] = (
                    None if answer_is_unverified_reference else reference_failure_reason
                )
                response_outcome = classify_business_outcome(
                    pipeline_execution,
                    answer_is_verified=answer_is_verified,
                    answer_is_unverified_reference=answer_is_unverified_reference,
                )
                pipeline_execution["response_outcome"] = response_outcome
                pipeline_execution["core_gateway_outcome_match"] = (
                    legacy_pipeline.get("response_outcome") == response_outcome
                    if legacy_pipeline.get("response_outcome") is not None
                    else None
                )
                answer_text = (
                    legacy_result.answer
                    if answer_is_verified or answer_is_unverified_reference or waiting_permission
                    else _verification_fallback_message(
                        reference_failure_reason
                        if reference_failure_reason
                        and reference_failure_reason.startswith(
                            "REFERENCE_ANSWER_ESSENTIAL_FACTS"
                        )
                        else legacy_pipeline.get("failure_reason_code")
                        or failure_reason_code
                    )
                )
                pipeline_execution["answer_withheld"] = not (
                    answer_is_verified or answer_is_unverified_reference
                )
                if failure_reason_code:
                    core_failure_stage = legacy_pipeline.get("failure_stage")
                    core_failure_reason = legacy_pipeline.get("failure_reason_code")
                    if core_failure_stage:
                        pipeline_execution["upstream_failure_stage"] = core_failure_stage
                    if core_failure_reason:
                        pipeline_execution["upstream_failure_reason_code"] = core_failure_reason
                    pipeline_execution["gateway_citation_failure_reason_code"] = failure_reason_code
                    if isinstance(core_failure_stage, str) and isinstance(core_failure_reason, str):
                        pipeline_execution["failure_stage"] = core_failure_stage
                        pipeline_execution["failure_reason_code"] = core_failure_reason
                    else:
                        pipeline_execution["failure_stage"] = "gateway_citation_validation"
                        pipeline_execution["failure_reason_code"] = failure_reason_code
                    logger.warning(
                        "[PIPELINE STAGE] request_id=%s stage=gateway_citation_validation status=failed reason_code=%s core_failure_stage=%s core_failure_reason_code=%s verifier_status=%s citation_status=%s evidence_count=%s cited_evidence_count=%s",
                        request_id,
                        failure_reason_code,
                        core_failure_stage,
                        core_failure_reason,
                        legacy_result.verifier_status,
                        legacy_pipeline.get("citation_verification", "unknown"),
                        len(legacy_pipeline.get("evidence_ids", []))
                        if isinstance(legacy_pipeline.get("evidence_ids"), list)
                        else 0,
                        len(legacy_pipeline.get("cited_evidence_ids", []))
                        if isinstance(legacy_pipeline.get("cited_evidence_ids"), list)
                        else 0,
                    )

                # BƯỚC 5: Đóng gói kết quả trả về cho Client
                total_processing_ms = round(
                    (time.perf_counter() - start_time) * 1000, 2
                )

                response_data = CoreAPIResponse(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    processing_time_ms=total_processing_ms,
                    status=(
                        "SUCCESS"
                        if answer_is_verified
                        else "FAILED"
                    ),
                    outcome=response_outcome,
                    query_analysis=analysis_result,
                    routing_and_planning=routing_plan_result,
                    next_step=(
                        "WAITING_FOR_EXTERNAL_PERMISSION"
                        if legacy_pipeline.get("response_status")
                        == "waiting_permission"
                        else "PIPELINE_COMPLETED"
                    ),
                    domain_policy={
                        "domain_id": domain_config.domain_id,
                        "mode": domain_config.policy.mode,
                        "verification_threshold": (
                            domain_config.policy.verification_threshold
                        ),
                        "allowed_mcp_tools": (
                            domain_config.policy.allowed_mcp_tools
                        ),
                        "risk_alert_enabled": (
                            domain_config.policy.risk_alert_enabled
                        ),
                    },
                    answer=answer_text,
                    pipeline_execution=pipeline_execution,
                )

                logger.info(
                    "[API GATEWAY] Hoàn tất request %s trong %.2f ms.",
                    request_id,
                    total_processing_ms,
                )
                return response_data

            except HTTPException as ex:
                failure_headers = ex.headers or {}
                logger.warning(
                    "[PIPELINE STAGE] request_id=%s stage=%s status=failed reason_code=%s exception_type=HTTPException upstream_status_code=%s",
                    request_id,
                    failure_headers.get("X-Failure-Stage", "gateway"),
                    failure_headers.get("X-Failure-Reason-Code", "UNKNOWN_PIPELINE_FAILURE"),
                    failure_headers.get("X-Upstream-Status-Code", "unknown"),
                )
                raise

            except Exception as ex:
                logger.error(
                    "[PIPELINE STAGE] request_id=%s stage=gateway status=failed reason_code=UNKNOWN_PIPELINE_FAILURE exception_type=%s",
                    request_id,
                    type(ex).__name__,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Đã xảy ra lỗi nội bộ khi xử lý yêu cầu. Vui lòng cung cấp mã yêu cầu cho nhóm hỗ trợ.",
                    headers={
                        "X-Request-ID": request_id,
                        "X-Failure-Stage": "gateway",
                        "X-Failure-Reason-Code": "UNKNOWN_PIPELINE_FAILURE",
                        "X-Exception-Type": type(ex).__name__,
                    },
                )


# ==============================================================================
# KHỞI CHẠY ỨNG DỤNG TRỰC TIẾP TRÊN Ổ E
# ==============================================================================
gateway_instance = CoreAPIGateway()
app = gateway_instance.app

if __name__ == "__main__":
    gateway_host = os.getenv("CORE_GATEWAY_HOST", "127.0.0.1")
    gateway_port = int(os.getenv("CORE_GATEWAY_PORT", "8002"))
    logger.info(
        "Đang khởi động DAI Legal API Gateway tại http://127.0.0.1:%s ...",
        gateway_port,
    )
    uvicorn.run(
        app,
        host=gateway_host,
        port=gateway_port,
    )
