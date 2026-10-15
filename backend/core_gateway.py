import os
import sys
import time
import logging
import json
import httpx
from datetime import datetime
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend import answer


from backend.query_analysis import QueryAnalysisService, AnalysisResult
from backend.orchestrator import DomainRouterPlanner, RoutingPlanResult
from backend.domain_registry import registry_service, domain_router
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
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Khởi tạo các Service nghiệp vụ cốt lõi

        self.query_analyzer = QueryAnalysisService()
        self.orchestrator = DomainRouterPlanner()

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
    ) -> str:
        """Gọi Core API cũ để tái sử dụng pipeline Retrieval/Evidence/Prompt/Qwen."""
        payload = {
            "prompt": query,
            "allow_external": allow_external,
            "system_instruction": system_instruction,
            "max_new_tokens": 512,
        }

        try:
            async with httpx.AsyncClient(timeout=210.0) as client:
                response = await client.post(
                    self.legacy_api_url,
                    json=payload,
                )
            response.raise_for_status()

        except httpx.HTTPError as exc:
            logger.exception(
                "Không gọi được Core API cũ tại %s",
                self.legacy_api_url,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Không thể gọi pipeline Retrieval/Evidence/Qwen hiện có. "
                    "Hãy kiểm tra Core API trên cổng 8000 và "
                    "biến môi trường LEGACY_CORE_API_URL."
                ),
            ) from exc

        # Core API cũ có thể trả về Server-Sent Events (SSE).
        # Gom các delta thành một câu trả lời hoàn chỉnh.
        answer_parts: list[str] = []

        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue

            raw = line[len("data:"):].strip()

            try:
                event_data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            delta = event_data.get("delta")
            if isinstance(delta, str):
                answer_parts.append(delta)

        answer = "".join(answer_parts).strip()

        # Hỗ trợ trường hợp Core API trả JSON thông thường thay vì SSE.
        if not answer:
            try:
                result = response.json()
                if isinstance(result, dict):
                    answer_value = result.get("answer")
                    if isinstance(answer_value, str):
                        answer = answer_value.strip()
            except ValueError:
                pass

        if not answer:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Core API đã phản hồi nhưng không trích xuất được "
                    "nội dung câu trả lời."
                ),
            )

        return answer

    def _register_middleware_and_routes(self):
        # Middleware đo lường hiệu năng thực thi (Latency Measurement)
        @self.app.middleware("http")
        async def add_process_latency_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            response.headers["X-Gateway-Latency-MS"] = str(duration_ms)
            return response

        # Endpoint kiểm tra sức khỏe hệ thống (Health Check)
        @self.app.get("/health", tags=["System Diagnostics"])
        async def health_check():
            return {
                "status": "HEALTHY",
                "service_name": "DAI-Legal-Model Core Gateway API",
                "environment": "Production-Ready Local (Drive E)",
                "timestamp": datetime.utcnow().isoformat()
            }

        # Endpoint xử lý câu hỏi pháp lý chính thức qua 3 Module đầu tiên
        @self.app.post("/api/v2/legal-query", response_model=CoreAPIResponse, tags=["Core Legal Processing Pipeline"])
        async def process_legal_consultation_query(request_payload: QueryRequest):
            start_time = time.time()
            request_id = f"trace_legal_{int(time.time() * 1000)}"

            logger.info(f"\n==================================================")
            logger.info(f"[API GATEWAY] NHẬN YÊU CẦU MỚI - Request ID: {request_id}")
            logger.info(f"Nội dung truy vấn: '{request_payload.query}'")
            if request_payload.domain_hint:
                logger.info(f"Gợi ý Domain từ Client: '{request_payload.domain_hint}'")
            logger.info(f"Cơ chế HyDE: {'Đang bật' if request_payload.enable_hyde else 'Đang tắt'}")
            logger.info(f"==================================================")

            try:
                # BƯỚC 1: Gọi Module Query Analysis & Clarification (Thay thế Synonym Mapping)
                analysis_result = self.query_analyzer.analyze(
                    query=request_payload.query,
                    enable_hyde=request_payload.enable_hyde
                )

                # BƯỚC 2: Gọi Module Orchestrator (Domain Router + Agent Planner)
                routing_plan_result = self.orchestrator.process(
                    query_analysis=analysis_result,
                    domain_hint=request_payload.domain_hint
                )


                # BƯỚC 3: Đọc Domain Registry và áp dụng Policy
                selected_domain = routing_plan_result.selected_domain
                available_domains = registry_service.list_all_domains()

                # Orchestrator có thể trả về domain con chưa được đăng ký.
                # Khi đó, dùng cấu hình pháp lý tổng quát legal_ai.
                if selected_domain in available_domains:
                    registry_domain_id = selected_domain
                else:
                    registry_domain_id = "legal_ai"

                domain_config = registry_service.get_domain_config(
                    registry_domain_id
                )

                policy_mode = domain_config.policy.mode
                persona_instruction = (
                    domain_config.persona.system_prompt_addition
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

                # BƯỚC 4: Gọi pipeline Retrieval -> Evidence -> Prompt -> Qwen
                logger.info(
                    "[API GATEWAY] Gọi pipeline xử lý với domain=%s, policy=%s",
                    registry_domain_id,
                    policy_mode,
                )

                answer_text = await self._call_legacy_pipeline(
                    query=request_payload.query,
                    allow_external=request_payload.allow_external,
                    system_instruction=system_instruction,
                )

                # BƯỚC 5: Đóng gói kết quả trả về cho Client
                total_processing_ms = round(
                    (time.time() - start_time) * 1000, 2
                )

                response_data = CoreAPIResponse(
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    processing_time_ms=total_processing_ms,
                    status="SUCCESS",
                    query_analysis=analysis_result,
                    routing_and_planning=routing_plan_result,
                    next_step="PIPELINE_COMPLETED",
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
                    pipeline_execution={
                        "query_analysis": "completed",
                        "orchestrator": "completed",
                        "domain_policy": "resolved",
                        "retrieval_evidence_prompt_qwen": "delegated_to_legacy_api",
                        "external_search_allowed": (
                            request_payload.allow_external
                        ),
                        "verifier": "not_integrated_in_this_gateway_path",
                    },
                )

                logger.info(
                    "[API GATEWAY] Hoàn tất request %s trong %.2f ms.",
                    request_id,
                    total_processing_ms,
                )
                return response_data

            except HTTPException:
                raise

            except Exception as ex:
                logger.error(f"[API GATEWAY ERROR] Lỗi nghiêm trọng khi xử lý request {request_id}: {str(ex)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Lỗi hệ thống nội bộ tại Gateway: {str(ex)}"
                )


# ==============================================================================
# KHỞI CHẠY ỨNG DỤNG TRỰC TIẾP TRÊN Ổ E
# ==============================================================================
gateway_instance = CoreAPIGateway()
app = gateway_instance.app

if __name__ == "__main__":
    logger.info("Đang khởi động máy chủ DAI Legal Model Core Server tại http://127.0.0.1:8000 ...")
    uvicorn.run(
        "core_gateway:app",  # <--- Sửa chính xác thành tên file hiện tại là "core_gateway:app"
        app_dir=os.path.dirname(os.path.abspath(__file__)),
        host="127.0.0.1",
        port=8000,
        reload=True,
    )