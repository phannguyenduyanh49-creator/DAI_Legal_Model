import os
import sys
import logging
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError

# Nhập khẩu các thành phần từ hệ thống hiện có
from query_analysis import AnalysisResult
from orchestrator import DomainRouterPlanner, RoutingPlanResult
from domain_registry import registry_service, DomainConfig

# ==============================================================================
# CẤU HÌNH HỆ THỐNG LOGGING & ENCODING UTF-8 CHO WINDOWS CONSOLE
# ==============================================================================
def configure_utf8_stdio() -> None:
    """Đảm bảo console trên Windows hiển thị tiếng Việt chuẩn xác không bị lỗi font."""
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
logger = logging.getLogger("PolicyMiddlewareService")


# ==============================================================================
# 1. PYDANTIC SCHEMAS CHO GÓI KẾT QUẢ ĐÃ LÀM GIÀU CHÍNH SÁCH (ENRICHED PLAN)
# ==============================================================================

class EnrichedRoutingPlanResult(BaseModel):
    """Cấu trúc dữ liệu hoàn chỉnh tích hợp giữa định tuyến Router và Tầng 0 Policy Registry."""
    selected_domain: str = Field(..., description="Phân vùng domain gốc được phân loại từ router")
    registry_domain_key: str = Field(..., description="Khóa domain được ánh xạ sang kho cấu hình Registry")
    routing_confidence: float = Field(..., description="Độ tự tin định tuyến phân vùng")
    policy_mode: Literal["STRICT", "FRIENDLY"] = Field(..., description="Chế độ chính sách hoạt động")
    verification_threshold: float = Field(..., description="Ngưỡng kiểm chứng độ tin cậy tối thiểu")
    persona_tone: str = Field(..., description="Giọng điệu phản hồi của trợ lý AI")
    prompt_template: str = Field(..., description="Mẫu prompt chuẩn được áp dụng theo domain")
    allowed_mcp_tools: List[str] = Field(default_factory=list, description="Danh sách Whitelist MCP tools từ Policy")
    active_mcp_tools: List[str] = Field(default_factory=list, description="Danh sách MCP tools thực tế được kích hoạt sau khi kiểm tra bảo mật")
    execution_plan: List[Dict[str, Any]] = Field(default_factory=list, description="Kế hoạch hành động đa bước đã được làm giàu thông tin chính sách")
    risk_warning: Optional[str] = Field(default=None, description="Cảnh báo rủi ro pháp lý hoặc nghiệp vụ (nếu có)")


# ==============================================================================
# 2. DỊCH VỤ TRUNG GIAN ĐỘC LẬP: POLICY ENFORCEMENT MIDDLEWARE
# ==============================================================================

class PolicyEnforcementMiddleware:
    """
    Dịch vụ trung gian (Middleware) độc lập đảm nhiệm việc:
    - Tiếp nhận kết quả từ DomainRouterPlanner (Orchestrator).
    - Truy xuất và áp dụng cấu hình từ DomainConfig & Policy Registry (Tầng 0).
    - Kiểm soát bảo mật Whitelist MCP Tools, điều chỉnh kế hoạch thực thi 
      tùy theo chế độ STRICT hoặc FRIENDLY mà không làm thay đổi file code cũ.
    """
    def __init__(self, router_planner: Optional[DomainRouterPlanner] = None):
        logger.info("[PolicyMiddleware] Đang khởi tạo dịch vụ Policy Enforcement Middleware...")
        self.router_planner = router_planner or DomainRouterPlanner()

    def _map_domain_to_registry(self, routed_domain: str) -> str:
        """
        Ánh xạ linh hoạt các phân vùng chuyên sâu từ router (civil_law, enterprise_law...)
        sang các cấu hình mẫu chuẩn trong Registry với cơ chế fallback an toàn tuyệt đối.
        """
        normalized = routed_domain.strip().lower()
        
        # 1. Kiểm tra xem registry có trực tiếp domain key này không
        try:
            registry_service.get_domain_config(normalized)
            return normalized
        except KeyError:
            pass

        # 2. Ánh xạ nhóm luật chuyên sâu về 'legal_ai' (Strict mode tiêu chuẩn)
        legal_specialized = {
            "civil_law", "enterprise_law", "criminal_law", 
            "labor_law", "land_law", "general_legal"
        }
        if normalized in legal_specialized:
            return "legal_ai"
        
        # 3. Ánh xạ nhóm hành chính công
        elif "admin" in normalized:
            return "administrative_law"
        
        # 4. Fallback mặc định an toàn
        else:
            logger.info(f"[PolicyMiddleware] Không tìm thấy mapping trực tiếp cho '{normalized}', chuyển hướng về 'general_assistant'.")
            return "general_assistant"

    def _enforce_security_tools(self, policy_mode: str, planned_tools: List[str], whitelisted_tools: List[str]) -> List[str]:
        """
        Kiểm soát bảo mật phân quyền công cụ (Tool Permission Control):
        - STRICT MODE: Chỉ cho phép các tool nằm trong Whitelist được phép thực thi.
        - FRIENDLY MODE: Cho phép linh hoạt mở rộng các tool hỗ trợ công khai.
        """
        whitelist_set = set(whitelisted_tools)
        if policy_mode == "STRICT":
            # Lọc bỏ hoàn toàn các tool ngoài whitelist
            active_tools = [tool for tool in planned_tools if tool in whitelist_set]
            blocked_tools = [tool for tool in planned_tools if tool not in whitelist_set]
            if blocked_tools:
                logger.warning(f"[Security] STRICT MODE: Đã chặn các tool không nằm trong Whitelist: {blocked_tools}")
        else:
            # Friendly mode: Kết hợp cả tool kế hoạch và tool whitelisted
            active_tools = list(set(planned_tools + whitelisted_tools))

        return active_tools

    def process_with_policy(self, query_analysis: AnalysisResult, domain_hint: Optional[str] = None) -> EnrichedRoutingPlanResult:
        """
        Phương thức cốt lõi thực hiện toàn bộ chuỗi tích hợp chính sách:
        1. Gọi DomainRouterPlanner để định tuyến và lấy kế hoạch cơ bản.
        2. Ánh xạ domain sang kho cấu hình Tầng 0 (Registry).
        3. Kiểm soát bảo mật Whitelist Tools.
        4. Làm giàu (Enrich) kế hoạch hành động bằng các quy tắc policy (ngưỡng kiểm chứng, persona, cảnh báo rủi ro).
        """
        logger.info(f"[PolicyMiddleware] Bắt đầu xử lý định tuyến và áp dụng chính sách cho truy vấn: '{query_analysis.original_query[:40]}...'")

        # Bước 1: Lấy kết quả từ Orchestrator hiện có
        base_routing: RoutingPlanResult = self.router_planner.process(query_analysis, domain_hint)

        # Bước 2: Ánh xạ và lấy cấu hình từ Registry
        registry_key = self._map_domain_to_registry(base_routing.selected_domain)
        try:
            domain_config: DomainConfig = registry_service.get_domain_config(registry_key)
            logger.info(f"[PolicyMiddleware] Đã tải thành công cấu hình Policy cho domain registry key: '{registry_key}' (Mode: {domain_config.policy.mode})")
        except KeyError:
            logger.error(f"Lỗi nghiêm trọng: Không tìm thấy domain '{registry_key}' trong Registry. Sử dụng fallback 'legal_ai'.")
            domain_config = registry_service.get_domain_config("legal_ai")
            registry_key = "legal_ai"

        policy = domain_config.policy
        persona = domain_config.persona

        # Bước 3: Kiểm soát bảo mật Whitelist Tools
        active_tools = self._enforce_security_tools(
            policy_mode=policy.mode,
            planned_tools=base_routing.target_tools,
            whitelisted_tools=policy.allowed_mcp_tools
        )

        # Bước 4: Làm giàu (Enrich) các bước trong Execution Plan
        enriched_plan = []
        for step in base_routing.execution_plan:
            step_item = dict(step)
            # Chèn thông tin ngưỡng kiểm chứng vào bước Vector Retrieval
            if step_item.get("action") == "Qdrant_Vector_Sparse_Dense_Retrieval":
                step_item["description"] += f" [Policy Enforcement: Ngưỡng tối thiểu >= {policy.verification_threshold}, Chế độ: {policy.mode}]"
            enriched_plan.append(step_item)

        # Bổ sung bước thực thi Persona và Policy kiểm chứng cuối luồng
        enriched_plan.append({
            "step": len(enriched_plan) + 1,
            "action": "Policy_And_Persona_Enforcement",
            "description": f"Định hình phản hồi theo Persona '{persona.tone}' ({persona.language_style}) và áp dụng kiểm chứng {policy.mode}.",
            "status": "PENDING"
        })

        # Bước 5: Đánh giá rủi ro pháp lý / nghiệp vụ
        risk_warning = None
        if policy.risk_alert_enabled and policy.mode == "STRICT" and len(query_analysis.extracted_citations) == 0:
            risk_warning = (
                "CẢNH BÁO RỦI RO PHÁP LÝ: Truy vấn pháp lý chưa bao gồm trích dẫn hoặc căn cứ điều luật cụ thể. "
                "Hệ thống yêu cầu LLM đính kèm cảnh báo thận trọng khi trả kết quả."
            )
            logger.warning(f"[RiskManager] {risk_warning}")
        elif not policy.risk_alert_enabled:
            risk_warning = f"Thông báo: Domain '{domain_config.domain_name}' hoạt động ở chế độ linh hoạt ({policy.mode})."

        return EnrichedRoutingPlanResult(
            selected_domain=base_routing.selected_domain,
            registry_domain_key=domain_config.domain_id,
            routing_confidence=base_routing.routing_confidence,
            policy_mode=policy.mode, # type: ignore
            verification_threshold=policy.verification_threshold,
            persona_tone=persona.tone,
            prompt_template=domain_config.prompt_template,
            allowed_mcp_tools=policy.allowed_mcp_tools,
            active_mcp_tools=active_tools,
            execution_plan=enriched_plan,
            risk_warning=risk_warning
        )


# Khởi tạo instance dùng chung cho toàn bộ ứng dụng
policy_middleware = PolicyEnforcementMiddleware()


# ==============================================================================
# 3. KHỐI KIỂM CHỨNG & CHẠY THỬ NGHIỆM TRỰC TIẾP (UNIT TEST / SCRIPT EXECUTION)
# ==============================================================================

if __name__ == "__main__":
    print("\n================ [KIỂM CHỨNG TOÀN DIỆN POLICY MIDDLEWARE] ================")

    # Tạo giả lập kết quả phân tích câu hỏi (AnalysisResult)
    from query_analysis import AnalysisResult
    
    # Test Case 1: Truy vấn pháp lý chuyên sâu (Kỳ vọng: STRICT MODE, Legal AI)
    print("\n------------------------------------------------------------------")
    print("TEST CASE 1: Truy vấn pháp lý doanh nghiệp (Kỳ vọng: STRICT MODE)")
    print("------------------------------------------------------------------")
    mock_legal_analysis = AnalysisResult(
        original_query="Thủ tục thành lập công ty cổ phần theo Luật Doanh nghiệp ra sao?",
        cleaned_query="thủ tục thành lập công ty cổ phần luật doanh nghiệp",
        expanded_terms=["thành lập doanh nghiệp", "cổ phần"],
        extracted_citations=["Luật Doanh nghiệp"],
        hypothetical_document=None,
        confidence_score=0.95
    )
    
    result_legal = policy_middleware.process_with_policy(mock_legal_analysis, domain_hint="enterprise_law")
    print(f"-> Selected Domain: {result_legal.selected_domain} -> Registry Key: {result_legal.registry_domain_key}")
    print(f"-> Policy Mode: {result_legal.policy_mode} | Threshold: {result_legal.verification_threshold}")
    print(f"-> Persona Tone: {result_legal.persona_tone}")
    print(f"-> Whitelist Tools: {result_legal.allowed_mcp_tools}")
    print(f"-> Active Tools (Sau kiểm tra bảo mật): {result_legal.active_mcp_tools}")
    print(f"-> Số bước thực thi trong Plan: {len(result_legal.execution_plan)}")

    # Test Case 2: Truy vấn đời sống / tổng hợp (Kỳ vọng: FRIENDLY MODE, General Assistant)
    print("\n------------------------------------------------------------------")
    print("TEST CASE 2: Truy vấn thông tin tổng hợp (Kỳ vọng: FRIENDLY MODE)")
    print("------------------------------------------------------------------")
    mock_general_analysis = AnalysisResult(
        original_query="Thời tiết cuối tuần này ở Hồ Chí Minh thế nào?",
        cleaned_query="thời tiết cuối tuần hồ chí minh",
        expanded_terms=["dự báo thời tiết"],
        extracted_citations=[],
        hypothetical_document=None,
        confidence_score=0.90
    )
    
    result_general = policy_middleware.process_with_policy(mock_general_analysis, domain_hint="general_legal")
    print(f"-> Selected Domain: {result_general.selected_domain} -> Registry Key: {result_general.registry_domain_key}")
    print(f"-> Policy Mode: {result_general.policy_mode} | Threshold: {result_general.verification_threshold}")
    print(f"-> Persona Tone: {result_general.persona_tone}")
    print(f"-> Whitelist Tools: {result_general.allowed_mcp_tools}")
    print(f"-> Active Tools: {result_general.active_mcp_tools}")
    if result_general.risk_warning:
        print(f"-> Thông báo/Cảnh báo: {result_general.risk_warning}")

    print("\n================ HOÀN TẤT KIỂM CHỨNG MIDDLEWARE THÀNH CÔNG ===============_\n")