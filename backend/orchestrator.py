
import logging
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
from backend.query_analysis import AnalysisResult
from backend.domain_registry import registry_service

logger = logging.getLogger("DAI_Legal_Orchestrator")


class RoutingPlanResult(BaseModel):
    selected_domain: str = Field(
        ..., description="Domain được phân loại"
    )
    routing_confidence: float = Field(
        ..., description="Độ tự tin của thuật toán định tuyến domain"
    )
    execution_plan: List[Dict[str, Any]] = Field(
        ..., description="Kế hoạch hành động chi tiết dạng đa bước"
    )
    target_tools: List[str] = Field(
        ..., description="Danh sách Agent Tools / Micro-services cần kích hoạt"
    )


class DomainRouterPlanner:
    """
    Bộ điều phối trung tâm:
    1. Domain Router: Phân loại câu hỏi theo domain.
    2. Agent Planner: Lập kế hoạch thực thi.
    """

    def __init__(self):
        logger.info(
            "[Module 2 - Orchestrator] "
            "Khởi tạo Domain Router và Agent Planner..."
        )

        # Các domain legacy được giữ để tương thích code cũ.
        self.domain_keyword_registry: Dict[str, List[str]] = {
            "civil_law": [
                "hợp đồng", "thừa kế", "ly hôn", "bồi thường",
                "tài sản", "dân sự", "vay mượn", "di chúc", "nợ nần",
            ],
            "enterprise_law": [
                "doanh nghiệp", "công ty", "cổ phần", "góp vốn",
                "đăng ký kinh doanh", "phá sản", "giám đốc",
                "hội đồng quản trị", "mã số thuế",
            ],
            "criminal_law": [
                "tội phạm", "hình sự", "hành vi vi phạm", "truy tố",
                "xét xử", "tù giam", "cấu thành tội phạm",
                "tang vật", "cố ý gây thương tích",
            ],
            "administrative_law": [
                "thủ tục hành chính", "khiếu nại", "tố cáo",
                "giấy phép", "cơ quan nhà nước",
                "xử phạt vi phạm hành chính", "ubnd",
                "quyết định hành chính",
            ],
            "labor_law": [
                "hợp đồng lao động", "sa thải", "lương thưởng",
                "bảo hiểm xã hội", "thôi việc", "đình công",
                "tiền lương", "giờ làm việc",
            ],
            "land_law": [
                "đất đai", "sổ đỏ", "sổ hồng",
                "quyền sử dụng đất", "thu hồi đất",
                "giải phóng mặt bằng", "tranh chấp ranh giới",
                "sang nhượng nhà đất",
            ],
            "tourism": [
                "du lịch", "điểm du lịch", "khách sạn",
                "đặt phòng", "giá vé", "địa điểm tham quan",
                "lịch trình", "resort", "homestay", "tour",
                "điểm đến", "nhà hàng", "vé tham quan",
            ],
            "general_legal": [],
        }

        # Ánh xạ domain pháp lý legacy về ID Registry hiện có.
        self.legacy_domain_mapping: Dict[str, str] = {
            "civil_law": "legal_ai",
            "enterprise_law": "legal_ai",
            "criminal_law": "legal_ai",
            "labor_law": "legal_ai",
            "land_law": "legal_ai",
            "general_legal": "legal_ai",
            "legal_ai": "legal_ai",
            "administrative_law": "administrative_law",
            "general_assistant": "general_assistant",
            "tourism": "tourism",
        }

    def _get_registered_domain(self, domain_id: str) -> Optional[str]:
        """Trả về domain ID nếu tồn tại trong Registry."""
        normalized_id = domain_id.strip().lower()
        try:
            registry_service.get_domain_config(normalized_id)
            return normalized_id
        except (KeyError, AttributeError):
            return None

    def route_domain(
        self,
        query_analysis: AnalysisResult,
        domain_hint: Optional[str] = None,
    ) -> tuple[str, float]:
        """Định tuyến theo hint hợp lệ hoặc nội dung câu hỏi."""

        # 1. Ưu tiên domain hint nếu domain tồn tại trong Registry.
        normalized_hint = (
            domain_hint.strip().lower()
            if isinstance(domain_hint, str)
            else ""
        )

        if normalized_hint:
            mapped_hint = self.legacy_domain_mapping.get(
                normalized_hint, normalized_hint
            )
            registered_domain = self._get_registered_domain(mapped_hint)

            if registered_domain:
                logger.info(
                    "[Domain Router] Hint '%s' -> '%s'",
                    normalized_hint,
                    registered_domain,
                )
                return registered_domain, 1.0

            logger.warning(
                "[Domain Router] Domain hint không hợp lệ: %r. "
                "Định tuyến theo nội dung.",
                domain_hint,
            )

        # 2. Chuẩn bị nội dung để phân loại.
        target_corpus = (
            query_analysis.cleaned_query
            + " "
            + " ".join(query_analysis.expanded_terms)
            + " "
            + " ".join(query_analysis.extracted_citations)
        ).lower()

        # 3. Tính điểm từ khóa.
        domain_scores: Dict[str, int] = {}

        for domain_id, keywords in self.domain_keyword_registry.items():
            if not keywords:
                continue

            score = sum(
                1 for keyword in keywords
                if keyword in target_corpus
            )

            if score > 0:
                domain_scores[domain_id] = score

        if not domain_scores:
            logger.warning(
                "[Domain Router] Không khớp từ khóa. "
                "Fallback về legal_ai."
            )
            return "legal_ai", 0.50

        # 4. Chọn domain có điểm từ khóa cao nhất.
        best_domain = max(
            domain_scores,
            key=domain_scores.get,
        )

        raw_score = domain_scores[best_domain]
        selected_domain = self.legacy_domain_mapping.get(
            best_domain, best_domain
        )

        # Chỉ chọn domain nếu domain đã được đăng ký.
        if not self._get_registered_domain(selected_domain):
            logger.warning(
                "[Domain Router] Domain '%s' chưa được đăng ký. "
                "Fallback về legal_ai.",
                selected_domain,
            )
            return "legal_ai", 0.50

        confidence = round(
            min(0.98, 0.55 + raw_score * 0.12), 2
        )

        logger.info(
            "[Domain Router] Nội dung '%s' -> '%s', confidence=%s",
            best_domain,
            selected_domain,
            confidence,
        )

        return selected_domain, confidence

    def create_execution_plan(
        self,
        domain: str,
        query_analysis: AnalysisResult,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Tạo kế hoạch thực thi; giữ pipeline retrieval hiện tại."""

        logger.info(
            "[Agent Planner] Lập kế hoạch cho domain [%s]",
            domain,
        )

        has_hyde = query_analysis.hypothetical_document is not None

        plan = [
            {
                "step": 1,
                "action": "Query_Analysis_And_HyDE",
                "description": (
                    "Đã hoàn tất phân tích query và sinh HyDE "
                    f"(Trạng thái: {'Bật' if has_hyde else 'Tắt'})."
                ),
                "status": "COMPLETED",
            },
            {
                "step": 2,
                "action": "Qdrant_Vector_Sparse_Dense_Retrieval",
                "description": (
                    "Truy xuất dữ liệu theo domain "
                    f"[{domain}]. Cần bảo đảm retriever thực sự "
                    "lọc dữ liệu theo domain này."
                ),
                "status": "PENDING",
            },
            {
                "step": 3,
                "action": "BGE_Cross_Encoder_Reranking",
                "description": (
                    "Sắp xếp các đoạn truy xuất theo độ phù hợp."
                ),
                "status": "PENDING",
            },
            {
                "step": 4,
                "action": "Self_RAG_Evidence_Verification",
                "description": (
                    "Kiểm tra mức độ đầy đủ và phù hợp của ngữ cảnh."
                ),
                "status": "PENDING",
            },
            {
                "step": 5,
                "action": "Qwen_Legal_Generation_With_Citations",
                "description": (
                    "Sinh câu trả lời dựa trên ngữ cảnh truy xuất."
                ),
                "status": "PENDING",
            },
        ]

        target_tools = [
            "QdrantHybridRetriever",
            "BGERerankerService",
            "SelfRAGVerifierEngine",
            "LegalCitationFormatter",
        ]

        if domain == "administrative_law":
            target_tools.append("LegalPrecedentSearchTool")

        return plan, target_tools

    def process(
        self,
        query_analysis: AnalysisResult,
        domain_hint: Optional[str] = None,
    ) -> RoutingPlanResult:
        """Định tuyến domain và tạo kế hoạch tác vụ."""

        selected_domain, confidence = self.route_domain(
            query_analysis,
            domain_hint,
        )

        execution_plan, target_tools = self.create_execution_plan(
            selected_domain,
            query_analysis,
        )

        return RoutingPlanResult(
            selected_domain=selected_domain,
            routing_confidence=confidence,
            execution_plan=execution_plan,
            target_tools=target_tools,
        )
