import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from backend.query_analysis import AnalysisResult

logger = logging.getLogger("DAI_Legal_Orchestrator")

class RoutingPlanResult(BaseModel):
    selected_domain: str = Field(..., description="Lĩnh vực pháp lý được phân loại")
    routing_confidence: float = Field(..., description="Độ tự tin của thuật toán định tuyến domain")
    execution_plan: List[Dict[str, Any]] = Field(..., description="Kế hoạch hành động chi tiết dạng đa bước (Planner Steps)")
    target_tools: List[str] = Field(..., description="Danh sách các Agent Tools / Micro-services cần kích hoạt")

class DomainRouterPlanner:
    """
    Bộ điều phối trung tâm (Orchestrator Engine):
    1. Domain Router: Phân loại câu hỏi vào đúng phân vùng pháp luật chuyên biệt.
    2. Agent Planner: Xây dựng sơ đồ thực thi pipeline tìm kiếm và sinh câu trả lời.
    """
    def __init__(self):
        logger.info("[Module 2 - Orchestrator] Đang khởi tạo Domain Router và Agent Planner...")

        # Bảng ánh xạ từ khóa phân vùng lĩnh vực luật chi tiết
        self.domain_keyword_registry: Dict[str, List[str]] = {
            "civil_law": ["hợp đồng", "thừa kế", "ly hôn", "bồi thường", "tài sản", "dân sự", "vay mượn", "di chúc", "nợ nần"],
            "enterprise_law": ["doanh nghiệp", "công ty", "cổ phần", "góp vốn", "đăng ký kinh doanh", "phá sản", "giám đốc", "hội đồng quản trị", "mã số thuế"],
            "criminal_law": ["tội phạm", "hình sự", "hành vi vi phạm", "truy tố", "xét xử", "tù giam", "cấu thành tội phạm", "tang vật", "cố ý gây thương tích"],
            "administrative_law": ["thủ tục hành chính", "khiếu nại", "tố cáo", "giấy phép", "cơ quan nhà nước", "xử phạt vi phạm hành chính", "UBND", "quyết định hành chính"],
            "labor_law": ["hợp đồng lao động", "sa thải", "lương thưởng", "bảo hiểm xã hội", "thôi việc", "đình công", "tiền lương", "giờ làm việc"],
            "land_law": ["đất đai", "sổ đỏ", "sổ hồng", "quyền sử dụng đất", "thu hồi đất", "giải phóng mặt bằng", "tranh chấp ranh giới", "sang nhượng nhà đất"],
            "general_legal": [],
            "general_assistant": []
        }

    def route_domain(self, query_analysis: AnalysisResult, domain_hint: Optional[str] = None) -> tuple[str, float]:
        """Phân loại domain dựa trên gợi ý thủ công hoặc phân tích từ khóa tần suất cao."""
        # 1. Kiểm tra domain hint từ client (nếu có và hợp lệ)
        if domain_hint:
            normalized_hint = domain_hint.strip().lower()
            if normalized_hint in self.domain_keyword_registry:
                logger.info(f"[Domain Router] Ưu tiên sử dụng Domain Hint từ Client: '{normalized_hint}' (Confidence: 1.0)")
                return normalized_hint, 1.0

        # 2. Tổng hợp văn bản phân tích để quét từ khóa
        target_corpus = (
            query_analysis.cleaned_query + " " +
            " ".join(query_analysis.expanded_terms) + " " +
            " ".join(query_analysis.extracted_citations)
        ).lower()

        domain_scores: Dict[str, int] = {}
        for domain, keywords in self.domain_keyword_registry.items():
            if not keywords:
                continue
            score = sum(1 for kw in keywords if kw in target_corpus)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            logger.warning("[Domain Router] Không khớp từ khóa chuyên sâu nào. Chuyển hướng về phân vùng 'general_legal'.")
            return "general_legal", 0.50

        # Chọn domain có điểm số cao nhất
        best_domain = max(domain_scores, key=domain_scores.get)
        raw_score = domain_scores[best_domain]

        # Tính toán điểm confidence tương đối (giới hạn từ 0.60 đến 0.98)
        confidence = round(min(0.98, 0.55 + (raw_score * 0.12)), 2)

        logger.info(f"[Domain Router] Đã phân loại thành công vào Domain: '{best_domain}' với điểm tin cậy {confidence}")
        return best_domain, confidence

    def create_execution_plan(self, domain: str, query_analysis: AnalysisResult) -> tuple[List[Dict[str, Any]], List[str]]:
        """Lập kế hoạch hành động chi tiết (Agent Planner) theo chuẩn ReAct/Plan-and-Solve đa bước."""
        logger.info(f"[Agent Planner] Đang biên soạn chuỗi kế hoạch thực thi cho phân vùng: [{domain}]")

        has_hyde = query_analysis.hypothetical_document is not None

        plan = [
            {
                "step": 1,
                "action": "Query_Analysis_And_HyDE",
                "description": f"Đã hoàn tất làm sạch query, bóc tách {len(query_analysis.extracted_citations)} điều luật và sinh HyDE (Trạng thái: {'Bật' if has_hyde else 'Tắt'}).",
                "status": "COMPLETED"
            },
            {
                "step": 2,
                "action": "Qdrant_Vector_Sparse_Dense_Retrieval",
                "description": f"Truy xuất Hybrid Search (Dense + Sparse embeddings) từ Qdrant Vector DB giới hạn trong collection [{domain}].",
                "status": "PENDING"
            },
            {
                "step": 3,
                "action": "BGE_Cross_Encoder_Reranking",
                "description": "Đưa các đoạn văn bản luật trích xuất qua mô hình BGE-Reranker để lọc nhiễu và sắp xếp theo độ phù hợp cao nhất.",
                "status": "PENDING"
            },
            {
                "step": 4,
                "action": "Self_RAG_Evidence_Verification",
                "description": "Kiểm chứng mức độ đầy đủ của ngữ cảnh, xác minh các điều luật đối chiếu có chính xác với câu hỏi không.",
                "status": "PENDING"
            },
            {
                "step": 5,
                "action": "Qwen_Legal_Generation_With_Citations",
                "description": "Tổng hợp ngữ cảnh đã rerank để LLM (Qwen 7B) sinh câu trả lời tư vấn pháp lý hoàn chỉnh kèm trích dẫn điều khoản.",
                "status": "PENDING"
            }
        ]

        # Khai báo các tool hệ thống cần gọi tương ứng trong pipeline
        target_tools = [
            "QdrantHybridRetriever",
            "BGERerankerService",
            "SelfRAGVerifierEngine",
            "LegalCitationFormatter"
        ]

        if domain == "enterprise_law" or domain == "administrative_law":
            target_tools.append("LegalPrecedentSearchTool")

        return plan, target_tools

    def process(self, query_analysis: AnalysisResult, domain_hint: Optional[str] = None) -> RoutingPlanResult:
        """Thực thi toàn bộ quy trình định tuyến domain và tạo lập kế hoạch tác vụ."""
        selected_domain, conf = self.route_domain(query_analysis, domain_hint)
        execution_plan, target_tools = self.create_execution_plan(selected_domain, query_analysis)

        return RoutingPlanResult(
            selected_domain=selected_domain,
            routing_confidence=conf,
            execution_plan=execution_plan,
            target_tools=target_tools
        )
