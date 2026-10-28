import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.query_analysis import AnalysisResult
from backend.domain_registry import DomainRegistryService, registry_service

logger = logging.getLogger("DAI_Legal_Orchestrator")

class RoutingPlanResult(BaseModel):
    selected_domain: Optional[str] = Field(..., description="Canonical domain/route alias, or null when clarification is required")
    routing_confidence: float = Field(..., description="Heuristic route score; not a calibrated probability")
    routing_reason: str = ""
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    execution_plan: List[Dict[str, Any]] = Field(..., description="Kế hoạch hành động chi tiết dạng đa bước (Planner Steps)")
    target_tools: List[str] = Field(..., description="Danh sách các Agent Tools / Micro-services cần kích hoạt")

class DomainRouterPlanner:
    """Classify with the active domain registry's routing rules."""

    MINIMUM_ROUTE_SCORE = 2
    AMBIGUITY_MARGIN = 1

    def __init__(self, registry: DomainRegistryService = registry_service):
        self.registry = registry
        if self.registry.fallback_domain_id not in self.registry.list_enabled_domains():
            raise ValueError("The configured fallback domain must be enabled.")

    @staticmethod
    def _score_rule(rule, corpus: str) -> int:
        strong_matches = sum(term in corpus for term in rule.strong_keywords)
        regular_matches = sum(term in corpus for term in rule.keywords)
        score = strong_matches * 2 + regular_matches
        # A clearly expressed legal intent wins over generic business language.
        if not strong_matches:
            score -= 2 * sum(term in corpus for term in rule.exclusions)
        return score

    def _route_decision(
        self,
        query_analysis: AnalysisResult,
        domain_hint: Optional[str],
    ) -> tuple[Optional[str], float, str, bool]:
        corpus = " ".join([
            query_analysis.cleaned_query,
            *query_analysis.expanded_terms,
            *query_analysis.extracted_citations,
        ]).casefold()
        scored = []
        strong_categories = set()
        for owner_domain, rule in self.registry.iter_routing_rules():
            score = self._score_rule(rule, corpus)
            if any(term in corpus for term in rule.strong_keywords):
                strong_categories.add(rule.category)
            if score >= self.MINIMUM_ROUTE_SCORE:
                scored.append({
                    "score": score,
                    "priority": rule.priority,
                    "category": rule.category,
                    "route_id": rule.route_id,
                    "owner": owner_domain,
                })
        scored.sort(key=lambda item: (item["score"], item["priority"]), reverse=True)

        category_scores = {}
        for candidate in scored:
            category_scores[candidate["category"]] = max(
                category_scores.get(candidate["category"], 0), candidate["score"]
            )

        tied_categories = [
            category for category, score in category_scores.items()
            if category != max(category_scores, key=category_scores.get)
            and max(category_scores.values()) - score <= self.AMBIGUITY_MARGIN
        ] if category_scores else []
        competing_strong_categories = (
            len(strong_categories) > 1
            and {"legal", "tourism"}.issubset(strong_categories)
        )

        if domain_hint:
            hinted_domain = self.registry.route_domain_id(domain_hint)
            hinted_rules = [
                rule for owner, rule in self.registry.iter_routing_rules()
                if owner == hinted_domain and rule.route_id == domain_hint.strip().lower()
            ]
            hinted_categories = {rule.category for rule in hinted_rules}
            if not hinted_categories:
                hinted_categories = {
                    rule.category for owner, rule in self.registry.iter_routing_rules()
                    if owner == hinted_domain
                }
            conflicting = [
                item for item in scored
                if item["owner"] != hinted_domain
                and (not hinted_categories or item["category"] not in hinted_categories)
            ]
            if conflicting:
                best_conflict = conflicting[0]
                if tied_categories or competing_strong_categories:
                    return None, 0.0, "AMBIGUOUS_DOMAIN_SIGNALS", True
                logger.info(
                    "[Domain Router] Content overrides stale domain hint '%s' with route '%s'.",
                    domain_hint,
                    best_conflict["route_id"],
                )
                return (
                    best_conflict["route_id"],
                    float(best_conflict["score"]),
                    "CONTENT_OVERRIDES_STALE_HINT",
                    False,
                )
            return domain_hint.strip().lower(), 1.0, "DOMAIN_HINT", False

        if not scored:
            return (
                self.registry.fallback_domain_id,
                0.0,
                "NO_DOMAIN_RULE_MATCH",
                False,
            )
        if tied_categories or competing_strong_categories:
            return None, 0.0, "AMBIGUOUS_DOMAIN_SIGNALS", True
        return (
            scored[0]["route_id"],
            float(scored[0]["score"]),
            "REGISTRY_KEYWORD_RULE",
            False,
        )

    def route_domain(
        self,
        query_analysis: AnalysisResult,
        domain_hint: Optional[str] = None,
    ) -> tuple[Optional[str], float]:
        domain, score, _reason, _clarification = self._route_decision(
            query_analysis, domain_hint
        )
        return domain, score

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
        selected_domain, score, reason, clarification_required = self._route_decision(
            query_analysis, domain_hint
        )
        if clarification_required or selected_domain is None:
            execution_plan, target_tools = [], []
        else:
            execution_plan, target_tools = self.create_execution_plan(
                selected_domain, query_analysis
            )

        return RoutingPlanResult(
            selected_domain=selected_domain,
            routing_confidence=score,
            routing_reason=reason,
            clarification_required=clarification_required,
            clarification_question=(
                "Câu hỏi có dấu hiệu thuộc nhiều lĩnh vực. Bạn muốn tra cứu theo pháp luật hay du lịch?"
                if clarification_required else None
            ),
            execution_plan=execution_plan,
            target_tools=target_tools
        )
