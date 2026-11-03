from typing import Optional
from dataclasses import replace

from backend.evidence.schemas import EvidenceBundle
from backend.answer.citation_contract import CITATION_RULES, format_allowed_evidence_ids


class PromptAssembler:
    def __init__(self, max_evidence_chars: int = 12000):
        self.max_evidence_chars = max_evidence_chars

    def build(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        system_instruction: Optional[str] = None,
        domain_prompt_template: Optional[str] = None,
    ) -> str:
        """Build a grounded legal-answer prompt using only included evidence."""
        system_part = system_instruction or (
            "Bạn là trợ lý AI hỗ trợ tra cứu thủ tục hành chính và thông tin pháp lý tại Việt Nam.\n"
            "Chỉ sử dụng dữ kiện được nêu trong phần Evidence bên dưới để trả lời. "
            "Không bịa đặt hoặc tự bổ sung điều luật, nghị định, thời hạn, lệ phí, "
            "thành phần hồ sơ, điều kiện hay cơ quan giải quyết.\n"
            "Trước khi trả lời, xác định những phần của câu hỏi mà Evidence thực sự hỗ trợ. "
            "Nếu câu hỏi yêu cầu quy trình, hãy trình bày theo thứ tự từng bước khi Evidence "
            "có đủ căn cứ; nếu Evidence chỉ mô tả một phần quy trình, nêu rõ đó là phần "
            "được nguồn cung cấp và không khẳng định là toàn bộ quy trình.\n"
            "Tổng hợp các Evidence liên quan thay vì chỉ nhắc lại một câu trích riêng lẻ. "
            "Không gộp các thủ tục hoặc loại hình pháp lý khác nhau nếu nguồn không xác nhận chúng giống nhau.\n"
            "Mỗi nhận định thực tế hoặc pháp lý phải có trích dẫn Evidence ID chính xác. "
            "Chỉ sử dụng các ID nằm trong danh sách Allowed Evidence IDs; không tự tạo hoặc sửa ID.\n"
            "Nếu Evidence không đủ để trả lời một ý, hãy nói rõ thông tin đó chưa được nguồn cung cấp. "
            "Không đoán và không trình bày suy luận như một sự thật đã được xác minh.\n"
            "Trả lời bằng tiếng Việt rõ ràng, trực tiếp; ưu tiên danh sách đánh số khi câu hỏi hỏi về các bước.\n"
        )

        included = self.included_evidences(evidence_bundle)
        allowed_ids = format_allowed_evidence_ids(
            item.evidence_id for item in included
        )
        evidence_context = self._format_evidences(included)
        # Render domain templates with the same placeholders used by the
        # existing contract. A malformed template is ignored safely so it
        # cannot leak raw placeholders into the model prompt.
        domain_prompt = ""
        if domain_prompt_template:
            try:
                domain_prompt = domain_prompt_template.format(
                    query=str(question).strip(),
                    context="the shared EVIDENCE section above",
                ).strip()
            except (KeyError, IndexError, ValueError, AttributeError):
                domain_prompt = ""

        parts = [
            system_part.strip(),
            "=== USER QUESTION ===\n" + str(question).strip(),
            "=== EVIDENCE ===\n" + evidence_context,
        ]
        if domain_prompt:
            parts.append("=== DOMAIN-SPECIFIC INSTRUCTIONS ===\n" + domain_prompt)
        parts.extend(
            [
                "=== REQUIRED CITATION FORMAT ===\n" + CITATION_RULES,
                "Allowed Evidence IDs: " + allowed_ids,
                (
                    "=== RESPONSE CHECKLIST ===\n"
                    "- Trả lời đúng trọng tâm câu hỏi.\n"
                    "- Với câu hỏi về quy trình, liệt kê các bước theo thứ tự nếu Evidence hỗ trợ.\n"
                    "- Gắn Evidence ID chính xác cho từng nhận định có căn cứ.\n"
                    "- Nêu rõ phần nào chưa có đủ căn cứ; không tự điền thông tin còn thiếu.\n"
                    "- Không tuyên bố câu trả lời đã được xác minh độc lập nếu hệ thống chưa thực hiện xác minh đó."
                ),
            ]
        )
        return "\n\n".join(parts)

    def included_evidences(self, evidence_bundle: EvidenceBundle):
        """Return exactly the evidence sections that fit in the assembled prompt."""
        selected = []
        total_chars = 0
        # Prioritize approved official MCP evidence, then preserve relative order.
        ordered_evidences = sorted(
            evidence_bundle.evidences,
            key=lambda evidence: evidence.source_type != "mcp",
        )
        for evidence in ordered_evidences:
            section = self._format_evidence(evidence)
            remaining = self.max_evidence_chars - total_chars
            if len(section) > remaining:
                # Keep a truncated first relevant item rather than dropping all evidence.
                if not selected and remaining > 0:
                    header = (
                        f"[Evidence ID: {evidence.evidence_id}]\n"
                        f"Source: {evidence.source_type}\n"
                        f"Title: {evidence.title or ''}\n"
                        f"Score: {evidence.score}\n"
                        "Content:\n"
                    )
                    marker = "\n[Evidence content truncated to fit prompt budget.]"
                    content_budget = max(0, remaining - len(header) - len(marker))
                    if content_budget > 0:
                        selected.append(
                            replace(
                                evidence,
                                content=evidence.content[:content_budget] + marker,
                            )
                        )
                break
            selected.append(evidence)
            total_chars += len(section)
        return selected

    @staticmethod
    def _format_evidence(evidence) -> str:
        return (
            f"[Evidence ID: {evidence.evidence_id}]\n"
            f"Source: {evidence.source_type}\n"
            f"Title: {evidence.title or ''}\n"
            f"Score: {evidence.score}\n"
            f"Content:\n{evidence.content}\n"
        )

    def _format_evidences(self, evidences) -> str:
        if not evidences:
            return "Không có Evidence phù hợp. Không được tự tạo câu trả lời pháp lý khi thiếu nguồn."
        return "\n".join(self._format_evidence(item) for item in evidences)

    def _build_evidence_context(self, evidence_bundle: EvidenceBundle) -> str:
        """Compatibility wrapper retained for existing callers/tests."""
        return self._format_evidences(self.included_evidences(evidence_bundle))
