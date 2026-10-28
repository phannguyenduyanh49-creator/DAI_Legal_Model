from typing import Optional
from dataclasses import replace

from backend.evidence.schemas import EvidenceBundle
from backend.answer.citation_contract import CITATION_RULES, format_allowed_evidence_ids


class PromptAssembler:

    def __init__(
        self,
        max_evidence_chars: int = 12000,
    ):
        self.max_evidence_chars = max_evidence_chars

    def build(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        system_instruction: Optional[str] = None,
        domain_prompt_template: Optional[str] = None,
    ) -> str:

        system_part = system_instruction or (
            "Bạn là trợ lý AI tư vấn thủ tục hành chính. "
            "Chỉ sử dụng thông tin trong Evidence để trả lời. "
            "Không tự suy đoán thông tin không có trong Evidence."
        )

        included_evidences = self.included_evidences(evidence_bundle)
        system_part += " " + CITATION_RULES
        evidence_context = self._build_evidence_context(
            evidence_bundle
        )

        domain_prompt = ""
        if domain_prompt_template:
            try:
                domain_prompt = domain_prompt_template.format(
                    context="the shared EVIDENCE section above",
                    query=question,
                )
            except (KeyError, IndexError, ValueError):
                # A malformed registry template must not break the answer pipeline.
                domain_prompt = ""

        prompt = (
            f"{system_part}\n\n"
            "=== USER QUESTION ===\n"
            f"{question}\n\n"
            "=== EVIDENCE ===\n"
            f"{evidence_context}\n\n"
            "=== INSTRUCTIONS ===\n"
            "Chỉ sử dụng thông tin được cung cấp trong Evidence để trả lời.\n"
            "Ưu tiên thông tin thuộc đúng thủ tục hành chính mà người dùng hỏi.\n"
            "Trích xuất chính xác thành phần hồ sơ, thời gian giải quyết, "
            "phí và lệ phí nếu Evidence có cung cấp.\n"
            "Không tự thay đổi số liệu, đơn vị, thời hạn hoặc điều kiện "
            "được ghi trong Evidence.\n"
            "Phân biệt rõ lệ phí bằng 0, không thu phí và thông tin "
            "về lệ phí chưa được cung cấp.\n"
            "Không tự bổ sung giấy tờ, thời hạn hoặc mức phí từ kiến thức bên ngoài.\n"
            "Nếu Evidence không có thông tin cần thiết, hãy nói rõ "
            "'Evidence hiện có chưa cung cấp thông tin này'.\n"
            "Nếu có nhiều Evidence, ưu tiên nguồn liên quan trực tiếp "
            "đến thủ tục đang được hỏi.\n"
            "Trình bày câu trả lời rõ ràng theo các mục phù hợp với câu hỏi.\n"
        )

        allowed_ids = format_allowed_evidence_ids(
            item.evidence_id for item in included_evidences
        )
        return (
            prompt
            + ("\n=== DOMAIN-SPECIFIC INSTRUCTIONS ===\n" + domain_prompt + "\n" if domain_prompt else "")
            + "\n=== REQUIRED CITATION FORMAT ===\n"
            + CITATION_RULES
            + "\n"
            + f"Allowed Evidence IDs: {allowed_ids}\n"
        )

    def included_evidences(self, evidence_bundle: EvidenceBundle):
        """Return exactly the evidence sections that fit in the assembled prompt."""
        selected = []
        total_chars = 0
        # When an approved external lookup supplied official DVC evidence,
        # give it prompt-budget priority over similarly ranked local snippets.
        ordered_evidences = sorted(
            evidence_bundle.evidences,
            key=lambda evidence: evidence.source_type != "mcp",
        )
        for evidence in ordered_evidences:
            section = (
                f"[Evidence ID: {evidence.evidence_id}]\n"
                f"Source: {evidence.source_type}\n"
                f"Title: {evidence.title or ''}\n"
                f"Score: {evidence.score}\n"
                f"Content:\n{evidence.content}\n"
            )
            remaining = self.max_evidence_chars - total_chars
            if len(section) > remaining:
                # The first relevant Evidence item can exceed the entire
                # budget. Keep its citation identity and include a marked
                # prefix instead of silently dropping all usable evidence.
                if not selected:
                    header = (
                        f"[Evidence ID: {evidence.evidence_id}]\n"
                        f"Source: {evidence.source_type}\n"
                        f"Title: {evidence.title or ''}\n"
                        f"Score: {evidence.score}\n"
                        "Content:\n"
                    )
                    truncation_marker = "\n[Evidence content truncated to fit prompt budget.]"
                    content_budget = max(0, remaining - len(header) - len(truncation_marker))
                    if content_budget:
                        selected.append(
                            replace(
                                evidence,
                                content=evidence.content[:content_budget]
                                + truncation_marker,
                            )
                        )
                break
            selected.append(evidence)
            total_chars += len(section)
        return selected

    def _build_evidence_context(
        self,
        evidence_bundle: EvidenceBundle,
    ) -> str:

        if not evidence_bundle.evidences:
            return "Không có Evidence phù hợp."

        sections = []
        for evidence in self.included_evidences(evidence_bundle):
            section = (
                f"[Evidence ID: {evidence.evidence_id}]\n"
                f"Source: {evidence.source_type}\n"
                f"Title: {evidence.title or ''}\n"
                f"Score: {evidence.score}\n"
                f"Content:\n{evidence.content}\n"
            )

            sections.append(section)

        return "\n".join(sections)
