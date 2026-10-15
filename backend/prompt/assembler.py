from typing import Optional

from backend.evidence.schemas import EvidenceBundle


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
    ) -> str:

        system_part = system_instruction or (
            "Bạn là trợ lý AI tư vấn thủ tục hành chính. "
            "Chỉ sử dụng thông tin trong Evidence để trả lời. "
            "Không tự suy đoán thông tin không có trong Evidence."
        )

        evidence_context = self._build_evidence_context(
            evidence_bundle
        )


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

        return prompt

    def _build_evidence_context(
        self,
        evidence_bundle: EvidenceBundle,
    ) -> str:

        if not evidence_bundle.evidences:
            return "Không có Evidence phù hợp."

        sections = []
        total_chars = 0

        for evidence in evidence_bundle.evidences:

            section = (
                f"[Evidence ID: {evidence.evidence_id}]\n"
                f"Source: {evidence.source_type}\n"
                f"Title: {evidence.title or ''}\n"
                f"Score: {evidence.score}\n"
                f"Content:\n{evidence.content}\n"
            )

            if (
                total_chars + len(section)
                > self.max_evidence_chars
            ):
                break

            sections.append(section)
            total_chars += len(section)

        return "\n".join(sections)