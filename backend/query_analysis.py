import re
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("DAI_Legal_Query_Analysis")

class AnalysisResult(BaseModel):
    original_query: str
    cleaned_query: str
    extracted_citations: List[str] = Field(default_factory=list, description="Các điều luật hoặc bộ luật được trích xuất trực tiếp từ câu hỏi")
    expanded_terms: List[str] = Field(default_factory=list, description="Từ khóa và từ đồng nghĩa chuyên ngành pháp lý")
    hypothetical_document: Optional[str] = Field(default=None, description="Văn bản pháp lý giả định sinh ra theo cơ chế HyDE")
    confidence_score: float = Field(..., description="Độ tin cậy của quá trình bóc tách và phân tích ý định")

class QueryAnalysisService:
    """
    Dịch vụ phân tích câu truy vấn chuyên sâu cho hệ thống trợ lý pháp lý AI.
    Thay thế mô hình Synonym Mapping cũ bằng pipeline làm sạch, bóc tách trích dẫn,
    mở rộng từ khóa ngữ nghĩa và sinh HyDE (Hypothetical Document Embeddings).
    """
    def __init__(self):
        logger.info("[Module 3 - QueryAnalysis] Đang khởi tạo bộ xử lý ngôn ngữ pháp lý và từ điển chuyên ngành...")

        # Từ điển mở rộng thuật ngữ pháp lý Việt Nam chuyên sâu
        self.legal_thesaurus: Dict[str, List[str]] = {
            "doanh nghiệp": ["công ty TNHH", "công ty cổ phần", "doanh nghiệp tư nhân", "hợp tác xã", "thương nhân"],
            "thủ tục": ["trình tự thực hiện", "hồ sơ pháp lý", "thẩm quyền giải quyết", "điều kiện cấp phép", "quy trình hành chính"],
            "hợp đồng": ["giao dịch dân sự", "thỏa thuận cam kết", "điều khoản hợp đồng", "quyền và nghĩa vụ các bên"],
            "ly hôn": ["chấm dứt hôn nhân", "giải quyết tranh chấp tài sản chung", "giành quyền nuôi con", "ly hôn thuận tình", "ly hôn đơn phương"],
            "bồi thường": ["trách nhiệm bồi thường thiệt hại ngoài hợp đồng", "khắc phục hậu quả", "mức đền bù tài chính", "tổn thất vật chất và tinh thần"],
            "sa thải": ["chấm dứt hợp đồng lao động", "kỷ luật lao động", "buộc thôi việc", "quyền lợi trợ cấp thôi việc"],
            "đất đai": ["quyền sử dụng đất", "giấy chứng nhận", "chuyển nhượng nhà đất", "thu hồi đất", "bồi thường giải phóng mặt bằng"]
        }

    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản đầu vào, loại bỏ ký tự rác, định dạng thừa và chuẩn hóa khoảng trắng."""
        if not text:
            return ""
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_legal_citations(self, query: str) -> List[str]:
        """Bóc tách các căn cứ pháp luật có sẵn trong câu hỏi (Ví dụ: 'Điều 12 Luật Doanh nghiệp')."""
        citations = []
        article_matches = re.findall(r'(điều\s+\d+(?:\s+khoản\s+\d+)?(?:\s+điểm\s+[a-zđ])?)', query, re.IGNORECASE)
        law_matches = re.findall(r'((?:luật|bộ luật|nghị định|thông tư)\s+[\w\s,]+)', query, re.IGNORECASE)

        if article_matches:
            citations.extend([m.strip() for m in article_matches])
        if law_matches:
            citations.extend([m.strip() for m in law_matches])

        return list(dict.fromkeys(citations))

    def expand_legal_terms(self, cleaned_query: str) -> List[str]:
        """Phân tích câu hỏi, bóc tách từ khóa và tra cứu từ điển đồng nghĩa chuyên sâu."""
        words = cleaned_query.lower().split()
        expanded = []

        for word in words:
            clean_w = re.sub(r'[^\w\s]', '', word).strip()
            if not clean_w or len(clean_w) < 2:
                continue

            expanded.append(clean_w)
            for key, synonyms in self.legal_thesaurus.items():
                if key in clean_w or clean_w in key:
                    expanded.extend(synonyms)

        return list(dict.fromkeys(expanded))

    def generate_hypothetical_document(self, query: str, citations: List[str]) -> str:
        """Thực hiện kỹ thuật HyDE: Sinh văn bản phản hồi giả định."""
        logger.info(f"[HyDE Engine] Đang tổng hợp tài liệu giả định cho truy vấn: '{query[:50]}...'")

        citation_context = f" Căn cứ vào các quy định tại {', '.join(citations)}." if citations else ""

        hypothetical_doc = (
            f"Văn bản hướng dẫn và giải quyết vấn đề pháp lý liên quan đến '{query}':{citation_context} "
            f"Theo quy định của pháp luật hiện hành, các bên tham gia phải tuân thủ nghiêm ngặt các điều kiện, "
            f"trình tự thủ tục hồ sơ và thẩm quyền giải quyết của cơ quan nhà nước có thẩm quyền. "
            f"Quyền và nghĩa vụ hợp pháp của các chủ thể được bảo vệ dựa trên các nguyên tắc pháp lý chuẩn mực, "
            f"đảm bảo giải quyết tranh chấp hoặc thực thi quyền lợi theo đúng khung pháp luật quy định."
        )
        return hypothetical_doc

    def analyze(self, query: str, enable_hyde: bool = True) -> AnalysisResult:
        """Hàm thực thi tổng thể chuỗi quy trình phân tích và làm sạch câu truy vấn."""
        cleaned_q = self.clean_text(query)
        extracted_cites = self.extract_legal_citations(cleaned_q)
        expanded_terms = self.expand_legal_terms(cleaned_q)

        hypo_doc = None
        if enable_hyde:
            hypo_doc = self.generate_hypothetical_document(cleaned_q, extracted_cites)

        base_confidence = 0.60
        bonus = min(0.35, (len(expanded_terms) * 0.03) + (len(extracted_cites) * 0.10))
        confidence_score = round(min(0.98, base_confidence + bonus), 2)

        return AnalysisResult(
            original_query=query,
            cleaned_query=cleaned_q,
            extracted_citations=extracted_cites,
            expanded_terms=expanded_terms,
            hypothetical_document=hypo_doc,
            confidence_score=confidence_score
        )