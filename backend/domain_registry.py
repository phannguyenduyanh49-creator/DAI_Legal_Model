import os
import sys
import logging
from typing import Dict, List, Literal, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator, ValidationError

# ==============================================================================
# CẤU HÌNH HỆ THỐNG LOGGING & ENCODING UTF-8 (HỖ TRỢ WINDOWS CONSOLE)
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
logger = logging.getLogger("DomainRegistryModule")


# ==============================================================================
# 1. PYDANTIC v2 SCHEMAS CHO CẤU HÌNH & CHÍNH SÁCH DOMAIN (TẦNG 0)
# ==============================================================================

class ChunkingStrategy(BaseModel):
    """Cấu hình chiến lược phân đoạn văn bản (Chunking) cho từng domain."""
    strategy_name: str = Field(..., description="Tên chiến lược chia đoạn (ví dụ: semantic_legal, recursive)", json_schema_extra={"example": "semantic_legal"})
    chunk_size: int = Field(..., gt=0, description="Kích thước tối đa của mỗi đoạn văn bản", json_schema_extra={"example": 512})
    chunk_overlap: int = Field(..., ge=0, description="Độ chồng lặp giữa các đoạn văn bản", json_schema_extra={"example": 64})

class PersonaConfig(BaseModel):
    """Cấu hình nhân vật và phong cách phản hồi của trợ lý AI."""
    tone: Literal["formal", "friendly", "strict", "professional"] = Field(..., description="Giọng điệu phản hồi chính", json_schema_extra={"example": "formal"})
    language_style: str = Field(..., description="Phong cách ngôn ngữ chi tiết", json_schema_extra={"example": "Chuẩn mực pháp lý, khách quan, chính xác"})
    system_prompt_addition: str = Field(..., description="Prompt bổ sung đặc thù cho domain", json_schema_extra={"example": "Tuyệt đối tuân thủ điều luật hiện hành, không suy đoán ngoài văn bản."})

class PolicyConfig(BaseModel):
    """Cấu hình chính sách an toàn, kiểm chứng và công cụ MCP được phép."""
    mode: Literal["STRICT", "FRIENDLY"] = Field(..., description="Chế độ hoạt động: STRICT (nghiêm ngặt) hoặc FRIENDLY (thân thiện)", json_schema_extra={"example": "STRICT"})
    verification_threshold: float = Field(..., ge=0.0, le=1.0, description="Ngưỡng kiểm chứng độ tin cậy tối thiểu", json_schema_extra={"example": 0.85})
    allowed_mcp_tools: List[str] = Field(default_factory=list, description="Danh sách Whitelist các MCP tools được phép gọi trong domain này", json_schema_extra={"example": ["tra_cuu_van_ban", "kiem_tra_hieu_luc"]})
    risk_alert_enabled: bool = Field(default=True, description="Kích hoạt cảnh báo rủi ro pháp lý/nghiệp vụ")

class DomainConfig(BaseModel):
    """Schema cấu hình tổng thể cho một domain độc lập."""
    domain_id: str = Field(..., description="Mã định danh duy nhất của domain", json_schema_extra={"example": "legal_ai"})
    domain_name: str = Field(..., description="Tên hiển thị đầy đủ của domain", json_schema_extra={"example": "Universal Legal AI Assistant"})
    description: str = Field(..., description="Mô tả phạm vi nghiệp vụ chuyên sâu")
    prompt_template: str = Field(..., description="Mẫu prompt chính dùng để sinh câu trả lời")
    chunking_strategy: ChunkingStrategy
    persona: PersonaConfig
    policy: PolicyConfig

    @field_validator("domain_id")
    @classmethod
    def validate_domain_id(cls, v: str) -> str:
        """Kiểm tra tính hợp lệ của domain_id (không được để trống hoặc chứa khoảng trắng)."""
        v = v.strip().lower()
        if not v or " " in v:
            raise ValueError("domain_id không được để trống và không được chứa khoảng trắng.")
        return v


# ==============================================================================
# 2. DOMAIN REGISTRY SERVICE (QUẢN LÝ, TRUY XUẤT VÀ CẬP NHẬT CÔ LẬP)
# ==============================================================================

class DomainRegistryService:
    """
    Kho lưu trữ và quản lý cấu hình domain độc lập (Tầng 0).
    Cung cấp các phương thức nạp, truy xuất, cập nhật chính sách đảm bảo tính cô lập tuyệt đối.
    """
    def __init__(self):
        logger.info("[DomainRegistryService] Đang khởi tạo kho lưu trữ cấu hình Domain & Policy...")
        self._registry: Dict[str, DomainConfig] = {}
        self._load_initial_domain_configs()

    def _load_initial_domain_configs(self) -> None:
        """Nạp tối thiểu các cấu hình mẫu chuẩn doanh nghiệp để kiểm thử."""

        # 1. Cấu hình mẫu: Lĩnh vực Pháp lý (Strict Mode)
        legal_config = DomainConfig(
            domain_id="legal_ai",
            domain_name="Universal Legal AI Assistant",
            description="Chuyên gia tư vấn pháp luật Việt Nam, tuân thủ chặt chẽ văn bản quy phạm pháp luật.",
            prompt_template="Bạn là Luật sư chuyên nghiệp. Hãy trả lời câu hỏi dựa TRÊN HẾT các điều luật được cung cấp sau:\n{context}\nCâu hỏi: {query}\nYêu cầu bắt buộc: Trích dẫn rõ điều luật.",
            chunking_strategy=ChunkingStrategy(strategy_name="semantic_legal", chunk_size=512, chunk_overlap=64),
            persona=PersonaConfig(tone="formal", language_style="Chuẩn mực pháp lý, khách quan", system_prompt_addition="Tuyệt đối không suy đoán ngoài điều luật."),
            policy=PolicyConfig(mode="STRICT", verification_threshold=0.85, allowed_mcp_tools=["tra_cuu_van_ban", "kiem_tra_hieu_luc"], risk_alert_enabled=True)
        )

        # 2. Cấu hình mẫu: Lĩnh vực Tổng hợp / Dịch vụ công (Friendly Mode)
        general_config = DomainConfig(
            domain_id="general_assistant",
            domain_name="General & Customer Service Assistant",
            description="Trợ lý tổng hợp hỗ trợ tra cứu thông tin đời sống, du lịch và dịch vụ công cơ bản.",
            prompt_template="Bạn là trợ lý thân thiện. Hãy hỗ trợ người dùng với thông tin: {context}\nCâu hỏi: {query}",
            chunking_strategy=ChunkingStrategy(strategy_name="recursive", chunk_size=1024, chunk_overlap=128),
            persona=PersonaConfig(tone="friendly", language_style="Cởi mở, dễ hiểu, tận tình", system_prompt_addition="Luôn giữ thái độ lịch sự, hỗ trợ hết mình."),
            policy=PolicyConfig(mode="FRIENDLY", verification_threshold=0.50, allowed_mcp_tools=["tra_cuu_thoi_tiet", "ban_do_cong_khai"], risk_alert_enabled=False)
        )

        # 3. Cấu hình mẫu bổ sung: Lĩnh vực Hành chính công (Administrative Law)
        admin_config = DomainConfig(
            domain_id="administrative_law",
            domain_name="Public Administration & Procedures Assistant",
            description="Trợ lý chuyên hỗ trợ tra cứu thủ tục hành chính, giấy phép và hồ sơ cơ quan nhà nước.",
            prompt_template="Bạn là chuyên gia thủ tục hành chính công. Hướng dẫn quy trình dựa trên dữ liệu:\n{context}\nNội dung cần xử lý: {query}",
            chunking_strategy=ChunkingStrategy(strategy_name="paragraph", chunk_size=768, chunk_overlap=96),
            persona=PersonaConfig(tone="professional", language_style="Rõ ràng, mạch lạc, dễ thực hiện", system_prompt_addition="Liệt kê rõ các bước hồ sơ và lệ phí nếu có."),
            policy=PolicyConfig(mode="STRICT", verification_threshold=0.80, allowed_mcp_tools=["tra_cuu_thu_tuc_hanh_chinh", "kiem_tra_ho_so"], risk_alert_enabled=True)
        )

        self.register_domain(legal_config)
        self.register_domain(general_config)
        self.register_domain(admin_config)

    def register_domain(self, config: DomainConfig) -> None:
        """Đăng ký hoặc thêm mới một cấu hình domain vào kho."""
        self._registry[config.domain_id] = config
        logger.info(f"[DomainRegistry] Đã đăng ký thành công domain: '{config.domain_id}'")

    def get_domain_config(self, domain_id: str) -> DomainConfig:
        """Truy xuất cấu hình theo domain_id. Báo lỗi KeyError rõ ràng nếu không tồn tại."""
        normalized_id = domain_id.strip().lower()
        if normalized_id not in self._registry:
            raise KeyError(f"Lỗi: Domain '{domain_id}' không tồn tại trong hệ thống Registry.")
        return self._registry[normalized_id]

    def list_all_domains(self) -> List[str]:
        """Liệt kê toàn bộ các mã domain đang được quản lý."""
        return list(self._registry.keys())

    def update_domain_policy(self, domain_id: str, new_mode: Optional[str] = None, new_tools: Optional[List[str]] = None) -> DomainConfig:
        """
        Cập nhật chính sách (Strict/Friendly mode và Whitelist tools) cho một domain cụ thể.
        Đảm bảo tính cô lập tuyệt đối: Sử dụng model_copy để không làm ảnh hưởng domain khác.
        """
        config = self.get_domain_config(domain_id)

        # Tạo bản sao policy độc lập
        updated_policy = config.policy.model_copy()

        if new_mode is not None:
            normalized_mode = new_mode.upper()
            if normalized_mode not in ["STRICT", "FRIENDLY"]:
                raise ValueError(f"Chế độ chính sách không hợp lệ: '{new_mode}'. Chỉ chấp nhận 'STRICT' hoặc 'FRIENDLY'.")
            updated_policy.mode = normalized_mode # type: ignore

        if new_tools is not None:
            updated_policy.allowed_mcp_tools = list(new_tools)

        # Cập nhật cấu hình mới vào registry
        updated_config = config.model_copy(update={"policy": updated_policy})
        self._registry[domain_id.lower()] = updated_config

        logger.info(f"[DomainRegistry] Đã cập nhật policy cho domain '{domain_id}' thành công.")
        return updated_config


# Khởi tạo instance dùng chung cho toàn bộ ứng dụng backend
registry_service = DomainRegistryService()


# ==============================================================================
# 3. FASTAPI ROUTER (API TRUY XUẤT VÀ CẬP NHẬT CẤU HÌNH TẦNG 0)
# ==============================================================================

domain_router = APIRouter(prefix="/api/v2/domains", tags=["Domain Config & Policy Registry"])

@domain_router.get("/", response_model=List[str])
async def api_list_all_domains():
    """API lấy danh sách toàn bộ các domain đang hoạt động."""
    return registry_service.list_all_domains()

@domain_router.get("/{domain_id}", response_model=DomainConfig)
async def api_get_domain_config(domain_id: str):
    """API truy xuất toàn bộ cấu hình chi tiết (prompt, policy, persona, tools) theo domain_id."""
    try:
        return registry_service.get_domain_config(domain_id)
    except KeyError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))


# ==============================================================================
# 4. KHỐI KIỂM CHỨNG & CHẠY THỬ NGHIỆM TRỰC TIẾP (UNIT TEST / SCRIPT EXECUTION)
# ==============================================================================

if __name__ == "__main__":
    print("\n================ [KIỂM CHỨNG CHI TIẾT TẦNG 0: DOMAIN CONFIG & POLICY] ================")

    # 1. Kiểm tra danh sách các domain đã nạp
    domains = registry_service.list_all_domains()
    print(f"\n1. Danh sách các Domain đang quản lý ({len(domains)} domain): {domains}")

    # 2. Tải và in cấu hình khác nhau giữa các domain
    print(f"\n2. Kiểm tra sự khác biệt cấu hình giữa các Domain:")
    for d_id in domains:
        cfg = registry_service.get_domain_config(d_id)
        print(f"   - Domain ID: [{cfg.domain_id}]")
        print(f"     + Tên: {cfg.domain_name}")
        print(f"     + Chế độ Policy Mode: {cfg.policy.mode} (Threshold: {cfg.policy.verification_threshold})")
        print(f"     + Chunking Strategy: {cfg.chunking_strategy.strategy_name} (Size: {cfg.chunking_strategy.chunk_size})")
        print(f"     + Whitelist Tools: {cfg.policy.allowed_mcp_tools}")

    # 3. Kiểm thử cập nhật chính sách và tính cô lập
    print(f"\n3. Kiểm thử cập nhật chính sách & Tính cô lập:")
    tools_before_general = list(registry_service.get_domain_config("general_assistant").policy.allowed_mcp_tools)

    print(f"   -> Cập nhật domain 'legal_ai' sang chế độ FRIENDLY và thay đổi whitelist tools...")
    registry_service.update_domain_policy(
        domain_id="legal_ai",
        new_mode="FRIENDLY",
        new_tools=["tool_dac_biet_1", "tool_dac_biet_2"]
    )

    legal_updated = registry_service.get_domain_config("legal_ai")
    general_unchanged = registry_service.get_domain_config("general_assistant")

    print(f"   -> Kết quả sau cập nhật 'legal_ai': Mode = {legal_updated.policy.mode}, Tools = {legal_updated.policy.allowed_mcp_tools}")
    print(f"   -> Kiểm chứng tính cô lập ('general_assistant'): Mode = {general_unchanged.policy.mode}, Tools = {general_unchanged.policy.allowed_mcp_tools}")
    assert general_unchanged.policy.allowed_mcp_tools == tools_before_general, "LỖI: Tính cô lập bị phá vỡ!"
    print(f"   -> [THÀNH CÔNG] Đã chứng minh tính cô lập tuyệt đối giữa các domain!")

    # 4. Kiểm tra xử lý lỗi khi domain không tồn tại
    print(f"\n4. Kiểm thử xử lý lỗi khi truy xuất domain không tồn tại:")
    try:
        registry_service.get_domain_config("domain_ao_khong_ton_tai")
    except KeyError as e:
        print(f"   -> [THÀNH CÔNG] Bắt lỗi ngoại lệ thành công, không làm crash hệ thống: {e}")

    # 5. Kiểm tra bắt lỗi Pydantic Validation (thiếu trường bắt buộc)
    print(f"\n5. Kiểm thử cơ chế Validation Schema (Báo lỗi khi thiếu dữ liệu):")
    try:
        DomainConfig(
            domain_id="bad_domain",
            domain_name="Bad Config",
            description="Thiếu cấu hình chunking",
            prompt_template="...",
            # chunking_strategy bị thiếu cố ý để test lỗi
            persona=PersonaConfig(tone="formal", language_style="...", system_prompt_addition="..."),
            policy=PolicyConfig(mode="STRICT", verification_threshold=0.9, allowed_mcp_tools=[])
        )
    except ValidationError as ve:
        print(f"   -> [THÀNH CÔNG] Pydantic bắt lỗi thiếu trường bắt buộc chuẩn xác:\n{ve.errors()[0]}")

    print("\n================ HOÀN TẤT KIỂM CHỨNG ================\n")