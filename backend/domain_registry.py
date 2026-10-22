
import logging
import sys
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator, ValidationError


# ==============================================================================
# LOGGING & UTF-8
# ==============================================================================

def configure_utf8_stdio() -> None:
    """Đảm bảo console Windows hiển thị UTF-8."""
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
# 1. DOMAIN CONFIGURATION SCHEMAS
# ==============================================================================

class ChunkingStrategy(BaseModel):
    """Cấu hình chia đoạn dữ liệu."""
    strategy_name: str = Field(..., description="Tên chiến lược chia đoạn")
    chunk_size: int = Field(..., gt=0, description="Kích thước đoạn")
    chunk_overlap: int = Field(..., ge=0, description="Độ chồng lặp")


class PersonaConfig(BaseModel):
    """Cấu hình phong cách phản hồi."""
    tone: Literal["formal", "friendly", "strict", "professional"]
    language_style: str
    system_prompt_addition: str


class PolicyConfig(BaseModel):
    """Cấu hình chính sách và công cụ được phép."""
    mode: Literal["STRICT", "FRIENDLY"]
    verification_threshold: float = Field(..., ge=0.0, le=1.0)
    allowed_mcp_tools: List[str] = Field(default_factory=list)
    risk_alert_enabled: bool = True


class DomainConfig(BaseModel):
    """Cấu hình hoàn chỉnh của một domain."""
    domain_id: str
    domain_name: str
    description: str
    prompt_template: str
    chunking_strategy: ChunkingStrategy
    persona: PersonaConfig
    policy: PolicyConfig

    @field_validator("domain_id")
    @classmethod
    def validate_domain_id(cls, value: str) -> str:
        """Chuẩn hóa và kiểm tra ID domain."""
        value = value.strip().lower()
        if not value or any(char.isspace() for char in value):
            raise ValueError(
                "domain_id không được rỗng hoặc chứa khoảng trắng."
            )
        return value


# ==============================================================================
# 2. DOMAIN REGISTRY SERVICE
# ==============================================================================

class DomainRegistryService:
    """Quản lý cấu hình và chính sách riêng cho từng domain."""

    def __init__(self):
        logger.info(
            "[DomainRegistryService] Khởi tạo Domain Registry..."
        )
        self._registry: Dict[str, DomainConfig] = {}
        self._load_initial_domain_configs()

    def _load_initial_domain_configs(self) -> None:
        """Nạp các domain cấu hình sẵn."""

        # 1. Pháp lý tổng quát
        legal_config = DomainConfig(
            domain_id="legal_ai",
            domain_name="Universal Legal AI Assistant",
            description=(
                "Trợ lý tư vấn pháp luật Việt Nam, "
                "ưu tiên văn bản quy phạm pháp luật."
            ),
            prompt_template=(
                "Bạn là trợ lý pháp lý. Trả lời dựa trên "
                "các điều luật được cung cấp:\n"
                "{context}\nCâu hỏi: {query}\n"
                "Nêu rõ căn cứ pháp lý khi có dữ liệu."
            ),
            chunking_strategy=ChunkingStrategy(
                strategy_name="semantic_legal",
                chunk_size=512,
                chunk_overlap=64,
            ),
            persona=PersonaConfig(
                tone="formal",
                language_style="Chuẩn mực, khách quan, chính xác",
                system_prompt_addition=(
                    "Không suy đoán ngoài dữ liệu pháp lý được cung cấp."
                ),
            ),
            policy=PolicyConfig(
                mode="STRICT",
                verification_threshold=0.85,
                allowed_mcp_tools=[
                    "tra_cuu_van_ban",
                    "kiem_tra_hieu_luc",
                ],
                risk_alert_enabled=True,
            ),
        )

        # 2. Trợ lý tổng hợp
        general_config = DomainConfig(
            domain_id="general_assistant",
            domain_name="General & Customer Service Assistant",
            description=(
                "Trợ lý tổng hợp hỗ trợ thông tin đời sống, "
                "du lịch và dịch vụ công cơ bản."
            ),
            prompt_template=(
                "Bạn là trợ lý thân thiện. "
                "Hãy trả lời dựa trên thông tin sau:\n"
                "{context}\nCâu hỏi: {query}"
            ),
            chunking_strategy=ChunkingStrategy(
                strategy_name="recursive",
                chunk_size=1024,
                chunk_overlap=128,
            ),
            persona=PersonaConfig(
                tone="friendly",
                language_style="Cởi mở, dễ hiểu, tận tình",
                system_prompt_addition=(
                    "Giữ thái độ lịch sự và hỗ trợ người dùng."
                ),
            ),
            policy=PolicyConfig(
                mode="FRIENDLY",
                verification_threshold=0.50,
                allowed_mcp_tools=[
                    "tra_cuu_thoi_tiet",
                    "ban_do_cong_khai",
                ],
                risk_alert_enabled=False,
            ),
        )

        # 3. Thủ tục hành chính
        admin_config = DomainConfig(
            domain_id="administrative_law",
            domain_name="Public Administration & Procedures Assistant",
            description=(
                "Trợ lý tra cứu thủ tục hành chính, giấy phép "
                "và hồ sơ cơ quan nhà nước."
            ),
            prompt_template=(
                "Bạn là chuyên gia thủ tục hành chính. "
                "Hướng dẫn dựa trên dữ liệu:\n"
                "{context}\nNội dung cần xử lý: {query}"
            ),
            chunking_strategy=ChunkingStrategy(
                strategy_name="paragraph",
                chunk_size=768,
                chunk_overlap=96,
            ),
            persona=PersonaConfig(
                tone="professional",
                language_style="Rõ ràng, mạch lạc, dễ thực hiện",
                system_prompt_addition=(
                    "Nêu các bước hồ sơ và lệ phí nếu có dữ liệu."
                ),
            ),
            policy=PolicyConfig(
                mode="STRICT",
                verification_threshold=0.80,
                allowed_mcp_tools=[
                    "tra_cuu_thu_tuc_hanh_chinh",
                    "kiem_tra_ho_so",
                ],
                risk_alert_enabled=True,
            ),
        )

        # 4. Du lịch
        tourism_config = DomainConfig(
            domain_id="tourism",
            domain_name="Tourism Assistant",
            description=(
                "Trợ lý du lịch tra cứu điểm đến, khách sạn, "
                "giá vé, địa điểm tham quan và dịch vụ du lịch."
            ),
            prompt_template=(
                "Bạn là trợ lý du lịch. Chỉ trả lời dựa trên "
                "dữ liệu được cung cấp:\n"
                "{context}\nCâu hỏi: {query}\n"
                "Nếu dữ liệu không đủ, hãy nói rõ thông tin còn thiếu."
            ),
            chunking_strategy=ChunkingStrategy(
                strategy_name="recursive",
                chunk_size=1024,
                chunk_overlap=128,
            ),
            persona=PersonaConfig(
                tone="friendly",
                language_style="Thân thiện, rõ ràng, dễ hiểu",
                system_prompt_addition=(
                    "Không tự bịa giá vé, địa chỉ hoặc thông tin du lịch."
                ),
            ),
            policy=PolicyConfig(
                mode="FRIENDLY",
                verification_threshold=0.50,
                allowed_mcp_tools=[],
                risk_alert_enabled=False,
            ),
        )

        self.register_domain(legal_config)
        self.register_domain(general_config)
        self.register_domain(admin_config)
        self.register_domain(tourism_config)

    def register_domain(self, config: DomainConfig) -> None:
        """Đăng ký hoặc cập nhật cấu hình domain."""
        normalized_id = config.domain_id.strip().lower()

        if not normalized_id or any(
            char.isspace() for char in normalized_id
        ):
            raise ValueError("domain_id không hợp lệ.")

        normalized_config = config.model_copy(
            update={"domain_id": normalized_id}
        )

        self._registry[normalized_id] = normalized_config

        logger.info(
            "[DomainRegistry] Đăng ký domain thành công: '%s'",
            normalized_id,
        )

    def get_domain_config(self, domain_id: str) -> DomainConfig:
        """Lấy cấu hình domain theo ID."""
        normalized_id = domain_id.strip().lower()

        if normalized_id not in self._registry:
            raise KeyError(
                f"Domain '{domain_id}' không tồn tại trong Registry."
            )

        return self._registry[normalized_id]

    def list_all_domains(self) -> List[str]:
        """Trả về danh sách ID domain."""
        return list(self._registry.keys())

    def update_domain_policy(
        self,
        domain_id: str,
        new_mode: Optional[str] = None,
        new_tools: Optional[List[str]] = None,
    ) -> DomainConfig:
        """Cập nhật policy mà không sửa policy của domain khác."""
        config = self.get_domain_config(domain_id)
        updated_policy = config.policy.model_copy(deep=True)

        if new_mode is not None:
            normalized_mode = new_mode.upper()

            if normalized_mode not in ("STRICT", "FRIENDLY"):
                raise ValueError(
                    f"Policy mode không hợp lệ: '{new_mode}'."
                )

            updated_policy.mode = normalized_mode

        if new_tools is not None:
            updated_policy.allowed_mcp_tools = list(new_tools)

        updated_config = config.model_copy(
            update={"policy": updated_policy}
        )

        normalized_id = domain_id.strip().lower()
        self._registry[normalized_id] = updated_config

        logger.info(
            "[DomainRegistry] Cập nhật policy domain '%s' thành công.",
            normalized_id,
        )

        return updated_config


# Registry dùng chung trong backend
registry_service = DomainRegistryService()


# ==============================================================================
# 3. FASTAPI ROUTER
# ==============================================================================

domain_router = APIRouter(
    prefix="/api/v2/domains",
    tags=["Domain Config & Policy Registry"],
)


@domain_router.get("/", response_model=List[str])
async def api_list_all_domains():
    """API lấy danh sách domain."""
    return registry_service.list_all_domains()


@domain_router.get("/{domain_id}", response_model=DomainConfig)
async def api_get_domain_config(domain_id: str):
    """API lấy cấu hình domain."""
    try:
        return registry_service.get_domain_config(domain_id)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )


# ==============================================================================
# 4. LOCAL VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    print("Registered domains:", registry_service.list_all_domains())

    for domain_id in registry_service.list_all_domains():
        config = registry_service.get_domain_config(domain_id)
        print(
            domain_id,
            "| mode:", config.policy.mode,
            "| threshold:", config.policy.verification_threshold,
            "| chunking:", config.chunking_strategy.strategy_name,
        )

    # Kiểm tra tính cô lập policy.
    general_tools_before = list(
        registry_service.get_domain_config(
            "general_assistant"
        ).policy.allowed_mcp_tools
    )

    registry_service.update_domain_policy(
        domain_id="legal_ai",
        new_mode="FRIENDLY",
        new_tools=["temporary_test_tool"],
    )

    assert (
        registry_service.get_domain_config(
            "general_assistant"
        ).policy.allowed_mcp_tools == general_tools_before
    )

    # Kiểm tra lỗi domain không tồn tại.
    try:
        registry_service.get_domain_config("unknown_domain")
    except KeyError:
        print("Unknown-domain validation: PASS")

    print("Registry verification completed.")
