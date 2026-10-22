"""Resolve domain policy and enforce its MCP allowlist at runtime."""

from dataclasses import dataclass
from typing import Iterable

from backend.domain_registry import DomainConfig, registry_service


REGISTERED_MCP_TOOLS = frozenset({
    "search_legal_documents",
    "search_procedure",
    "search_dichvucong",
})


@dataclass(frozen=True)
class EnforcedPolicy:
    domain_config: DomainConfig
    allowed_mcp_tools: tuple[str, ...]


class PolicyEnforcementMiddleware:
    """Map routed domains to registry policies and narrow permissions safely."""

    def __init__(self, registry=registry_service):
        self.registry = registry

    def registry_domain_for(self, routed_domain: str) -> str:
        normalized = str(routed_domain or "").strip().lower()
        if normalized in self.registry.list_all_domains():
            return normalized
        if "admin" in normalized:
            return "administrative_law"
        if normalized in {
            "civil_law", "enterprise_law", "criminal_law", "labor_law",
            "land_law", "general_legal",
        }:
            return "legal_ai"
        return "legal_ai"

    @staticmethod
    def effective_tools(allowed_tools: Iterable[str]) -> tuple[str, ...]:
        """Keep only exact registered tool names, independent of policy mode."""
        return tuple(sorted(set(allowed_tools) & REGISTERED_MCP_TOOLS))

    def apply(self, routed_domain: str) -> EnforcedPolicy:
        domain_id = self.registry_domain_for(routed_domain)
        config = self.registry.get_domain_config(domain_id)
        return EnforcedPolicy(
            domain_config=config,
            allowed_mcp_tools=self.effective_tools(
                config.policy.allowed_mcp_tools
            ),
        )

    def constrain_tools(
        self,
        domain_id: str,
        requested_tools: Iterable[str],
    ) -> tuple[str, ...]:
        """Recheck forwarded permissions against registry at the MCP boundary."""
        config = self.registry.get_domain_config(domain_id)
        registry_allowlist = self.effective_tools(config.policy.allowed_mcp_tools)
        return tuple(sorted(set(requested_tools) & set(registry_allowlist)))
