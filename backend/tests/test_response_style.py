from backend.response_style import resolve_response_style


def test_default_response_style_uses_friendly_only_for_tourism():
    assert resolve_response_style("default", "tourism") == "friendly"
    for domain_id in (
        "legal_ai",
        "administrative_law",
        "healthcare",
        "education",
        "shopping",
        "food",
        "general_assistant",
        None,
        "unknown-domain",
    ):
        assert resolve_response_style("default", domain_id) == "strict"


def test_explicit_response_style_overrides_domain_default():
    assert resolve_response_style("strict", "tourism") == "strict"
    assert resolve_response_style("friendly", "healthcare") == "friendly"


def test_unknown_style_fails_safe_to_domain_default():
    assert resolve_response_style("invalid", "tourism") == "friendly"
    assert resolve_response_style("invalid", "healthcare") == "strict"


def test_gateway_request_schema_accepts_default_style():
    from backend.core_gateway import QueryRequest

    assert QueryRequest(query="question").response_style == "default"
    assert QueryRequest(query="question", response_style="friendly").response_style == "friendly"
