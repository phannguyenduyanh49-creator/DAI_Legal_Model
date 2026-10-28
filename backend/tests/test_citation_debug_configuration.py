from backend.answer.diagnostics import citation_raw_debug_enabled
from run_gateway_stack import build_child_environment


def test_stack_launcher_passes_explicit_debug_environment_to_each_child():
    parent_environment = {
        "DAI_ENVIRONMENT": "development",
        "DAI_DEBUG_CITATION_RAW_ANSWER": "true",
    }

    child_environment = build_child_environment(parent_environment)

    assert child_environment["DAI_ENVIRONMENT"] == "development"
    assert child_environment["DAI_DEBUG_CITATION_RAW_ANSWER"] == "true"
    assert citation_raw_debug_enabled(child_environment) is True


def test_stack_launcher_binds_services_to_loopback_by_default():
    child_environment = build_child_environment({})

    assert child_environment["CORE_API_HOST"] == "127.0.0.1"
    assert child_environment["MODEL_API_HOST"] == "127.0.0.1"
    assert child_environment["CORE_GATEWAY_HOST"] == "127.0.0.1"


def test_stack_launcher_preserves_explicit_service_bind_overrides():
    parent_environment = {
        "CORE_API_HOST": "0.0.0.0",
        "MODEL_API_HOST": "10.0.0.8",
        "CORE_GATEWAY_HOST": "192.168.1.10",
    }

    child_environment = build_child_environment(parent_environment)

    assert child_environment["CORE_API_HOST"] == "0.0.0.0"
    assert child_environment["MODEL_API_HOST"] == "10.0.0.8"
    assert child_environment["CORE_GATEWAY_HOST"] == "192.168.1.10"


def test_raw_answer_debug_requires_both_development_and_explicit_opt_in():
    assert citation_raw_debug_enabled({}) is False
    assert citation_raw_debug_enabled({
        "DAI_ENVIRONMENT": "production",
        "DAI_DEBUG_CITATION_RAW_ANSWER": "true",
    }) is False
    assert citation_raw_debug_enabled({
        "DAI_ENVIRONMENT": "local",
        "DAI_DEBUG_CITATION_RAW_ANSWER": "1",
    }) is True
