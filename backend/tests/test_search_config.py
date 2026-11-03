import pytest

from backend.search.config import resolve_search_config


def test_search_defaults_to_existing_qdrant_backend():
    assert resolve_search_config({}) == ("qdrant", None)


def test_elasticsearch_requires_explicit_calibrated_minimum_score():
    with pytest.raises(ValueError, match="ELASTICSEARCH_MIN_SCORE"):
        resolve_search_config({"SEARCH_BACKEND": "elasticsearch"})


def test_elasticsearch_accepts_nonnegative_configured_minimum_score():
    assert resolve_search_config({
        "SEARCH_BACKEND": "elasticsearch",
        "ELASTICSEARCH_MIN_SCORE": "1.25",
    }) == ("elasticsearch", 1.25)


@pytest.mark.parametrize("raw", ["invalid", "-0.1"])
def test_elasticsearch_rejects_invalid_minimum_score(raw):
    with pytest.raises(ValueError):
        resolve_search_config({
            "SEARCH_BACKEND": "elasticsearch",
            "ELASTICSEARCH_MIN_SCORE": raw,
        })
