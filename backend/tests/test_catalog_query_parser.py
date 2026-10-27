from types import SimpleNamespace

import pytest

from backend.catalog.matcher import CatalogMatcher
from backend.catalog.query_parser import catalog_query_from_text


@pytest.mark.parametrize(
    "question",
    [
        "What is the name of the administrative procedure with code 1.002421?",
        "Tên thủ tục có mã 1.002421 là gì?",
    ],
)
def test_explicit_catalog_procedure_code_uses_exact_code_lookup(question):
    query = catalog_query_from_text(question)
    assert query.code == "1.002421"
    assert query.procedure_name is None

    expected = SimpleNamespace(procedure_name="procedure name")

    class Repository:
        def find_by_code(self, code):
            assert code == "1.002421"
            return expected

        def find_by_formality_id(self, _value):
            raise AssertionError("unexpected formality lookup")

        def find_by_name(self, _value):
            raise AssertionError("exact code must take precedence")

        def find_name_candidates(self, _tokens):
            raise AssertionError("exact code must not fall back to fuzzy name")

    matcher = CatalogMatcher()
    matcher.repository = Repository()
    result = matcher.match(query)
    assert result.matched is True
    assert result.record is expected
    assert result.match_type == "exact_code"


def test_natural_language_without_explicit_code_keeps_existing_name_lookup():
    question = "Tôi muốn biết thủ tục đăng ký tạm trú"
    query = catalog_query_from_text(question)
    assert query.code is None
    assert query.procedure_name == question


def test_multiple_explicit_codes_do_not_guess_which_record_to_lookup():
    question = "Compare code 1.002421 with 1.000123"
    query = catalog_query_from_text(question)
    assert query.code is None
    assert query.procedure_name == question


def test_code_embedded_in_alphanumeric_text_is_not_treated_as_catalog_code():
    query = catalog_query_from_text("reference X1.002421Z")
    assert query.code is None
    assert query.procedure_name == "reference X1.002421Z"
