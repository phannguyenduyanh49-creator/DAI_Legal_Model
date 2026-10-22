from types import SimpleNamespace

import pytest

from backend.catalog.matcher import CatalogMatcher
from backend.catalog.repository import CatalogRepository
from backend.catalog.schemas import CatalogQuery
from backend.evidence.normalizer import catalog_record_to_evidence


def procedure(name, *, code="A1", formality_id="F1"):
    return SimpleNamespace(
        procedure_name=name,
        code=code,
        formality_id=formality_id,
    )


class FakeRepository:
    def __init__(self, *, by_id=None, by_code=None, by_name=None, candidates=None):
        self.by_id = by_id
        self.by_code = by_code
        self.by_name = by_name
        self.candidates = candidates or []
        self.candidate_tokens = None

    def find_by_formality_id(self, _value):
        return self.by_id

    def find_by_code(self, _value):
        return self.by_code

    def find_by_name(self, _value):
        return self.by_name

    def find_name_candidates(self, tokens):
        self.candidate_tokens = tokens
        return self.candidates


def catalog_query(value, **kwargs):
    return CatalogQuery(
        domain="legal", intent="procedure_lookup", procedure_name=value, **kwargs
    )


def test_exact_code_match_has_priority_over_natural_language_candidates():
    exact = procedure("Thành lập hộ kinh doanh")
    repo = FakeRepository(by_code=exact, candidates=[procedure("Other")])
    matcher = CatalogMatcher()
    matcher.repository = repo

    result = matcher.match(catalog_query("unrelated wording", code="A1"))

    assert result.matched
    assert result.record is exact
    assert result.match_type == "exact_code"
    assert repo.candidate_tokens is None


def test_exact_procedure_name_is_preserved():
    exact = procedure("Đăng ký thành lập hộ kinh doanh")
    matcher = CatalogMatcher()
    matcher.repository = FakeRepository(by_name=exact)

    result = matcher.match(catalog_query(exact.procedure_name))

    assert result.matched
    assert result.record is exact
    assert result.match_type == "exact_name"


@pytest.mark.parametrize(
    "query",
    [
        "Tôi muốn hỏi thủ tục đăng ký thành lập hộ kinh doanh ở đâu?",
        "toi muon hoi thu tuc dang ky thanh lap ho kinh doanh o dau",
    ],
)
def test_natural_language_match_supports_accented_and_unaccented_query(query):
    candidate = procedure("Đăng ký thành lập hộ kinh doanh")
    matcher = CatalogMatcher()
    matcher.repository = FakeRepository(candidates=[candidate])

    result = matcher.match(catalog_query(query))

    assert result.matched
    assert result.record is candidate
    assert result.match_type == "natural_language_name"
    assert 0.58 <= result.confidence <= 1.0


def test_unrelated_query_does_not_match_generic_title_overlap():
    matcher = CatalogMatcher()
    matcher.repository = FakeRepository(
        candidates=[procedure("Giải quyết chế độ đối với quân nhân")]
    )

    result = matcher.match(catalog_query("đăng ký thành lập hộ kinh doanh"))

    assert not result.matched
    assert result.match_type == "none"


def test_near_tied_candidates_return_ambiguous_instead_of_first_result():
    candidates = [
        procedure("Đăng ký thành lập hộ kinh doanh ngành dịch vụ", code="A1"),
        procedure("Đăng ký thành lập hộ kinh doanh ngành sản xuất", code="A2"),
    ]
    matcher = CatalogMatcher()
    matcher.repository = FakeRepository(candidates=candidates)

    result = matcher.match(catalog_query("đăng ký thành lập hộ kinh doanh ngành"))

    assert not result.matched
    assert result.match_type == "ambiguous"


def test_empty_catalog_returns_no_match():
    matcher = CatalogMatcher()
    matcher.repository = FakeRepository()

    result = matcher.match(catalog_query("đăng ký thành lập hộ kinh doanh"))

    assert not result.matched
    assert result.match_type == "none"


def test_catalog_query_errors_are_not_silently_converted_to_no_match():
    class BrokenRepository(FakeRepository):
        def find_name_candidates(self, _tokens):
            raise RuntimeError("query failed")

    matcher = CatalogMatcher()
    matcher.repository = BrokenRepository()

    with pytest.raises(RuntimeError, match="query failed"):
        matcher.match(catalog_query("đăng ký thành lập hộ kinh doanh"))


def test_repository_maps_procedure_orm_to_evidence_catalog_contract():
    row = SimpleNamespace(
        formality_id="procedure-123",
        code="A1",
        procedure_name="Đăng ký hộ kinh doanh",
        target_type="Tổ chức",
        category_name="Đăng ký kinh doanh",
        description="Mô tả",
        execution_steps="Bước 1",
        execution_methods="Trực tuyến",
        profile_components="Đơn đề nghị",
        requirements_conditions="Điều kiện",
        fees="Không",
        legal_basis="Văn bản hiện hành",
        results="Giấy chứng nhận",
        target_objects="Hộ kinh doanh",
        department_promulgate="Cơ quan ban hành",
        executing_agencies="Cơ quan tiếp nhận",
        coordinating_agencies=None,
        source="official_catalog",
        source_url="https://example.invalid/procedure",
    )

    record = CatalogRepository._to_catalog_record(row)
    evidence = catalog_record_to_evidence(record)

    assert record.record_id == "procedure-123"
    assert evidence.evidence_id == "catalog:procedure-123"
    assert evidence.source_id == "procedure-123"
    assert record.source.source_type == "official_catalog"
    assert evidence.source_type == "direct_catalog"
    assert evidence.source_url == "https://example.invalid/procedure"
    assert "Đăng ký hộ kinh doanh" in evidence.content
