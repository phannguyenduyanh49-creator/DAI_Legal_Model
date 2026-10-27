import pytest

from backend.mcp.dvc_search import (
    DVCUpstreamError,
    extract_dvc_rows,
    fetch_dvc_rows,
    procedure_match_score,
    rank_local_index,
    select_best_procedure,
    validate_dvc_detail_response,
)


QUERY = (
    "T\u00f4i c\u1ea7n l\u00e0m th\u1ee7 t\u1ee5c \u0111\u0103ng k\u00fd "
    "h\u1ed9 kinh doanh tr\u1ef1c tuy\u1ebfn: h\u1ed3 s\u01a1 g\u1ed3m "
    "nh\u1eefng g\u00ec v\u00e0 th\u1eddi h\u1ea1n gi\u1ea3i quy\u1ebft bao l\u00e2u?"
)

RELEVANT = {
    "id": "procedure-household-business",
    "procedure_name": "\u0110\u0103ng k\u00fd th\u00e0nh l\u1eadp h\u1ed9 kinh doanh",
    "code": "1.001612",
}

UNRELATED = {
    "id": "procedure-war-benefit",
    "procedure_name": (
        "Gi\u1ea3i quy\u1ebft ch\u1ebf \u0111\u1ed9 \u0111\u1ed1i v\u1edbi "
        "qu\u00e2n nh\u00e2n th\u1eddi k\u1ef3 kh\u00e1ng chi\u1ebfn"
    ),
}


def test_full_query_selects_relevant_title_instead_of_generic_resolution_match():
    cancelled = {
        "id": "procedure-stop-household-business",
        "procedure_name": (
            "D\u1eebng th\u1ef1c hi\u1ec7n th\u1ee7 t\u1ee5c "
            "\u0111\u0103ng k\u00fd h\u1ed9 kinh doanh"
        ),
    }
    selected = select_best_procedure([UNRELATED, cancelled, RELEVANT], QUERY)

    assert selected == RELEVANT
    assert procedure_match_score(UNRELATED, QUERY) is None
    assert procedure_match_score(cancelled, QUERY) is None


def test_accented_and_unaccented_query_match_the_same_procedure():
    unaccented = "dang ky thanh lap ho kinh doanh"

    assert select_best_procedure([RELEVANT], QUERY) == RELEVANT
    assert select_best_procedure([RELEVANT], unaccented) == RELEVANT


def test_unrelated_query_and_empty_candidates_return_no_match():
    assert select_best_procedure([RELEVANT], "quy trinh tro cap that nghiep") is None
    assert select_best_procedure([], QUERY) is None
    assert rank_local_index([], QUERY) == []


def test_multiple_candidates_are_ranked_by_title_relevance():
    similar_but_less_specific = {
        "id": "procedure-tax-update",
        "procedure_name": (
            "\u0110\u0103ng k\u00fd thay \u0111\u1ed5i th\u00f4ng tin "
            "thu\u1ebf cho h\u1ed9 kinh doanh"
        ),
    }
    assert select_best_procedure(
        [UNRELATED, similar_but_less_specific, RELEVANT], QUERY
    ) == RELEVANT


def test_tied_candidates_return_no_match_instead_of_arbitrary_fallback():
    duplicate = dict(RELEVANT, id="second-household-business")
    assert select_best_procedure([RELEVANT, duplicate], QUERY) is None


def test_optional_code_and_metadata_are_not_required_for_a_title_match():
    row = {"id": "without-code", "name": RELEVANT["procedure_name"]}
    ranked = rank_local_index([row], "dang ky thanh lap ho kinh doanh")

    assert len(ranked) == 1
    assert ranked[0]["id"] == "without-code"
    assert ranked[0]["row"] == row


def test_empty_official_search_result_is_valid_no_match():
    assert extract_dvc_rows({"code": "OK", "data": {"rows": []}}) == []


def test_upstream_timeout_is_not_converted_to_an_empty_search():
    def timeout(_endpoint, _payload):
        raise TimeoutError("timeout")

    with pytest.raises(DVCUpstreamError, match="request failed"):
        fetch_dvc_rows("keyword", timeout, "https://dichvucong.gov.vn/search")


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"code": "FAILED", "data": {"rows": []}},
        {"code": "OK", "data": {"unexpected": []}},
        {"code": "OK", "data": {"rows": "not-a-list"}},
    ],
)
def test_invalid_upstream_response_shapes_fail_closed(response):
    with pytest.raises(DVCUpstreamError):
        extract_dvc_rows(response)


def test_fetch_uses_current_query_and_expected_dvc_search_contract():
    captured = {}

    def post(endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"code": "OK", "data": {"rows": [RELEVANT]}}

    rows = fetch_dvc_rows("registered household business", post, "https://dvc/search")

    assert rows == [RELEVANT]
    assert captured["payload"]["q"] == "registered household business"
    assert captured["payload"]["limit"] == 10


def test_detail_response_must_be_valid_and_match_selected_id():
    validate_dvc_detail_response(
        {"code": "OK", "data": {"id": "official-1", "executionSteps": ["step"]}},
        "official-1",
    )
    with pytest.raises(DVCUpstreamError, match="ID does not match"):
        validate_dvc_detail_response(
            {"code": "OK", "data": {"id": "other", "executionSteps": ["step"]}},
            "official-1",
        )
    with pytest.raises(DVCUpstreamError, match="missing detail data"):
        validate_dvc_detail_response({"code": "OK", "data": {}}, "official-1")


def test_real_qdrant_payload_titles_keep_both_long_qualified_lltp_procedures():
    from backend.rag.relevance import procedure_relevance

    query = "C\u1ea5p phi\u1ebfu l\u00fd l\u1ecbch t\u01b0 ph\u00e1p c\u1ea7n l\u00e0m g\u00ec?"
    # Actual procedure_name/thu_tuc values from DVC rows 102075 and 102076.
    titles = [
        "C\u1ea5p Phi\u1ebfu l\u00fd l\u1ecbch t\u01b0 ph\u00e1p cho c\u00f4ng d\u00e2n Vi\u1ec7t Nam m\u00e0 kh\u00f4ng x\u00e1c \u0111\u1ecbnh \u0111\u01b0\u1ee3c n\u01a1i th\u01b0\u1eddng tr\u00fa ho\u1eb7c n\u01a1i t\u1ea1m tr\u00fa; ng\u01b0\u1eddi n\u01b0\u1edbc ngo\u00e0i \u0111\u00e3 c\u01b0 tr\u00fa t\u1ea1i Vi\u1ec7t Nam",
        "C\u1ea5p Phi\u1ebfu L\u00fd l\u1ecbch t\u01b0 ph\u00e1p cho c\u00f4ng d\u00e2n Vi\u1ec7t Nam, ng\u01b0\u1eddi n\u01b0\u1edbc ngo\u00e0i \u0111ang c\u01b0 tr\u00fa t\u1ea1i Vi\u1ec7t Nam, ng\u01b0\u1eddi Vi\u1ec7t Nam \u0111ang c\u01b0 tr\u00fa \u1edf n\u01b0\u1edbc ngo\u00e0i",
    ]
    for title in titles:
        result = procedure_relevance(query, title)
        assert result["relevant"] is True
        assert result["query_coverage"] == 1.0


@pytest.mark.parametrize(
    "title",
    [
        "Th\u1ee7 t\u1ee5c b\u1ed5 nhi\u1ec7m gi\u00e1m \u0111\u1ecbnh vi\u00ean t\u01b0 ph\u00e1p",
        "\u0110\u0103ng k\u00fd khai t\u1eed",
    ],
)
def test_lltp_query_does_not_accept_legal_examiner_or_death_registration(title):
    from backend.rag.relevance import procedure_relevance

    query = "C\u1ea5p phi\u1ebfu l\u00fd l\u1ecbch t\u01b0 ph\u00e1p c\u1ea7n l\u00e0m g\u00ec?"
    assert procedure_relevance(query, title)["relevant"] is False

