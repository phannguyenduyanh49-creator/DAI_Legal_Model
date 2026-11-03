from io import BytesIO
from types import SimpleNamespace

from openpyxl import Workbook

from backend.domain_documents import DomainDocumentService
from backend.domain_registry import DomainRegistryService
from backend.evidence.builder import EvidenceBuilder
from backend.evidence.schemas import Evidence, EvidenceBundle
from backend.evidence.source_summary import evidence_source_summary
from backend.prompt.assembler import PromptAssembler
from backend.rag.adapter import rag_result_to_evidence
from backend.verifier.verifier import Verifier
from backend.answer.citations import structure_answer_with_explicit_evidence


class StubQdrant:
    def __init__(self):
        self.collections = set()
        self.points = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)

    def upsert(self, collection_name, points, wait):
        self.points = list(points)
        return SimpleNamespace(status="completed")


class StubEmbedder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **_kwargs):
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_mocked_document_to_evidence_to_prompt_to_verifier_contract():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Products"
    sheet.append(["product", "price"])
    sheet.append(["Example kettle", 490000])
    content = BytesIO()
    workbook.save(content)

    qdrant = StubQdrant()
    indexed = DomainDocumentService(
        qdrant, StubEmbedder(), DomainRegistryService()
    ).upload("shopping", "products.xlsx", content.getvalue())
    assert indexed["status"] == "indexed"
    assert indexed["domain_id"] == "shopping"
    assert len(qdrant.points) == 1

    payload = dict(qdrant.points[0].payload)
    payload["score"] = 0.95
    evidence = rag_result_to_evidence(payload)
    bundle = EvidenceBuilder(min_score=0.3).build(
        rag_evidences=[evidence], query="What is the kettle price?"
    )
    assert bundle.total == 1
    assert bundle.evidences[0].metadata["domain_id"] == "shopping"
    assert bundle.evidences[0].metadata["document_id"] == indexed["document_id"]
    assert bundle.evidences[0].source_id == indexed["document_id"]
    assert bundle.evidences[0].metadata["chunk_index"] == 1
    assert bundle.evidences[0].metadata["row_number"] == 2
    assert "record_id" not in bundle.evidences[0].metadata
    assert bundle.evidences[0].metadata["record_fields"]["price"] == "490000"
    assert evidence_source_summary(bundle.evidences[0])["domain_id"] == "shopping"

    prompt = PromptAssembler().build(bundle.query, bundle)
    assert bundle.evidences[0].evidence_id in prompt
    assert "price: 490000" in prompt

    answer = structure_answer_with_explicit_evidence(
        f"The workbook lists the kettle at 490000. "
        f"[Evidence ID: {bundle.evidences[0].evidence_id}]"
    )
    verification = Verifier().verify(
        answer, bundle, expected_domain_id="shopping"
    )
    assert verification.citation_integrity_passed is True
    assert verification.claim_grounding_status == "not_evaluated"
    assert verification.passed is False
    assert verification.outcome == "evaluator_unavailable"


def test_source_summary_does_not_invent_domain_for_shared_catalog_evidence():
    source = evidence_source_summary(Evidence(
        evidence_id="catalog:procedure-1",
        source_type="direct_catalog",
        source_id="procedure-1",
        title="Catalog procedure",
        metadata={},
    ))
    assert "domain_id" not in source
