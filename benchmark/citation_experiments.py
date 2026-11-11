"""Read-only, sequential citation prompt experiments against the running Model API.

The fixed inputs are two public DVC records selected in live A01. This is an
offline generation experiment, not a Gateway E2E result or semantic evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from backend.answer.citations import structure_answer_with_explicit_evidence
from backend.evidence.schemas import Evidence, EvidenceBundle
from backend.prompt.assembler import PromptAssembler
from backend.verifier.verifier import Verifier


COLLECTION = "legal_docs_bge_m3"
POINT_IDS = (101718, 101416)
EVIDENCE_IDS = (
    "rag:dvc_019d2bfd-f3d2-7798",
    "rag:dvc_019d2bfe-9088-744d",
)
QUESTION = "Thời hạn giải quyết thủ tục cấp giấy phép xây dựng nhà ở riêng lẻ là bao nhiêu ngày?"


def load_fixed_evidence(client: httpx.Client) -> EvidenceBundle:
    evidences = []
    for point_id, evidence_id in zip(POINT_IDS, EVIDENCE_IDS, strict=True):
        response = client.get(
            f"http://127.0.0.1:6333/collections/{COLLECTION}/points/{point_id}"
        )
        response.raise_for_status()
        point = response.json()["result"]
        payload = point["payload"]
        if not payload.get("full_text") or not payload.get("formality_id"):
            raise ValueError(f"Qdrant point {point_id} has incomplete source content")
        if not evidence_id.startswith(f"rag:dvc_{payload['formality_id'][:18]}"):
            raise ValueError(f"Qdrant point {point_id} changed source identity")
        evidences.append(Evidence(
            evidence_id=evidence_id,
            source_type="rag",
            source_id=payload["formality_id"],
            title=payload.get("procedure_name"),
            content=payload["full_text"],
            source_url=payload.get("source_url"),
            metadata={"point_id": point_id, "domain_id": "administrative_law"},
        ))
    return EvidenceBundle(query=QUESTION, evidences=evidences, total=len(evidences))


def prompts(bundle: EvidenceBundle) -> dict[str, str]:
    evidence = "\n".join(PromptAssembler._format_evidence(item) for item in bundle.evidences)
    allowed = ", ".join(item.evidence_id for item in bundle.evidences)
    base = f"Câu hỏi: {QUESTION}\n=== EVIDENCE ===\n{evidence}\nAllowed Evidence IDs: {allowed}\n"
    concise = (
        base
        + "Chỉ trả lời phần Evidence hỗ trợ. Mỗi câu nêu sự kiện phải kết thúc ngay "
        "bằng [Evidence ID: ID-chính-xác] lấy từ danh sách trên. "
        "Viết câu ngắn, trích dẫn từ câu đầu. Nếu thiếu căn cứ, nói không đủ dữ liệu."
    )
    return {
        "E0_current": PromptAssembler().build(
            QUESTION, bundle,
            domain_prompt_template="Hướng dẫn thủ tục hành chính theo dữ liệu: {context}\nCâu hỏi: {query}",
        ),
        "E1_concise": concise,
        "E2_example": (
            concise
            + "\nVí dụ CÚ PHÁP, không phải thông tin pháp lý: "
            + f"'Một mệnh đề được nguồn hỗ trợ. [Evidence ID: {bundle.evidences[0].evidence_id}]' "
            + "Hãy thay mệnh đề ví dụ bằng nội dung thực sự có trong Evidence."
        ),
        "E3_structured": (
            base
            + "Trả về duy nhất JSON hợp lệ có dạng "
            '{"claims":[{"text":"mệnh đề được Evidence hỗ trợ",'
            '"evidence_ids":["ID-chính-xác"]}]}. '
            "Mỗi claim phải có ID từ danh sách; không có claim được hỗ trợ thì "
            '{"claims":[]}. Không thêm dữ kiện ngoài Evidence.'
        ),
    }


def assess(answer: str, bundle: EvidenceBundle, variant: str) -> dict:
    if variant == "E3_structured":
        try:
            parsed = json.loads(answer)
            claims = parsed["claims"]
            if not isinstance(claims, list):
                raise ValueError("claims is not a list")
            allowed = {e.evidence_id for e in bundle.evidences}
            valid_claims = all(
                isinstance(claim, dict)
                and isinstance(claim.get("text"), str)
                and bool(claim["text"].strip())
                and isinstance(claim.get("evidence_ids"), list)
                and bool(claim["evidence_ids"])
                and all(isinstance(item, str) and item in allowed for item in claim["evidence_ids"])
                for claim in claims
            )
            ids = [
                evidence_id for claim in claims if isinstance(claim, dict)
                for evidence_id in claim.get("evidence_ids", [])
                if isinstance(evidence_id, str)
            ]
            return {
                "json_valid": True,
                "claim_count": len(claims),
                "claims_with_citation": sum(
                    bool(claim.get("evidence_ids"))
                    for claim in claims if isinstance(claim, dict)
                ),
                "unknown_evidence_ids": sorted(set(ids) - allowed),
                "citation_integrity": (
                    "passed" if claims and valid_claims else "failed"
                ),
                "grounding_status": "not_evaluated",
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"json_valid": False, "citation_integrity": "failed"}
    structured = structure_answer_with_explicit_evidence(answer)
    verification = Verifier().verify(structured, bundle)
    return {
        "json_valid": None,
        "claim_count": len(structured.claims),
        "claims_with_citation": sum(bool(claim.evidence_ids) for claim in structured.claims),
        "parser_errors": structured.metadata.get("citation_parse_errors", []),
        "citation_integrity": "passed" if verification.citation_integrity_passed else "failed",
        "grounding_status": verification.claim_grounding_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=[
        "E0_current", "E1_concise", "E2_example", "E3_structured"
    ])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("benchmark/results") / (
        "citation_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with httpx.Client(timeout=180) as client:
        bundle = load_fixed_evidence(client)
        variants = prompts(bundle)
        rows = []
        for name in args.variants:
            if name not in variants:
                raise ValueError(f"Unknown variant: {name}")
            prompt = variants[name]
            started = time.perf_counter()
            response = client.post(
                "http://127.0.0.1:8001/generate",
                json={"prompt": prompt, "max_new_tokens": args.max_new_tokens},
                headers={"X-Request-ID": f"citation_exp_{name}"},
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            response.raise_for_status()
            result = response.json()
            answer = str(result.get("answer") or "")
            row = {
                "variant": name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": "Qwen/Qwen2.5-7B-Instruct + LoRA V4",
                "source_point_ids": POINT_IDS,
                "evidence_ids": EVIDENCE_IDS,
                "evidence_sha256": [
                    hashlib.sha256(e.content.encode("utf-8")).hexdigest()
                    for e in bundle.evidences
                ],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_chars": len(prompt),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "max_new_tokens": args.max_new_tokens,
                "model_processing_time_ms": result.get("processing_time_ms"),
                "e2e_latency_ms": elapsed_ms,
                "answer": answer,
                "assessment": assess(answer, bundle, name),
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "answer"}, ensure_ascii=True), flush=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
