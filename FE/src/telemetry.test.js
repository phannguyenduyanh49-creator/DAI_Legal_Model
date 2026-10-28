import test from "node:test";
import assert from "node:assert/strict";
import {
  pipelineTelemetryRows,
  safeCitationAttemptDiagnostics,
  safeFailureTelemetry,
} from "./telemetry.js";

test("telemetry displays allowlisted failure reason, request id, statuses and durations", () => {
  const rows = pipelineTelemetryRows({
    request_id: "trace_legal_0123456789abcdef0123456789abcdef",
    failure_stage: "model_api",
    failure_reason_code: "UPSTREAM_TIMEOUT",
    retrieval: "completed",
    model_api: "error",
    verifier: "error",
    stage_durations_ms: { retrieval: 12.5, model_api: 90000 },
  });
  assert.deepEqual(rows.slice(0, 3), [
    ["Request ID", "trace_legal_0123456789abcdef0123456789abcdef"],
    ["Stage thất bại", "model_api"],
    ["Mã nguyên nhân", "UPSTREAM_TIMEOUT"],
  ]);
  assert.ok(rows.some(([label, value]) => label === "Thời gian Retrieval" && value === "12.5 ms"));
});

test("untrusted telemetry strings and unsupported fields are not displayed", () => {
  const rows = pipelineTelemetryRows({
    request_id: "user@example.com",
    failure_stage: "C:\\private\\stack.py",
    failure_reason_code: "raw question text",
    exception_type: "C:\\secret\\file.py",
    retrieval: "private query",
    stage_durations_ms: { private_path: 42 },
  });
  assert.deepEqual(rows, []);
});

test("older response without optional telemetry produces no failure rows", () => {
  assert.deepEqual(safeFailureTelemetry({}), []);
  assert.deepEqual(pipelineTelemetryRows({ verifier: "passed" }), [["Verifier", "passed"]]);
});

test("retrieval collection availability uses only allowlisted domain states", () => {
  assert.deepEqual(pipelineTelemetryRows({
    stage_status: { retrieval_collection_status: "base_collection_missing" },
  }), [["Retrieval collection", "base_collection_missing"]]);
  assert.deepEqual(pipelineTelemetryRows({
    stage_status: { retrieval_collection_status: "private_collection_name" },
  }), []);
});

test("attempt diagnostics expose safe counts but never raw model text or unknown ID values", () => {
  const safe = safeCitationAttemptDiagnostics([{
    attempt: 2,
    raw_answer: "private model answer",
    raw_answer_chars: 1200,
    claim_count: 4,
    claims_with_citation: 1,
    allowed_evidence_count: 1,
    cited_evidence_count: 1,
    unknown_evidence_ids: ["private-evidence-title"],
    parser_errors: [],
    verifier_reason_counts: { CLAIM_WITHOUT_EVIDENCE: 3, private_reason: 1 },
    claim_grounding_status: "not_evaluated",
    confidence_status: "not_verified",
  }]);

  assert.equal(safe.length, 1);
  assert.equal(safe[0].unknown_evidence_count, 1);
  assert.deepEqual(safe[0].verifier_reasons, ["CLAIM_WITHOUT_EVIDENCE: 3"]);
  assert.equal(JSON.stringify(safe).includes("private"), false);
  assert.equal(JSON.stringify(safe).includes("1200"), true);
});

test("known citation and grounding failure codes are visible to the frontend", () => {
  for (const failure_reason_code of [
    "MODEL_API_REQUEST_FAILED",
    "CITATION_FORMAT_INVALID",
    "CLAIM_GROUNDING_NOT_EVALUATED",
    "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
  ]) {
    const rows = pipelineTelemetryRows({ failure_stage: "verifier", failure_reason_code });
    assert.ok(rows.some(([, value]) => value === failure_reason_code));
  }
});

test("empty Evidence failures identify retrieval, builder, or prompt stage", () => {
  for (const failure_stage of ["retrieval_pipeline", "evidence_builder", "prompt_assembly"]) {
    const rows = pipelineTelemetryRows({
      failure_stage,
      failure_reason_code: "EVIDENCE_IDS_MISSING",
    });
    assert.ok(rows.some(([label, value]) => label === "Stage thất bại" && value === failure_stage));
    assert.ok(rows.some(([, value]) => value === "EVIDENCE_IDS_MISSING"));
  }
});
