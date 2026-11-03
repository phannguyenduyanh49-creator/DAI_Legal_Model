import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { answerOutcomeNotice, citationMarkersAreTrusted, stripUnverifiedCitationMarkers } from "./answerPresentation.js";

const evidenceBackedPipeline = {
  evidence_integrity_status: "passed",
  evidence_count: 1,
  citation_verification: "passed",
  confidence_status: "not_verified",
  claim_grounding_status: "not_evaluated",
};

test("unverified answers explain that grounding or confidence was not evaluated", () => {
  const notice = answerOutcomeNotice("answered_unverified", "not_verified", evidenceBackedPipeline);
  assert.equal(notice.tone, "warning");
  assert.ok(notice.text.includes("chưa được xác minh đầy đủ"));
});

test("citation failure has a distinct message and no trusted source marker", () => {
  const notice = answerOutcomeNotice("answered_unverified", "failed", {
    ...evidenceBackedPipeline,
    citation_verification: "failed",
    failure_reason_code: "CITATION_EVIDENCE_ID_UNKNOWN",
  });
  assert.ok(notice.text.includes("Liên kết nguồn tham khảo"));
  assert.equal(stripUnverifiedCitationMarkers("A [Evidence ID: invalid:1] claim"), "A  claim");
  assert.equal(citationMarkersAreTrusted({ citation_verification: "failed" }), false);
  assert.equal(citationMarkersAreTrusted({
    citation_verification: "passed",
    gateway_citation_failure_reason_code: "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
  }), true);
});

test("no Evidence and Evidence integrity failures have separate warnings", () => {
  assert.ok(answerOutcomeNotice("withheld", "not_evaluated", { evidence_count: 0 }).text.includes("chưa tìm thấy tài liệu"));
  assert.ok(answerOutcomeNotice("answered_unverified", "failed", {
    evidence_integrity_status: "failed",
  }).text.includes("định danh hoặc domain"));
});

test("model generation failure is not mislabeled as insufficient Evidence", () => {
  const notice = answerOutcomeNotice("withheld", "not_evaluated", {
    evidence_count: 1,
    failure_stage: "model_api",
    model_api: "error",
    failure_reason_code: "MODEL_API_REQUEST_FAILED",
  });
  assert.equal(notice.tone, "error");
  assert.ok(notice.text.includes("Model"));
  assert.ok(!notice.text.includes("chÆ°a tÃ¬m tháº¥y tÃ i liá»‡u"));
});

test("withheld citation failure is shown as a citation failure", () => {
  const notice = answerOutcomeNotice("withheld", "not_evaluated", {
    evidence_count: 1,
    citation_verification: "failed",
    failure_reason_code: "CITATION_EVIDENCE_ID_UNKNOWN",
  });
  assert.equal(notice.tone, "error");
  const genericWithheld = answerOutcomeNotice("withheld", "not_evaluated", { evidence_count: 1 });
  assert.notEqual(notice.text, genericWithheld.text);
});

test("suggested question content and containers are absent from the chat component", () => {
  const source = readFileSync(new URL("./App.jsx", import.meta.url), "utf8");
  assert.doesNotMatch(source, /QUICK_QUESTIONS|suggestion-grid|suggestion-card|quick-question-row/);
});

test("withheld outcome explains insufficient evidence; verifier failure is specific", () => {
  assert.ok(answerOutcomeNotice("withheld", "not_evaluated", { evidence_count: 1 }).text.includes("chưa tìm đủ tài liệu thiết yếu"));
  assert.ok(answerOutcomeNotice("verification_failed", "failed", {}).text.includes("kiểm tra chất lượng"));
});

test("answered is called verified only when semantic grounding and integrity passed", () => {
  const verified = answerOutcomeNotice("answered", "passed", {
    evidence_integrity_status: "passed",
    citation_verification: "passed",
    claim_grounding_status: "passed",
    verification_status: "passed",
    verification_outcome: "verified",
    verifier_scope: "citation_integrity_and_semantic_grounding",
  });
  assert.equal(verified.tone, "success");
  assert.ok(verified.text.includes("grounding"));

  const incomplete = answerOutcomeNotice("answered", "passed", evidenceBackedPipeline);
  assert.equal(incomplete.tone, "warning");
  assert.ok(incomplete.text.includes("chưa có trong dữ liệu phản hồi"));

  const inconsistent = answerOutcomeNotice("answered", "passed", {
    ...evidenceBackedPipeline,
    citation_verification: "failed",
  });
  assert.equal(inconsistent.tone, "error");
  assert.ok(inconsistent.text.includes("Liên kết nguồn tham khảo"));
});

test("generation failures from a completed API response expose a retry action", () => {
  const source = readFileSync(new URL("./App.jsx", import.meta.url), "utf8");
  assert.match(source, /message\.pipeline\?\.failure_stage === "model_api"[\s\S]*message\.retryQuestion/);
  assert.match(source, /onRetry: \(\) => sendQuestion\(prompt, \{ retry: true \}\)/);
});
