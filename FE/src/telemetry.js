const SAFE_REASON_CODES = new Set([
  "VERIFIER_NOT_PASSED",
  "CITATION_VERIFICATION_NOT_PASSED",
  "EVIDENCE_IDS_MISSING",
  "CITED_EVIDENCE_IDS_MISSING",
  "EVIDENCE_ID_NOT_ALLOWED",
  "CITATION_TELEMETRY_MISMATCH",
  "UPSTREAM_TIMEOUT",
  "UPSTREAM_HTTP_ERROR",
  "UPSTREAM_CONNECTION_ERROR",
  "UNKNOWN_PIPELINE_FAILURE",
]);

const SAFE_FAILURE_STAGES = new Set([
  "gateway_citation_validation", "verifier", "model_api", "model_api_response",
  "generation_pipeline", "retrieval_pipeline", "legacy_api",
  "legacy_api_response_parse", "gateway",
]);
const SAFE_VERIFIER_REASONS = new Set([
  "CLAIM_EMPTY", "CLAIM_WITHOUT_EVIDENCE", "EVIDENCE_ID_NOT_FOUND",
  "EVIDENCE_CONTENT_EMPTY", "VERIFIER_REJECTED", "NO_CLAIMS_TO_VERIFY",
]);
const SAFE_STATES = new Set([
  "not_started", "not_observed", "not_evaluated", "not_integrated",
  "unknown", "unavailable", "completed", "passed", "failed", "error",
  "sufficient", "insufficient", "matched", "no_match", "delegated",
  "not_allowed", "not_required", "resolved", "partial_or_unknown",
]);
const SAFE_STAGE_LABELS = {
  catalog_lookup: "Catalog",
  retrieval: "Retrieval",
  local_retrieval: "Tìm kiếm nội bộ",
  reranking: "Reranking",
  evidence_and_sufficiency: "Evidence và độ đầy đủ dữ liệu",
  evidence_builder: "Evidence Builder",
  data_sufficiency: "Data sufficiency",
  prompt_assembly: "Prompt Assembly",
  model_api: "Model API",
  citation_verification: "Verifier / citation",
  query_analysis: "Phân tích câu hỏi",
  orchestrator: "Điều phối",
  domain_policy: "Domain policy",
  legacy_api: "Core API",
  gateway_citation_validation: "Gateway citation validation",
};
const SAFE_STAGE_STATUS_KEYS = {
  catalog_lookup: "Catalog",
  retrieval: "Retrieval",
  reranking: "Reranking",
  evidence_builder: "Evidence Builder",
  data_sufficiency: "Data sufficiency",
  prompt_assembly: "Prompt Assembly",
  model_api: "Model API",
  gateway_citation_validation: "Gateway citation validation",
};

function safeState(value) {
  return typeof value === "string" && SAFE_STATES.has(value) ? value : null;
}

export function pipelineTelemetryRows(pipeline = {}, requestId) {
  const rows = [];
  const add = (label, value) => {
    if (value !== undefined && value !== null && value !== "") rows.push([label, String(value)]);
  };
  const id = requestId || pipeline.request_id;
  if (typeof id === "string" && /^trace_legal_[a-f0-9]{32}$/.test(id)) add("Request ID", id);
  if (SAFE_FAILURE_STAGES.has(pipeline.failure_stage)) add("Stage thất bại", pipeline.failure_stage);
  if (SAFE_REASON_CODES.has(pipeline.failure_reason_code)) add("Mã nguyên nhân", pipeline.failure_reason_code);
  if (SAFE_REASON_CODES.has(pipeline.gateway_citation_failure_reason_code)) {
    add("Gateway validation", pipeline.gateway_citation_failure_reason_code);
  }

  for (const [key, label] of Object.entries(SAFE_STAGE_STATUS_KEYS)) {
    const value = safeState(pipeline[key] ?? pipeline.stage_status?.[key]);
    if (value) add(label, value);
  }
  const verifierState = safeState(pipeline.verifier_status ?? pipeline.verifier ?? pipeline.stage_status?.verifier);
  if (verifierState) {
    add("Verifier", verifierState);
  }
  if (safeState(pipeline.citation_verification)) {
    add("Citation verification", safeState(pipeline.citation_verification));
  }

  if (pipeline.verifier_reason_counts && typeof pipeline.verifier_reason_counts === "object") {
    const reasons = Object.entries(pipeline.verifier_reason_counts)
      .filter(([code, count]) => SAFE_VERIFIER_REASONS.has(code) && Number.isInteger(count) && count >= 0)
      .map(([code, count]) => `${code}: ${count}`);
    if (reasons.length) add("Lý do verifier", reasons.join(", "));
  }
  if (pipeline.stage_durations_ms && typeof pipeline.stage_durations_ms === "object") {
    for (const [stage, duration] of Object.entries(pipeline.stage_durations_ms)) {
      const label = SAFE_STAGE_LABELS[stage];
      if (label && Number.isFinite(duration) && duration >= 0) add(`Thời gian ${label}`, `${duration} ms`);
    }
  }
  const upstreamStatus = Number(pipeline.upstream_status_code);
  if (Number.isInteger(upstreamStatus) && upstreamStatus >= 100 && upstreamStatus <= 599) {
    add("HTTP upstream", upstreamStatus);
  }
  if (typeof pipeline.exception_type === "string" && /^[A-Za-z][A-Za-z0-9_]{0,79}$/.test(pipeline.exception_type)) {
    add("Loại lỗi", pipeline.exception_type);
  }
  return rows;
}

export function safeFailureTelemetry(telemetry = {}) {
  return pipelineTelemetryRows({
    request_id: telemetry.request_id,
    failure_stage: telemetry.failure_stage,
    failure_reason_code: telemetry.failure_reason_code,
    exception_type: telemetry.exception_type,
    upstream_status_code: telemetry.upstream_status_code,
    stage_durations_ms: telemetry.stage_durations_ms,
  }, telemetry.request_id);
}
