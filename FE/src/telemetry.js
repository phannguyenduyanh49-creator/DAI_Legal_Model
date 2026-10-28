const SAFE_REASON_CODES = new Set([
  "VERIFIER_NOT_PASSED",
  "CITATION_VERIFICATION_NOT_PASSED",
  "CITATION_FORMAT_INVALID",
  "CITATION_EVIDENCE_ID_UNKNOWN",
  "CITATION_MISSING",
  "CITATION_NO_CLAIMS",
  "CLAIM_GROUNDING_NOT_EVALUATED",
  "CLAIM_GROUNDING_NOT_PASSED",
  "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
  "VERIFICATION_CONFIDENCE_NOT_PASSED",
  "CONFIDENCE_EVALUATOR_UNAVAILABLE",
  "CONFIDENCE_EVALUATOR_TIMEOUT",
  "GROUNDING_EVALUATOR_UNAVAILABLE",
  "GROUNDING_EVALUATOR_TIMEOUT",
  "MODEL_API_REQUEST_FAILED",
  "MODEL_API_EMPTY_RESPONSE",
  "MODEL_API_GENERATION_FAILED",
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
  "legacy_api_response_parse", "gateway", "evidence_builder", "prompt_assembly",
]);
const SAFE_VERIFIER_REASONS = new Set([
  "CLAIM_EMPTY", "CLAIM_WITHOUT_EVIDENCE", "EVIDENCE_ID_NOT_FOUND",
  "EVIDENCE_CONTENT_EMPTY", "VERIFIER_REJECTED", "NO_CLAIMS_TO_VERIFY",
]);
const SAFE_PARSER_ERRORS = new Set([
  "MALFORMED_CITATION", "CITATION_ON_HEADING", "EMPTY_EVIDENCE_ID",
  "ORPHAN_CITATION", "CITATION_NOT_AT_CLAIM_END",
  "CITATION_WITHOUT_PRECEDING_CLAIM", "PROSE_AFTER_CITATION",
]);
const SAFE_STATES = new Set([
  "not_started", "not_observed", "not_evaluated", "not_integrated",
  "unknown", "unavailable", "completed", "passed", "failed", "error",
  "sufficient", "insufficient", "matched", "no_match", "delegated",
  "not_allowed", "not_required", "resolved", "partial_or_unknown",
  "rejected_by_relevance_filter", "needs_clarification", "withheld",
  "verification_failed", "answered", "answered_unverified", "partial",
  "base_collection_missing", "no_domain_documents", "available",
]);
const SAFE_STAGE_LABELS = {
  catalog_lookup: "Catalog",
  retrieval: "Retrieval",
  retrieval_collection_status: "Retrieval collection",
  local_retrieval: "Tìm kiếm nội bộ",
  mcp_process_start_and_stdio: "MCP process / stdio",
  mcp_initialize: "MCP initialize",
  mcp_tool_call: "MCP tool call",
  mcp_tool_call_timeout: "MCP tool timeout",
  mcp_tool_call_error: "MCP tool error",
  mcp_result_mapping: "MCP Evidence mapping",
  mcp_total: "MCP total",
  reranking: "Reranking",
  evidence_and_sufficiency: "Evidence và độ đầy đủ dữ liệu",
  evidence_builder: "Evidence Builder",
  data_sufficiency: "Data sufficiency",
  prompt_assembly: "Prompt Assembly",
  model_api: "Model API",
  citation_verification: "Verifier / citation",
  embedding: "Embedding",
  qdrant: "Qdrant",
  query_analysis: "Phân tích câu hỏi",
  orchestrator: "Điều phối",
  domain_policy: "Domain policy",
  legacy_api: "Core API",
  gateway_citation_validation: "Gateway citation validation",
};
const SAFE_STAGE_STATUS_KEYS = {
  catalog_lookup: "Catalog",
  retrieval: "Retrieval",
  retrieval_collection_status: "Retrieval collection",
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
  if (SAFE_REASON_CODES.has(pipeline.confidence_reason_code)) {
    add("Confidence evaluator", pipeline.confidence_reason_code);
  }
  if (SAFE_REASON_CODES.has(pipeline.grounding_reason_code)) {
    add("Grounding evaluator", pipeline.grounding_reason_code);
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

export function safeCitationAttemptDiagnostics(attempts) {
  if (!Array.isArray(attempts)) return [];
  return attempts.filter((item) => item && typeof item === "object").map((item) => {
    const counts = item.verifier_reason_counts && typeof item.verifier_reason_counts === "object"
      ? Object.entries(item.verifier_reason_counts)
        .filter(([code, count]) => SAFE_VERIFIER_REASONS.has(code) && Number.isInteger(count) && count >= 0)
        .map(([code, count]) => `${code}: ${count}`)
      : [];
    return {
      attempt: Number.isInteger(item.attempt) && item.attempt > 0 ? item.attempt : null,
      raw_answer_chars: Number.isInteger(item.raw_answer_chars) && item.raw_answer_chars >= 0 ? item.raw_answer_chars : null,
      claim_count: Number.isInteger(item.claim_count) && item.claim_count >= 0 ? item.claim_count : null,
      claims_with_citation: Number.isInteger(item.claims_with_citation) && item.claims_with_citation >= 0 ? item.claims_with_citation : null,
      allowed_evidence_count: Number.isInteger(item.allowed_evidence_count) && item.allowed_evidence_count >= 0 ? item.allowed_evidence_count : null,
      cited_evidence_count: Number.isInteger(item.cited_evidence_count) && item.cited_evidence_count >= 0 ? item.cited_evidence_count : null,
      unknown_evidence_count: Array.isArray(item.unknown_evidence_ids) ? item.unknown_evidence_ids.length : null,
      parser_errors: Array.isArray(item.parser_errors) ? item.parser_errors.filter((code) => SAFE_PARSER_ERRORS.has(code)) : [],
      verifier_reasons: counts,
      claim_grounding_status: safeState(item.claim_grounding_status),
      confidence_status: safeState(item.confidence_status),
    };
  }).filter((item) => item.attempt !== null);
}
