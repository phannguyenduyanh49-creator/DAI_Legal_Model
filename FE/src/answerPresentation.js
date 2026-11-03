export function stripUnverifiedCitationMarkers(content) {
  return String(content ?? "").replace(/\[Evidence ID:\s*[^\]]+\]/gi, "");
}

export function citationMarkersAreTrusted(pipeline = {}) {
  const gatewayFailure = pipeline.gateway_citation_failure_reason_code;
  const citationFailure = typeof gatewayFailure === "string"
    && /^(?:CITATION_|EVIDENCE_ID_|EVIDENCE_IDS_|CITED_EVIDENCE)/.test(gatewayFailure);
  return pipeline.citation_verification === "passed" && !citationFailure;
}

export function safeSourceUrl(value) {
  if (typeof value !== "string" || !value.trim()) return "";
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

export function answerOutcomeNotice(outcome, verifierStatus, pipeline = {}) {
  const citationStatus = pipeline.citation_verification;
  const evidenceStatus = pipeline.evidence_integrity_status;
  const grounding = pipeline.claim_grounding_status;
  const reason = pipeline.failure_reason_code || pipeline.gateway_citation_failure_reason_code;
  const generationFailed = pipeline.failure_stage === "model_api"
    || pipeline.model_api === "error"
    || pipeline.model_api === "failed"
    || reason === "MODEL_API_REQUEST_FAILED";

  if (outcome === "needs_clarification") {
    return { tone: "info", text: "Cần thêm thông tin hoặc quyền tra cứu trước khi đưa ra câu trả lời." };
  }
  if (generationFailed) {
    return {
      tone: "error",
      text: "Model không tạo được câu trả lời do lỗi sinh nội dung. Evidence chưa bị thay đổi; vui lòng thử lại sau.",
    };
  }
  if (outcome === "withheld"
      && evidenceStatus !== "failed"
      && citationStatus !== "failed"
      && citationStatus !== "error"
      && !reason?.startsWith("CITATION_")
      && !["failed", "insufficient_evidence", "unsupported", "contradicted"].includes(grounding)) {
    return {
      tone: "warning",
      text: pipeline.evidence_count === 0 || reason === "EVIDENCE_IDS_MISSING"
        ? "Hệ thống chưa tìm thấy tài liệu phù hợp để trả lời. Bạn có thể bổ sung thông tin hoặc cho phép tra cứu nguồn ngoài."
        : "Hệ thống chưa tìm đủ tài liệu thiết yếu nên đã giữ lại câu trả lời.",
    };
  }
  if (evidenceStatus === "failed") {
    return { tone: "error", text: "Tài liệu nội bộ không vượt qua kiểm tra định danh hoặc domain; câu trả lời đã được giữ lại." };
  }
  if (citationStatus === "failed" || citationStatus === "error"
      || reason?.startsWith("CITATION_") || reason === "CITATION_VERIFICATION_NOT_PASSED") {
    return {
      tone: outcome === "answered_unverified" ? "warning" : "error",
      text: "Liên kết nguồn tham khảo chưa vượt qua kiểm tra. Nội dung hiển thị ở trạng thái chưa xác minh; vui lòng kiểm tra lại tài liệu.",
    };
  }
  if (["failed", "insufficient_evidence", "unsupported", "contradicted"].includes(grounding)) {
    return { tone: "error", text: "Có nội dung chưa được tài liệu hỗ trợ đầy đủ nên câu trả lời đã bị giữ lại." };
  }
  if (outcome === "answered") {
    const fullyVerified = pipeline.verification_status === "passed"
      && pipeline.verification_outcome === "verified"
      && evidenceStatus === "passed"
      && citationStatus === "passed"
      && grounding === "passed"
      && pipeline.verifier_scope?.includes("semantic_grounding");
    return fullyVerified
      ? { tone: "success", text: "Câu trả lời đã vượt qua kiểm tra Evidence, citation và grounding." }
      : { tone: "warning", text: "Câu trả lời đã được tạo; trạng thái xác minh đầy đủ chưa có trong dữ liệu phản hồi." };
  }
  if (outcome === "answered_unverified") {
    if (["evaluator_unavailable", "not_evaluated", "inconclusive"].includes(grounding)
        || ["not_verified", "not_evaluated"].includes(pipeline.confidence_status)
        || verifierStatus === "not_verified") {
      return {
        tone: "warning",
        text: "Câu trả lời đã được tạo từ tài liệu tìm thấy nhưng mức độ hỗ trợ và độ tin cậy chưa được xác minh đầy đủ.",
      };
    }
    return { tone: "warning", text: "Câu trả lời được hiển thị nhưng chưa được xác minh đầy đủ." };
  }
  if (outcome === "verification_failed") {
    return { tone: "error", text: "Câu trả lời không vượt qua bước kiểm tra chất lượng và đã được giữ lại." };
  }
  if (outcome === "FAILED" || (verifierStatus && verifierStatus !== "passed")) {
    return { tone: "warning", text: "Yêu cầu đã được xử lý nhưng câu trả lời chưa được xác minh." };
  }
  return null;
}
