const localHostnames = new Set(["localhost", "127.0.0.1"]);
const configuredGatewayUrl = import.meta.env?.VITE_GATEWAY_API_URL;
const defaultGatewayUrl =
  typeof window !== "undefined" && localHostnames.has(window.location.hostname)
    ? `http://${window.location.hostname}:8002`
    : "";

export const API_BASE_URL = (configuredGatewayUrl || defaultGatewayUrl).replace(/\/$/, "");
export const LEGAL_QUERY_URL = `${API_BASE_URL}/api/v2/legal-query`;
export const GATEWAY_HEALTH_URL = `${API_BASE_URL}/health`;

function assertGatewayConfigured(baseUrl) {
  if (!baseUrl) {
    throw new Error("Chưa cấu hình VITE_GATEWAY_API_URL cho môi trường này.");
  }
}

function safeServerDetail(data) {
  const detail = typeof data?.detail === "string" ? data.detail.trim() : "";
  if (
    !detail
    || detail.length > 300
    || /[\r\n]/.test(detail)
    || /traceback|stack trace|exception|[A-Z]:\\|\/(?:home|root|users)\//i.test(detail)
  ) return "";
  return detail;
}

function safeRequestPath(value) {
  try {
    const url = new URL(value, typeof window !== "undefined" ? window.location.origin : "http://localhost");
    return url.pathname;
  } catch {
    return "";
  }
}

async function responseError(response, { method = "GET", url = "" } = {}) {
  const contentType = response.headers.get("content-type") || "";
  const statusMessage = `Dịch vụ chưa thể xử lý yêu cầu (HTTP ${response.status}).`;
  let data = {};

  if (contentType.includes("application/json")) {
    try {
      data = await response.json();
    } catch {
      // Keep the generic HTTP error; do not expose arbitrary server details.
    }
  }

  const detail = safeServerDetail(data);
  const error = new Error(statusMessage);
  error.telemetry = {
    http_method: method,
    http_path: safeRequestPath(url),
    http_status: response.status,
    http_response_detail: detail,
    request_id: response.headers.get("x-request-id") || data?.request_id,
    failure_stage: response.headers.get("x-failure-stage") || data?.failure_stage,
    failure_reason_code: response.headers.get("x-failure-reason-code") || data?.failure_reason_code,
    exception_type: response.headers.get("x-exception-type") || data?.exception_type,
    upstream_status_code: Number(response.headers.get("x-upstream-status-code")) || data?.upstream_status_code,
  };
  return error;
}

export async function getGatewayHealth({ baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const url = `${baseUrl.replace(/\/$/, "")}/health`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) throw await responseError(response, { method: "GET", url });
  return response.json();
}

export async function askLegalQuery(query, { allowExternal = false, domainHint = "", responseStyle = "default", signal, baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  if (!["default", "strict", "friendly"].includes(responseStyle)) {
    throw new Error("Choose a supported answer style.");
  }
  const requestBody = { query, allow_external: Boolean(allowExternal) };
  requestBody.response_style = responseStyle;
  if (typeof domainHint === "string" && domainHint.trim()) {
    requestBody.domain_hint = domainHint.trim();
  }
  const url = `${baseUrl.replace(/\/$/, "")}/api/v2/legal-query`;
  const response = await fetch(url, {
    method: "POST",
    credentials: "include",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) throw await responseError(response, { method: "POST", url });

  const result = await response.json();
  if (
    !result
    || typeof result !== "object"
    || !["SUCCESS", "FAILED"].includes(result.status)
    || (result.outcome !== undefined && ![
      "answered", "answered_unverified", "withheld", "needs_clarification", "verification_failed",
    ].includes(result.outcome))
    || !(typeof result.answer === "string" || result.answer === null)
    || typeof result.processing_time_ms !== "number"
    || !result.pipeline_execution
    || typeof result.pipeline_execution !== "object"
    || Array.isArray(result.pipeline_execution)
  ) {
    throw new Error("Gateway trả về dữ liệu không đúng định dạng.");
  }

  return result;
}

export async function getDomains({ baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const url = `${baseUrl.replace(/\/$/, "")}/api/v2/domains/`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) throw await responseError(response, { method: "GET", url });
  const domains = await response.json();
  if (!Array.isArray(domains) || domains.some((domain) => typeof domain !== "string")) {
    throw new Error("Domain API returned an invalid domain list.");
  }
  return domains;
}

export async function getUploadDomains({ baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const url = `${baseUrl.replace(/\/$/, "")}/api/v2/domains/upload-targets`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) throw await responseError(response, { method: "GET", url });
  const result = await response.json();
  const domains = result?.domains;
  if (
    !result
    || !Array.isArray(domains)
    || domains.some((domain) => (
      !domain
      || typeof domain.domain_id !== "string"
      || typeof domain.display_name !== "string"
      || typeof domain.display_name_vi !== "string"
      || typeof domain.enabled !== "boolean"
      || typeof domain.upload_enabled !== "boolean"
      || typeof domain.configured !== "boolean"
      || !Array.isArray(domain.allowed_extensions)
      || domain.allowed_extensions.some((extension) => ![".txt", ".md", ".xlsx"].includes(extension))
      || !["unknown", "available", "unavailable"].includes(domain.data_available)
      || !["unknown", "available", "unavailable"].includes(domain.retrieval_ready)
      || !["unknown", "available", "unavailable"].includes(domain.model_ready)
      || !["not_verified", "verified"].includes(domain.e2e_verified)
    ))
    || new Set(domains.map((domain) => domain.domain_id)).size !== domains.length
  ) {
    throw new Error("Domain upload API returned an invalid target schema.");
  }
  return domains;
}

const readinessStates = new Set(["unknown", "available", "unavailable"]);

export async function getDomainReadiness({ baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const url = `${baseUrl.replace(/\/$/, "")}/api/v2/domains/readiness`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) throw await responseError(response, { method: "GET", url });
  const result = await response.json();
  if (
    !result
    || !Array.isArray(result.domains)
    || result.domains.some((domain) => (
      !domain
      || typeof domain.domain_id !== "string"
      || typeof domain.domain_name !== "string"
      || typeof domain.display_name_vi !== "string"
      || typeof domain.registered !== "boolean"
      || typeof domain.enabled !== "boolean"
      || typeof domain.upload_enabled !== "boolean"
      || typeof domain.configured !== "boolean"
      || !readinessStates.has(domain.data_available)
      || !readinessStates.has(domain.retrieval_ready)
      || !readinessStates.has(domain.model_ready)
      || !["not_verified", "verified"].includes(domain.e2e_verified)
      || domain.response_policy !== "evidence_required"
      || !Array.isArray(domain.supported_extensions)
      || domain.supported_extensions.some((extension) => ![".txt", ".md", ".xlsx"].includes(extension))
      || typeof domain.reason_code !== "string"
    ))
  ) {
    throw new Error("Domain readiness API returned an invalid contract.");
  }
  return result;
}

export async function uploadDomainDocument(
  domainId,
  file,
  { allowedDomainIds, baseUrl = API_BASE_URL } = {},
) {
  assertGatewayConfigured(baseUrl);
  if (typeof domainId !== "string" || !domainId.trim() || !(file instanceof Blob)) {
    throw new Error("Choose a domain and a .xlsx, .txt, or .md file first.");
  }
  if (allowedDomainIds && !allowedDomainIds.includes(domainId)) {
    throw new Error("Choose a domain explicitly enabled for document uploads.");
  }
  const form = new FormData();
  form.append("file", file, file.name || "document.txt");
  const url = `${baseUrl.replace(/\/$/, "")}/api/v2/domains/${encodeURIComponent(domainId)}/documents`;
  const response = await fetch(url, { method: "POST", credentials: "include", body: form });
  if (!response.ok) throw await responseError(response, { method: "POST", url });
  const result = await response.json();
  if (!result || result.status !== "indexed" || result.domain_id !== domainId || !Number.isInteger(result.chunk_count)) {
    throw new Error("Domain document API returned an invalid indexing result.");
  }
  return result;
}
