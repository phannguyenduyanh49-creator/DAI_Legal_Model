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

async function responseError(response) {
  const contentType = response.headers.get("content-type") || "";
  const message = `Dịch vụ chưa thể xử lý yêu cầu (HTTP ${response.status}).`;
  let data = {};

  if (contentType.includes("application/json")) {
    try {
      data = await response.json();
    } catch {
      // Keep the generic HTTP error; do not expose arbitrary server details.
    }
  }

  const error = new Error(message);
  error.telemetry = {
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
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/health`, { method: "GET" });
  if (!response.ok) throw await responseError(response);
  return response.json();
}

export async function askLegalQuery(query, { allowExternal = false, domainHint = "", signal, baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const requestBody = { query, allow_external: Boolean(allowExternal) };
  if (typeof domainHint === "string" && domainHint.trim()) {
    requestBody.domain_hint = domainHint.trim();
  }
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/api/v2/legal-query`, {
    method: "POST",
    credentials: "include",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) throw await responseError(response);

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
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/api/v2/domains/`, { method: "GET" });
  if (!response.ok) throw await responseError(response);
  const domains = await response.json();
  if (!Array.isArray(domains) || domains.some((domain) => typeof domain !== "string")) {
    throw new Error("Domain API returned an invalid domain list.");
  }
  return domains;
}

const readinessStates = new Set(["unknown", "available", "unavailable"]);

export async function getDomainReadiness({ baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const response = await fetch(
    `${baseUrl.replace(/\/$/, "")}/api/v2/domains/readiness`,
    { method: "GET" },
  );
  if (!response.ok) throw await responseError(response);
  const result = await response.json();
  if (
    !result
    || !Array.isArray(result.domains)
    || result.domains.some((domain) => (
      !domain
      || typeof domain.domain_id !== "string"
      || typeof domain.registered !== "boolean"
      || typeof domain.enabled !== "boolean"
      || typeof domain.configured !== "boolean"
      || !readinessStates.has(domain.data_available)
      || !readinessStates.has(domain.retrieval_ready)
      || !readinessStates.has(domain.model_ready)
      || !["not_verified", "verified"].includes(domain.e2e_verified)
      || domain.response_policy !== "evidence_required"
      || typeof domain.reason_code !== "string"
    ))
  ) {
    throw new Error("Domain readiness API returned an invalid contract.");
  }
  return result;
}

export async function uploadDomainDocument(domainId, file, { baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  if (typeof domainId !== "string" || !domainId.trim() || !(file instanceof Blob)) {
    throw new Error("Choose a domain and a .txt or .md file first.");
  }
  const form = new FormData();
  form.append("file", file, file.name || "document.txt");
  const response = await fetch(
    `${baseUrl.replace(/\/$/, "")}/api/v2/domains/${encodeURIComponent(domainId)}/documents`,
    { method: "POST", credentials: "include", body: form },
  );
  if (!response.ok) throw await responseError(response);
  const result = await response.json();
  if (!result || result.status !== "indexed" || result.domain_id !== domainId || !Number.isInteger(result.chunk_count)) {
    throw new Error("Domain document API returned an invalid indexing result.");
  }
  return result;
}
