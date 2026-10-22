const localHostnames = new Set(["localhost", "127.0.0.1"]);
const configuredGatewayUrl = import.meta.env?.VITE_GATEWAY_API_URL;
const defaultGatewayUrl =
  typeof window !== "undefined" && localHostnames.has(window.location.hostname)
    ? "http://127.0.0.1:8002"
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

export async function askLegalQuery(query, { allowExternal = false, signal, baseUrl = API_BASE_URL } = {}) {
  assertGatewayConfigured(baseUrl);
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/api/v2/legal-query`, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, allow_external: Boolean(allowExternal) }),
  });

  if (!response.ok) throw await responseError(response);

  const result = await response.json();
  if (
    !result
    || typeof result !== "object"
    || !["SUCCESS", "FAILED"].includes(result.status)
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
