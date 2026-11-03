const localHostnames = new Set(["localhost", "127.0.0.1"]);

const configuredGatewayUrl = import.meta.env?.VITE_GATEWAY_API_URL;

const defaultGatewayUrl =
  typeof window !== "undefined" &&
  localHostnames.has(window.location.hostname)
    ? `http://${window.location.hostname}:8000`
    : "";

export const API_BASE_URL = (
  configuredGatewayUrl || defaultGatewayUrl
).replace(/\/$/, "");

export const LEGAL_QUERY_URL =
  `${API_BASE_URL}/api/v1/chat/completions`;

export const GATEWAY_HEALTH_URL = `${API_BASE_URL}/`;

function assertGatewayConfigured(baseUrl) {
  if (!baseUrl) {
    throw new Error(
      "Chưa cấu hình VITE_GATEWAY_API_URL cho môi trường này."
    );
  }
}

async function responseError(response) {
  const contentType =
    response.headers.get("content-type") || "";

  let detail = "";

  if (contentType.includes("application/json")) {
    try {
      const data = await response.json();

      detail =
        typeof data?.detail === "string"
          ? ` ${data.detail}`
          : "";
    } catch {
      // Giữ thông báo lỗi HTTP mặc định.
    }
  }

  const error = new Error(
    `Dịch vụ chưa thể xử lý yêu cầu (HTTP ${response.status}).${detail}`
  );

  error.telemetry = {
    request_id: response.headers.get("x-request-id"),
    failure_stage: response.headers.get("x-failure-stage"),
    failure_reason_code: response.headers.get(
      "x-failure-reason-code"
    ),
    exception_type: response.headers.get("x-exception-type"),
    upstream_status_code:
      Number(
        response.headers.get("x-upstream-status-code")
      ) || undefined,
  };

  return error;
}

export async function getGatewayHealth({
  baseUrl = API_BASE_URL,
} = {}) {
  assertGatewayConfigured(baseUrl);

  const response = await fetch(
    `${baseUrl.replace(/\/$/, "")}/`,
    { method: "GET" }
  );

  if (!response.ok) {
    throw await responseError(response);
  }

  const data = await response.json();

  return {
    status: data?.success ? "HEALTHY" : "UNHEALTHY",
    ...data,
  };
}

async function parseCoreStream(response) {
  const contentType =
    response.headers.get("content-type") || "";

  if (!contentType.includes("text/event-stream")) {
    const result = await response.json();
    return normalizeGatewayResult(result);
  }

  const raw = await response.text();

  let answer = "";
  let done = {};

  for (const block of raw.split(/\r?\n\r?\n/)) {
    if (!block.trim()) continue;

    let eventName = "";
    const dataLines = [];

    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (!dataLines.length) continue;

    let payload;

    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      continue;
    }

    if (
      eventName === "chunk" &&
      typeof payload?.delta === "string"
    ) {
      answer += payload.delta;
    }

    if (eventName === "done") {
      done = payload || {};
    }
  }

  const pipeline =
    done.pipeline && typeof done.pipeline === "object"
      ? done.pipeline
      : {};

  const outcome =
    done.outcome ||
    pipeline.response_outcome ||
    "answered_unverified";

  const success = done.status === "done";

  return {
    status: success ? "SUCCESS" : "FAILED",
    outcome,
    answer: answer || null,
    processing_time_ms: Number(
      pipeline.processing_time_ms ||
      pipeline.total_duration_ms ||
      0
    ),
    pipeline_execution: pipeline,
    session_id: done.session_id,
    verifier: done.verifier,
    verifier_scope: done.verifier_scope,
  };
}

function normalizeGatewayResult(result) {
  if (
    result &&
    typeof result === "object" &&
    ["SUCCESS", "FAILED"].includes(result.status) &&
    (
      typeof result.answer === "string" ||
      result.answer === null
    ) &&
    typeof result.processing_time_ms === "number" &&
    result.pipeline_execution &&
    typeof result.pipeline_execution === "object"
  ) {
    return result;
  }

  return {
    status: result?.success === false ? "FAILED" : "SUCCESS",
    outcome: result?.outcome || "answered_unverified",
    answer:
      typeof result?.answer === "string"
        ? result.answer
        : null,
    processing_time_ms: Number(
      result?.processing_time_ms || 0
    ),
    pipeline_execution:
      result?.pipeline_execution ||
      result?.pipeline ||
      {},
    ...result,
  };
}

export async function askLegalQuery(
  query,
  {
    allowExternal = false,
    domainHint = "",
    signal,
    baseUrl = API_BASE_URL,
  } = {}
) {
  assertGatewayConfigured(baseUrl);

  const prompt = String(query || "").trim();

  if (!prompt) {
    throw new Error("Vui lòng nhập câu hỏi pháp lý.");
  }

  const requestBody = {
    prompt,
    allow_external: Boolean(allowExternal),
  };

  if (
    typeof domainHint === "string" &&
    domainHint.trim()
  ) {
    requestBody.policy_domain = domainHint.trim();
  }

  const response = await fetch(
    `${baseUrl.replace(/\/$/, "")}/api/v1/chat/completions`,
    {
      method: "POST",
      credentials: "include",
      signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream, application/json",
      },
      body: JSON.stringify(requestBody),
    }
  );

  if (!response.ok) {
    throw await responseError(response);
  }

  return parseCoreStream(response);
}

export async function getDomains({
  baseUrl = API_BASE_URL,
} = {}) {
  assertGatewayConfigured(baseUrl);

  // Core API hiện tại chưa có endpoint liệt kê domains v2.
  return ["legal_ai"];
}

export async function getDomainReadiness({
  baseUrl = API_BASE_URL,
} = {}) {
  assertGatewayConfigured(baseUrl);

  const response = await fetch(
    `${baseUrl.replace(/\/$/, "")}/`,
    { method: "GET" }
  );

  if (!response.ok) {
    throw await responseError(response);
  }

  const health = await response.json();
  const available = Boolean(health?.success);

  return {
    domains: [
      {
        domain_id: "legal_ai",
        registered: available,
        enabled: available,
        configured: available,
        data_available: available
          ? "available"
          : "unavailable",
        retrieval_ready: available
          ? "available"
          : "unavailable",
        model_ready: available
          ? "available"
          : "unavailable",
        e2e_verified: "not_verified",
        response_policy: "evidence_required",
        reason_code: available
          ? "CORE_API_RESPONDING"
          : "CORE_API_UNAVAILABLE",
      },
    ],
  };
}

export async function uploadDomainDocument(
  domainId,
  file,
  { baseUrl = API_BASE_URL } = {}
) {
  assertGatewayConfigured(baseUrl);

  if (
    typeof domainId !== "string" ||
    !domainId.trim() ||
    !(file instanceof Blob)
  ) {
    throw new Error(
      "Choose a domain and a .txt or .md file first."
    );
  }

  const form = new FormData();

  form.append(
    "file",
    file,
    file.name || "document.txt"
  );

  const response = await fetch(
    `${baseUrl.replace(/\/$/, "")}/api/v1/domains/${encodeURIComponent(domainId)}/documents`,
    {
      method: "POST",
      credentials: "include",
      body: form,
    }
  );

  if (!response.ok) {
    throw await responseError(response);
  }

  const result = await response.json();

  if (
    result?.status === "indexed" &&
    result.domain_id === domainId &&
    Number.isInteger(result.chunk_count)
  ) {
    return result;
  }

  if (
    result?.success === true ||
    result?.domain_id === domainId
  ) {
    return {
      status: "indexed",
      domain_id: domainId,
      chunk_count: Number.isInteger(result.chunk_count)
        ? result.chunk_count
        : 0,
      ...result,
    };
  }

  throw new Error(
    "Core API trả về kết quả upload tài liệu không đúng định dạng."
  );
}