import test from "node:test";
import assert from "node:assert/strict";
import {
  askLegalQuery,
  getDomains,
  getDomainReadiness,
  getGatewayHealth,
  getUploadDomains,
  uploadDomainDocument,
} from "./api.js";

const TEST_GATEWAY_URL = "http://127.0.0.1:8002";
const uploadTargetIds = [
  "administrative_law", "tourism", "healthcare", "education", "shopping", "food",
];
const uploadTarget = (domain_id) => ({
  domain_id,
  display_name: domain_id,
  display_name_vi: domain_id,
  enabled: true,
  upload_enabled: true,
  configured: true,
  allowed_extensions: [".txt", ".md", ".xlsx"],
  data_available: "unknown",
  retrieval_ready: "unknown",
  model_ready: "unknown",
  e2e_verified: "not_verified",
});

test("legal query uses the Gateway JSON contract and preserves FAILED response", async () => {
  const originalFetch = globalThis.fetch;
  let captured;
  const payload = {
    request_id: "trace_legal_test",
    timestamp: "2026-10-19T00:00:00Z",
    processing_time_ms: 1240.5,
    status: "FAILED",
    query_analysis: {},
    routing_and_planning: { selected_domain: "enterprise_law", execution_plan: [] },
    next_step: "PIPELINE_COMPLETED",
    domain_policy: {},
    answer: "Fallback do Gateway",
    pipeline_execution: { verifier: "failed", evidence_ids: ["rag:case-1"] },
  };
  globalThis.fetch = async (url, options) => {
    captured = { url, options };
    return Response.json(payload);
  };

  try {
    const result = await askLegalQuery("Câu hỏi pháp lý", { allowExternal: true, baseUrl: TEST_GATEWAY_URL });
    assert.equal(captured.url, `${TEST_GATEWAY_URL}/api/v2/legal-query`);
    assert.equal(captured.options.method, "POST");
    assert.equal(captured.options.credentials, "include");
    assert.deepEqual(JSON.parse(captured.options.body), {
      query: "Câu hỏi pháp lý",
      allow_external: true,
      response_style: "default",
    });
    assert.deepEqual(result, payload);
    assert.equal(result.status, "FAILED");
    assert.equal(result.pipeline_execution.verifier, "failed");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("domain hint is sent only when the caller explicitly selects a domain", async () => {
  const originalFetch = globalThis.fetch;
  let body;
  const payload = {
    status: "SUCCESS", outcome: "answered", answer: "ok", processing_time_ms: 2,
    pipeline_execution: {},
  };
  globalThis.fetch = async (_url, options) => {
    body = JSON.parse(options.body);
    return Response.json(payload);
  };
  try {
    await askLegalQuery("question", { domainHint: "tourism", baseUrl: TEST_GATEWAY_URL });
    assert.equal(body.domain_hint, "tourism");
    await askLegalQuery("question", { baseUrl: TEST_GATEWAY_URL });
    assert.equal(Object.hasOwn(body, "domain_hint"), false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("every configured chat domain ID is sent unchanged and general_assistant represents Khác", async () => {
  const originalFetch = globalThis.fetch;
  const domainIds = [
    "legal_ai", "administrative_law", "tourism", "shopping",
    "healthcare", "education", "food", "general_assistant",
  ];
  const sent = [];
  globalThis.fetch = async (_url, options) => {
    sent.push(JSON.parse(options.body).domain_hint);
    return Response.json({
      status: "FAILED", outcome: "withheld", answer: "No Evidence",
      processing_time_ms: 1, pipeline_execution: {},
    });
  };
  try {
    for (const domainId of domainIds) {
      await askLegalQuery("question", { domainHint: domainId, baseUrl: TEST_GATEWAY_URL });
    }
    assert.deepEqual(sent, domainIds);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("friendly response style is sent separately from the selected domain", async () => {
  const originalFetch = globalThis.fetch;
  let body;
  globalThis.fetch = async (_url, options) => {
    body = JSON.parse(options.body);
    return Response.json({
      status: "SUCCESS", outcome: "answered", answer: "ok", processing_time_ms: 1,
      pipeline_execution: {},
    });
  };
  try {
    await askLegalQuery("question", {
      domainHint: "shopping",
      responseStyle: "friendly",
      baseUrl: TEST_GATEWAY_URL,
    });
    assert.equal(body.domain_hint, "shopping");
    assert.equal(body.response_style, "friendly");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("domain list and document upload use the Gateway domain API", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/upload-targets")) return Response.json({
      domains: [uploadTarget("legal_ai"), ...uploadTargetIds.map(uploadTarget), uploadTarget("general_assistant")],
    });
    if (options.method === "GET") return Response.json(["legal_ai", "tourism"]);
    return Response.json({ status: "indexed", domain_id: "tourism", title: "route.md", chunk_count: 1 });
  };
  try {
    assert.deepEqual(await getDomains({ baseUrl: TEST_GATEWAY_URL }), ["legal_ai", "tourism"]);
    assert.deepEqual(await getUploadDomains({ baseUrl: TEST_GATEWAY_URL }), [
      uploadTarget("legal_ai"), ...uploadTargetIds.map(uploadTarget), uploadTarget("general_assistant"),
    ]);
    const file = new File(["Pier 9 boards at 06:47."], "route.md", { type: "text/markdown" });
    assert.equal((await uploadDomainDocument("tourism", file, {
      allowedDomainIds: ["administrative_law", "tourism", "healthcare", "education", "shopping", "food"],
      baseUrl: TEST_GATEWAY_URL,
    })).chunk_count, 1);
    assert.equal(calls[0].url, `${TEST_GATEWAY_URL}/api/v2/domains/`);
    assert.equal(calls[1].url, `${TEST_GATEWAY_URL}/api/v2/domains/upload-targets`);
    assert.equal(calls[2].url, `${TEST_GATEWAY_URL}/api/v2/domains/tourism/documents`);
    assert.equal(calls[2].options.credentials, "include");
    assert.equal(calls[2].options.body.get("file").name, "route.md");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("shopping XLSX upload uses the Gateway multipart route and requires indexed confirmation", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  const file = new File(["workbook"], "Mua_Sam_Sai_Gon_100_San_Pham.xlsx", {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return Response.json({ status: "indexed", domain_id: "shopping", title: file.name, chunk_count: 4 });
  };
  try {
    const result = await uploadDomainDocument("shopping", file, {
      allowedDomainIds: uploadTargetIds,
      baseUrl: TEST_GATEWAY_URL,
    });
    assert.equal(result.status, "indexed");
    assert.equal(calls[0].url, `${TEST_GATEWAY_URL}/api/v2/domains/shopping/documents`);
    assert.equal(calls[0].options.method, "POST");
    assert.equal(calls[0].options.body.get("file").name, file.name);
    assert.equal(calls[0].options.body.get("domain_id"), null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("upload errors retain the actual safe server detail in diagnostics", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: "Unknown domain" }),
    { status: 404, headers: { "content-type": "application/json" } },
  );
  try {
    await assert.rejects(
      getUploadDomains({ baseUrl: TEST_GATEWAY_URL }),
      (error) => {
        assert.equal(error.message.includes("HTTP 404"), true);
        assert.equal(error.telemetry.http_method, "GET");
        assert.equal(error.telemetry.http_path, "/api/v2/domains/upload-targets");
        assert.equal(error.telemetry.http_response_detail, "Unknown domain");
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("automatic chat selection is never an upload destination", async () => {
  const originalFetch = globalThis.fetch;
  let called = false;
  globalThis.fetch = async () => { called = true; return Response.json({}); };
  try {
    const file = new File(["sample"], "note.txt", { type: "text/plain" });
    await assert.rejects(
      uploadDomainDocument("", file, { allowedDomainIds: ["tourism"], baseUrl: TEST_GATEWAY_URL }),
      /Choose a domain/,
    );
    await assert.rejects(
      uploadDomainDocument("automatic", file, { allowedDomainIds: ["tourism"], baseUrl: TEST_GATEWAY_URL }),
      /enabled for document uploads/,
    );
    assert.equal(called, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("failed Tourism indexing reports the HTTP error and is never treated as success", async () => {
  const originalFetch = globalThis.fetch;
  const file = new File(["Tourism note"], "route.txt", { type: "text/plain" });
  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: "Qdrant unavailable" }),
    { status: 503, headers: { "content-type": "application/json" } },
  );
  try {
    await assert.rejects(
      uploadDomainDocument("tourism", file, {
        allowedDomainIds: uploadTargetIds,
        baseUrl: TEST_GATEWAY_URL,
      }),
      (error) => error.message.includes("HTTP 503"),
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("domain readiness contract keeps runtime and E2E state explicitly unverified", async () => {
  const originalFetch = globalThis.fetch;
  const payload = { domains: [{
    domain_id: "legal_ai", domain_name: "Legal AI", display_name_vi: "Pháp lý AI",
    registered: true, enabled: true, upload_enabled: true, configured: true,
    data_available: "unknown", retrieval_ready: "unknown", model_ready: "unknown",
    e2e_verified: "not_verified", response_policy: "evidence_required",
    supported_extensions: [".txt", ".md", ".xlsx"],
    reason_code: "RUNTIME_DEPENDENCIES_NOT_PROBED", reason: "Dependencies were not probed.",
  }] };
  let url;
  globalThis.fetch = async (value) => { url = value; return Response.json(payload); };
  try {
    assert.deepEqual(await getDomainReadiness({ baseUrl: TEST_GATEWAY_URL }), payload);
    assert.equal(url, `${TEST_GATEWAY_URL}/api/v2/domains/readiness`);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("domain readiness rejects a response with an unknown response policy", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ domains: [{
    domain_id: "legal_ai", domain_name: "Legal AI", display_name_vi: "Pháp lý AI",
    registered: true, enabled: true, upload_enabled: true, configured: true,
    data_available: "unknown", retrieval_ready: "unknown", model_ready: "unknown",
    e2e_verified: "not_verified", response_policy: "optional",
    supported_extensions: [".txt", ".md"],
    reason_code: "RUNTIME_DEPENDENCIES_NOT_PROBED",
  }] });
  try {
    await assert.rejects(getDomainReadiness({ baseUrl: TEST_GATEWAY_URL }), /invalid contract/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("legal query preserves business outcome separately from HTTP success", async () => {
  const originalFetch = globalThis.fetch;
  const payload = {
    request_id: "trace_legal_outcome",
    timestamp: "2026-10-25T00:00:00Z",
    processing_time_ms: 1200,
    status: "FAILED",
    outcome: "verification_failed",
    query_analysis: {},
    routing_and_planning: {},
    next_step: "PIPELINE_COMPLETED",
    domain_policy: {},
    answer: "Withheld",
    pipeline_execution: { verification_status: "not_verified" },
  };
  globalThis.fetch = async () => Response.json(payload, { status: 200 });
  try {
    assert.deepEqual(
      await askLegalQuery("test", { baseUrl: TEST_GATEWAY_URL }),
      payload,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("legal query accepts answered_unverified while preserving legacy FAILED status", async () => {
  const originalFetch = globalThis.fetch;
  const payload = {
    request_id: "trace_legal_reference",
    timestamp: "2026-10-25T00:00:00Z",
    processing_time_ms: 1200,
    status: "FAILED",
    outcome: "answered_unverified",
    query_analysis: {},
    routing_and_planning: {},
    next_step: "PIPELINE_COMPLETED",
    domain_policy: {},
    answer: "Reference answer [Evidence ID: rag:1]",
    pipeline_execution: {
      verification_status: "not_verified",
      claim_grounding_status: "not_evaluated",
      confidence_status: "not_verified",
      answer_withheld: false,
    },
  };
  globalThis.fetch = async () => Response.json(payload, { status: 200 });
  try {
    assert.deepEqual(
      await askLegalQuery("test", { baseUrl: TEST_GATEWAY_URL }),
      payload,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("health check calls only the Gateway GET health endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let captured;
  globalThis.fetch = async (url, options) => {
    captured = { url, options };
    return Response.json({ status: "HEALTHY" });
  };

  try {
    assert.deepEqual(await getGatewayHealth({ baseUrl: TEST_GATEWAY_URL }), { status: "HEALTHY" });
    assert.equal(captured.url, `${TEST_GATEWAY_URL}/health`);
    assert.equal(captured.options.method, "GET");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Gateway HTTP errors preserve safe server detail and request diagnostics", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: "Gateway unavailable" }),
    { status: 502, headers: { "content-type": "application/json" } },
  );

  try {
    await assert.rejects(
      askLegalQuery("test", { baseUrl: TEST_GATEWAY_URL }),
      { message: "Dịch vụ chưa thể xử lý yêu cầu (HTTP 502)." },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Gateway HTTP error preserves safe trace telemetry from response headers", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: "Core API is unavailable" }),
    {
      status: 502,
      headers: {
        "content-type": "application/json",
        "x-request-id": "trace_legal_0123456789abcdef0123456789abcdef",
        "x-failure-stage": "legacy_api",
        "x-failure-reason-code": "UPSTREAM_TIMEOUT",
        "x-exception-type": "ReadTimeout",
      },
    },
  );

  try {
    await assert.rejects(
      askLegalQuery("test", { baseUrl: TEST_GATEWAY_URL }),
      (error) => {
        assert.equal(error.message, "Dịch vụ chưa thể xử lý yêu cầu (HTTP 502)." );
        assert.equal(error.telemetry.request_id, "trace_legal_0123456789abcdef0123456789abcdef");
        assert.equal(error.telemetry.http_method, "POST");
        assert.equal(error.telemetry.http_path, "/api/v2/legal-query");
        assert.equal(error.telemetry.http_status, 502);
        assert.equal(error.telemetry.http_response_detail, "Core API is unavailable");
        assert.equal(error.telemetry.failure_stage, "legacy_api");
        assert.equal(error.telemetry.failure_reason_code, "UPSTREAM_TIMEOUT");
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("sensitive upstream error detail is not exposed to the UI", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(
    JSON.stringify({ detail: "Traceback File \"D:\\\\internal\\\\server.py\", line 12" }),
    { status: 500, headers: { "content-type": "application/json" } },
  );

  try {
    await assert.rejects(
      askLegalQuery("test", { baseUrl: TEST_GATEWAY_URL }),
      (error) => {
        assert.notEqual(error.message.includes("D:\\internal"), true);
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("malformed Gateway success payload is rejected instead of shown as an empty answer", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ status: "OK", answer: "text" });

  try {
    await assert.rejects(
      askLegalQuery("test", { baseUrl: TEST_GATEWAY_URL }),
      { message: "Gateway trả về dữ liệu không đúng định dạng." },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("missing Gateway configuration fails with a clear message", async () => {
  await assert.rejects(askLegalQuery("test", { baseUrl: "" }), {
    message: "Chưa cấu hình VITE_GATEWAY_API_URL cho môi trường này.",
  });
});
