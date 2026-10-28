import test from "node:test";
import assert from "node:assert/strict";
import {
  askLegalQuery,
  getDomains,
  getDomainReadiness,
  getGatewayHealth,
  uploadDomainDocument,
} from "./api.js";

const TEST_GATEWAY_URL = "http://127.0.0.1:8002";

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

test("domain list and document upload use the Gateway domain API", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (options.method === "GET") return Response.json(["legal_ai", "tourism"]);
    return Response.json({ status: "indexed", domain_id: "tourism", title: "route.md", chunk_count: 1 });
  };
  try {
    assert.deepEqual(await getDomains({ baseUrl: TEST_GATEWAY_URL }), ["legal_ai", "tourism"]);
    const file = new File(["Pier 9 boards at 06:47."], "route.md", { type: "text/markdown" });
    assert.equal((await uploadDomainDocument("tourism", file, { baseUrl: TEST_GATEWAY_URL })).chunk_count, 1);
    assert.equal(calls[0].url, `${TEST_GATEWAY_URL}/api/v2/domains/`);
    assert.equal(calls[1].url, `${TEST_GATEWAY_URL}/api/v2/domains/tourism/documents`);
    assert.equal(calls[1].options.credentials, "include");
    assert.equal(calls[1].options.body.get("file").name, "route.md");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("domain readiness contract keeps runtime and E2E state explicitly unverified", async () => {
  const originalFetch = globalThis.fetch;
  const payload = { domains: [{
    domain_id: "legal_ai", registered: true, enabled: true, configured: true,
    data_available: "unknown", retrieval_ready: "unknown", model_ready: "unknown",
    e2e_verified: "not_verified", response_policy: "evidence_required",
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
    domain_id: "legal_ai", registered: true, enabled: true, configured: true,
    data_available: "unknown", retrieval_ready: "unknown", model_ready: "unknown",
    e2e_verified: "not_verified", response_policy: "optional",
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

test("Gateway HTTP errors use a generic message without exposing arbitrary API detail", async () => {
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
