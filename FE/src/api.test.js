import test from "node:test";
import assert from "node:assert/strict";
import {
  askLegalQuery,
  getGatewayHealth,
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
    assert.equal(captured.options.credentials, undefined);
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
