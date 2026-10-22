import test from "node:test";
import assert from "node:assert/strict";
import { pipelineTelemetryRows, safeFailureTelemetry } from "./telemetry.js";

test("telemetry displays allowlisted failure reason, request id, statuses and durations", () => {
  const rows = pipelineTelemetryRows({
    request_id: "trace_legal_0123456789abcdef0123456789abcdef",
    failure_stage: "model_api",
    failure_reason_code: "UPSTREAM_TIMEOUT",
    retrieval: "completed",
    model_api: "error",
    verifier: "error",
    stage_durations_ms: { retrieval: 12.5, model_api: 90000 },
  });
  assert.deepEqual(rows.slice(0, 3), [
    ["Request ID", "trace_legal_0123456789abcdef0123456789abcdef"],
    ["Stage thất bại", "model_api"],
    ["Mã nguyên nhân", "UPSTREAM_TIMEOUT"],
  ]);
  assert.ok(rows.some(([label, value]) => label === "Thời gian Retrieval" && value === "12.5 ms"));
});

test("untrusted telemetry strings and unsupported fields are not displayed", () => {
  const rows = pipelineTelemetryRows({
    request_id: "user@example.com",
    failure_stage: "C:\\private\\stack.py",
    failure_reason_code: "raw question text",
    exception_type: "C:\\secret\\file.py",
    retrieval: "private query",
    stage_durations_ms: { private_path: 42 },
  });
  assert.deepEqual(rows, []);
});

test("older response without optional telemetry produces no failure rows", () => {
  assert.deepEqual(safeFailureTelemetry({}), []);
  assert.deepEqual(pipelineTelemetryRows({ verifier: "passed" }), [["Verifier", "passed"]]);
});
