import test from "node:test";
import assert from "node:assert/strict";
import {
  getAvailableUploadDomains,
  loadUploadDomainOptions,
  UPLOAD_DOMAIN_FALLBACK,
  UPLOAD_DOMAIN_ALLOWLIST,
} from "./domainOptions.js";

const requestedTargets = [
  "administrative_law",
  "tourism",
  "healthcare",
  "education",
  "shopping",
  "food",
];

function target(domainId, updates = {}) {
  return {
    domain_id: domainId,
    display_name: domainId,
    display_name_vi: domainId,
    enabled: true,
    upload_enabled: true,
    configured: true,
    data_available: "unknown",
    retrieval_ready: "unknown",
    model_ready: "unknown",
    e2e_verified: "not_verified",
    response_policy: "evidence_required",
    allowed_extensions: [".txt", ".md", ".xlsx"],
    ...updates,
  };
}

test("upload UI allowlist contains exactly the six requested target domains", () => {
  assert.deepEqual(UPLOAD_DOMAIN_ALLOWLIST, requestedTargets);
});

test("dropdown uses only Registry published targets and excludes legal/general domains", () => {
  const published = [
    target("legal_ai"),
    ...requestedTargets.map((domainId) => target(domainId)),
    target("general_assistant"),
  ];

  assert.deepEqual(
    getAvailableUploadDomains(published).map((domain) => domain.domain_id),
    requestedTargets,
  );
  assert.deepEqual(getAvailableUploadDomains([]), []);
});

test("unconfigured, disabled, or non-upload Registry domains never appear", () => {
  const published = requestedTargets.map((domainId) => target(domainId));
  published[2] = target("healthcare", { configured: false });
  published[3] = target("education", { enabled: false });
  published[4] = target("shopping", { upload_enabled: false });
  published[5] = target("food", { allowed_extensions: [] });

  assert.deepEqual(
    getAvailableUploadDomains(published).map((domain) => domain.domain_id),
    ["administrative_law", "tourism"],
  );
});

test("fallback keeps six targets selectable without inventing readiness state", () => {
  assert.deepEqual(
    UPLOAD_DOMAIN_FALLBACK.map((domain) => domain.domain_id),
    requestedTargets,
  );
  assert.equal("e2e_verified" in UPLOAD_DOMAIN_FALLBACK[1], false);
  assert.equal("enabled" in UPLOAD_DOMAIN_FALLBACK[1], false);
});

test("Registry failure selects the marked six-domain fallback and preserves the real error", async () => {
  const result = await loadUploadDomainOptions(async () => {
    throw new Error("Dịch vụ chưa thể xử lý yêu cầu (HTTP 404).");
  });
  assert.equal(result.source, "fallback");
  assert.equal(result.error, "Dịch vụ chưa thể xử lý yêu cầu (HTTP 404).");
  assert.deepEqual(result.domains, UPLOAD_DOMAIN_FALLBACK);
  assert.equal(result.domains.some((domain) => "e2e_verified" in domain), false);
});

test("Registry success remains the source and does not fall back for unknown readiness", async () => {
  const registryDomains = requestedTargets.map((domainId) => target(domainId));
  const result = await loadUploadDomainOptions(async () => registryDomains);
  assert.equal(result.source, "registry");
  assert.equal(result.error, "");
  assert.deepEqual(result.domains.map((domain) => domain.domain_id), requestedTargets);
  assert.equal(result.domains[1].e2e_verified, "not_verified");
});

test("Registry failure retains safe HTTP diagnostics for the details panel", async () => {
  const result = await loadUploadDomainOptions(async () => {
    const error = new Error("Registry HTTP 404");
    error.telemetry = {
      http_method: "GET",
      http_path: "/api/v2/domains/upload-targets",
      http_status: 404,
      http_response_detail: "Unknown route",
    };
    throw error;
  });
  assert.equal(result.source, "fallback");
  assert.deepEqual(result.diagnostic, {
    http_method: "GET",
    http_path: "/api/v2/domains/upload-targets",
    http_status: 404,
    http_response_detail: "Unknown route",
  });
});
