import test from "node:test";
import assert from "node:assert/strict";
import { domainOptionLabel, getDomainChatOptions } from "./domainChatOptions.js";

test("chat options use requested labels while preserving authoritative IDs", () => {
  const ids = [
    "legal_ai", "administrative_law", "tourism", "shopping",
    "healthcare", "education", "food", "general_assistant",
  ];
  const options = getDomainChatOptions(ids, {
    tourism: { e2e_verified: "not_verified" },
  });

  assert.deepEqual(options.map((item) => item.domainId), ids);
  assert.deepEqual(options.map((item) => item.displayName), [
    "Legal AI",
    "Administrative law",
    "Tourism and travel",
    "Shopping and product information",
    "Healthcare",
    "Education",
    "Food and dining",
    "Other / general topics",
  ]);
});

test("Other is not invented when general_assistant is absent from the Registry", () => {
  assert.deepEqual(
    getDomainChatOptions(["tourism", "general_assistant", "tourism"]).map((item) => item.domainId),
    ["tourism", "general_assistant"],
  );
  assert.deepEqual(getDomainChatOptions(["tourism"]).map((item) => item.domainId), ["tourism"]);
  assert.deepEqual(getDomainChatOptions(["legal_ai", "unrequested_domain"]).map((item) => item.domainId), ["legal_ai"]);
});

test("domain dropdown labels show friendly names and IDs without unverified suffixes", () => {
  assert.equal(domainOptionLabel("Du lịch", "tourism"), "Du lịch (tourism)");
  assert.equal(domainOptionLabel("Legal AI", "legal_ai").includes("xác minh"), false);
});
