import test from "node:test";
import assert from "node:assert/strict";
import {
  isRawSourceIdentifier,
  resolveCitationTitle,
  visibleEvidenceSources,
} from "./sourcePresentation.js";

const tourismId = "domain:tourism:99d1a158-8cf2-4140-b036-bcffbc39199d:1";

test("citation ID resolves to an existing human title without exposing the ID", () => {
  const title = resolveCitationTitle(tourismId, [{ evidence_id: tourismId, title: "Cẩm nang du lịch" }]);
  assert.equal(title, "Cẩm nang du lịch");
  assert.equal(JSON.stringify(title).includes(tourismId), false);
});

test("citations without a readable source title are omitted", () => {
  assert.equal(resolveCitationTitle(tourismId, [{ evidence_id: tourismId, title: tourismId }]), null);
  assert.equal(resolveCitationTitle(tourismId, []), null);
});

test("raw source IDs are recognized across domains and title-only source cards remain", () => {
  for (const domain of ["legal_ai", "administrative_law", "tourism", "healthcare", "education", "shopping", "food"]) {
    assert.equal(isRawSourceIdentifier(`domain:${domain}:99d1a158-8cf2-4140-b036-bcffbc39199d:1`), true);
  }
  const visible = visibleEvidenceSources([
    { evidence_id: tourismId, source_id: tourismId, title: "Cẩm nang du lịch", location: "Chương 1" },
    { evidence_id: "catalog:89acbd18-4f93-4f18-b16d-a2db77365e37", title: "", source_id: "catalog:89acbd18-4f93-4f18-b16d-a2db77365e37" },
    { evidence_id: "doc:raw", title: "", source_url: "javascript:alert(1)" },
  ]);
  assert.deepEqual(visible, [{ title: "Cẩm nang du lịch", location: "Chương 1", sourceUrl: "" }]);
});
