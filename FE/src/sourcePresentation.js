import { safeSourceUrl } from "./answerPresentation.js";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function isRawSourceIdentifier(value) {
  if (typeof value !== "string") return true;
  const candidate = value.trim();
  if (!candidate) return true;
  return UUID.test(candidate)
    || /^domain:[^:\s]+:[^\s]+(?::\d+)?$/i.test(candidate)
    || /^(?:catalog|source|document|doc|evidence):[a-z0-9:_-]+$/i.test(candidate);
}

export function resolveCitationTitle(evidenceId, sources) {
  if (typeof evidenceId !== "string" || !Array.isArray(sources)) return null;
  const source = sources.find((item) => item && item.evidence_id === evidenceId);
  const title = typeof source?.title === "string" ? source.title.trim() : "";
  return title && !isRawSourceIdentifier(title) ? title : null;
}

export function visibleEvidenceSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.flatMap((source) => {
    if (!source || typeof source !== "object") return [];
    const rawTitle = typeof source.title === "string" ? source.title.trim() : "";
    const title = rawTitle && !isRawSourceIdentifier(rawTitle) ? rawTitle : "";
    const location = typeof source.location === "string" ? source.location.trim() : "";
    const sourceUrl = safeSourceUrl(source.source_url);
    if (!title && !sourceUrl) return [];
    return [{ title, location, sourceUrl }];
  });
}
