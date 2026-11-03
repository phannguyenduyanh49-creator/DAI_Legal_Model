const DOMAIN_LABELS = Object.freeze({
  legal_ai: "Legal AI",
  administrative_law: "Administrative law",
  tourism: "Tourism and travel",
  shopping: "Shopping and product information",
  healthcare: "Healthcare",
  education: "Education",
  food: "Food and dining",
  other: "Other / general topics",
  general_assistant: "Other / general topics",
});
const DOMAIN_ORDER = [
  "legal_ai",
  "administrative_law",
  "tourism",
  "shopping",
  "healthcare",
  "education",
  "food",
  "other",
  "general_assistant",
];

export function domainOptionLabel(displayName, domainId) {
  return `${displayName || domainId} (${domainId})`;
}

/** Build chat options only from enabled IDs returned by the authoritative Registry API. */
export function getDomainChatOptions(domainIds, readiness = {}) {
  if (!Array.isArray(domainIds)) return [];
  const registered = new Set(domainIds.filter((domainId) => typeof domainId === "string" && domainId.trim()));
  return DOMAIN_ORDER.filter((domainId) => registered.has(domainId))
    .map((domainId) => {
      const status = readiness[domainId];
      const displayName = DOMAIN_LABELS[domainId] || status?.display_name_vi || status?.domain_name || domainId;
      return { domainId, displayName, readiness: status || null };
    });
}
