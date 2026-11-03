export const UPLOAD_DOMAIN_ALLOWLIST = Object.freeze([
  "administrative_law",
  "tourism",
  "healthcare",
  "education",
  "shopping",
  "food",
]);

export const UPLOAD_DOMAIN_FALLBACK = Object.freeze([
  { domain_id: "administrative_law", display_name_vi: "Pháp luật hành chính", allowed_extensions: [".txt", ".md", ".xlsx"] },
  { domain_id: "tourism", display_name_vi: "Du lịch", allowed_extensions: [".txt", ".md", ".xlsx"] },
  { domain_id: "healthcare", display_name_vi: "Y tế", allowed_extensions: [".txt", ".md", ".xlsx"] },
  { domain_id: "education", display_name_vi: "Giáo dục", allowed_extensions: [".txt", ".md", ".xlsx"] },
  { domain_id: "shopping", display_name_vi: "Mua sắm", allowed_extensions: [".txt", ".md", ".xlsx"] },
  { domain_id: "food", display_name_vi: "Ẩm thực", allowed_extensions: [".txt", ".md", ".xlsx"] },
]);

/** Keep only published, enabled upload targets; runtime readiness is independent. */
export function getAvailableUploadDomains(registryDomains) {
  if (!Array.isArray(registryDomains)) return [];
  const byId = new Map(registryDomains.map((domain) => [domain.domain_id, domain]));

  return UPLOAD_DOMAIN_ALLOWLIST.flatMap((domainId) => {
    const domain = byId.get(domainId);
    if (
      !domain
      || domain.enabled !== true
      || domain.upload_enabled !== true
      || domain.configured !== true
      || !Array.isArray(domain.allowed_extensions)
      || domain.allowed_extensions.length === 0
    ) return [];
    return [domain];
  });
}

export async function loadUploadDomainOptions(loadFromRegistry) {
  try {
    const registryDomains = await loadFromRegistry();
    return {
      domains: getAvailableUploadDomains(registryDomains),
      source: "registry",
      error: "",
    };
  } catch (error) {
    return {
      domains: UPLOAD_DOMAIN_FALLBACK,
      source: "fallback",
      error: error instanceof Error ? error.message : "Registry request failed",
      diagnostic: error?.telemetry || null,
    };
  }
}
