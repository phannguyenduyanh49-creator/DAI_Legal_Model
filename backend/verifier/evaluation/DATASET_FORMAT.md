# Evaluation dataset format

The evaluation runner consumes one JSON object per line. Candidate records from
the Catalog are exported with a blank `gold_label` and `review_status=unreviewed`.
The NLI model must never be used to fill the gold label.

```json
{
  "sample_id": "catalog-candidate-FORM-001-description",
  "domain": "legal_ai",
  "claim": "The verbatim claim text to review.",
  "evidence_id": "catalog:FORM-001",
  "evidence_text": "Exact Evidence text from the source record.",
  "gold_label": "",
  "review_status": "unreviewed",
  "source_record_id": "FORM-001",
  "source_field": "description",
  "source_url": "",
  "notes": "Candidate only; reviewer must validate source and context."
}
```

Required fields are `sample_id`, `domain`, `claim`, `evidence_id`,
`evidence_text`, `gold_label`, `review_status`, `source_record_id`, and
`notes`. `source_field` and `source_url` preserve Catalog provenance when
available. `sample_id` must be unique.

Allowed `gold_label` values are `supported`, `unsupported`, `contradicted`,
`partially_supported`, `insufficient_evidence`, or the empty string while unlabeled. Allowed review
states are `unreviewed`, `in_review`, `expert_confirmed`, and
`needs_adjudication`. Metrics use only valid labels with
`review_status=expert_confirmed`; all other rows are excluded from metrics.

Labels:

- `supported`: Evidence entails all material parts of the claim.
- `unsupported`: Evidence does not establish the claim but does not directly
  contradict it.
- `contradicted`: Evidence explicitly conflicts with the claim.
- `partially_supported`: Evidence establishes only part of a compound claim.
- `insufficient_evidence`: The supplied excerpt lacks relevant scope or
  context, so reviewers cannot determine whether the claim is supported or
  contradicted. This is distinct from completed evaluation with an unsupported
  claim.

If the source or relevant context cannot be checked, or reviewers cannot
adjudicate whether the excerpt is insufficient, leave the label blank and set
`needs_adjudication`. Use `insufficient_evidence` when reviewers can confirm
that the supplied excerpt omits relevant scope or context. The reviewer guide is
`REVIEW_GUIDE.md`. Candidate extraction is read-only and outputs unlabeled
records; it does not assert that Catalog text is authoritative or current law.

Use at least two qualified reviewers and adjudicate disagreements. Split by
source record, legal instrument/version, and time to prevent near-duplicate
leakage. Keep annotator identities in a separate protected manifest.

The bundled synthetic fixture is for contract tests only. It is not part of the
real benchmark and cannot support legal accuracy metrics.
