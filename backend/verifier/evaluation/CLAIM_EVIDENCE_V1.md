# Claim–Evidence v1 review queue

`claim_evidence_v1.jsonl` contains 80 **candidate_unreviewed** pairs exported
from the existing Catalog candidate file. It contains no approved legal labels.
The claim is source text copied from the Catalog, not an independently checked
legal proposition. `question`, `source_document_id`, `label`, `rationale`, and
`reviewer` are null where their values were not established by the export.

Validate the queue without loading a model or changing a database:

```powershell
.\.venv\Scripts\python.exe -m backend.verifier.evaluation.claim_dataset --validate backend/verifier/evaluation/claim_evidence_v1.jsonl
```

For review, each annotator should independently open the original Catalog
record identified by `source_metadata.source_record_id`, check the full source
and its date/version, write a specific question and atomic claim, then assign
one of `supported`, `contradicted`, `not_supported`, or
`insufficient_context`. Record the exact excerpt, source document identity,
reasoning, reviewer ID, and timestamp in separate review copies. An
adjudicator resolves disagreements before a row becomes gold data. Keep the
individual reviews and adjudication trail; do not overwrite this candidate
queue or automatically promote model suggestions. The existing
`REVIEW_GUIDE.md` provides detailed label definitions, but its older label
names differ (`unsupported`/`insufficient_evidence`); map these explicitly in
the reviewed-data import rather than silently treating them as identical.

The current validator checks duplicate IDs/pairs and source identity conflicts.
It is a structural check only. It does not query PostgreSQL, establish source
authenticity, or measure semantic support.
