# HAKIM — Clinical terminology and differential-support architecture

Status: **design specification**, not a validated diagnostic engine. The existing small synonym list is a temporary search convenience, not a clinical knowledge base.

## Non-negotiable requirements
- No artificial cap on diseases, symptoms, phenotypes, or synonyms. Coverage depends on upstream licensed/available terminology sources, version and scope; **never claim 100% completeness**.
- Never treat keyword matching or symptom overlap as a confirmed diagnosis. Human clinician review and history/examination/testing remain necessary.
- Distinguish **symptom/phenotype**, **finding**, **disease**, **drug**, **botanical taxon**, **procedure**, and **research topic**. Avoid automatically converting a symptom to a specific disease.
- Keep the original query and language; show normalized/translated terms, source, version and confidence separately. Let the user edit the search.
- Preserve evidence provenance, date, original IDs and source links; show unsupported/unmapped queries rather than guessing.
- Separate research discovery from clinical decision support. Explicitly mark emergency warning signs; never promise safe triage or diagnosis without clinical validation.
- Protect health data: data minimization, informed consent, access controls, retention/deletion, encryption and audit logs.

## Data source roles (verify access and terms before use)
| Source | Purpose | Constraint |
|---|---|---|
| MeSH (NLM) | Biomedical search concepts, synonyms and PubMed indexing | Versioned vocabulary, not a complete symptom-disease diagnostic graph |
| HPO | Phenotypes and phenotype-to-disease annotations | Associations vary by disease and evidence; absent annotation is not proof of absence |
| ICD-11 (WHO) | Disease classification/coding | Classification, not diagnostic decision engine; API/license conditions |
| SNOMED CT | Rich clinical terminology and relationships | Territorial/member licensing and distribution conditions; no assumption of unrestricted access |
| UMLS | Mapping/normalization across vocabularies | UTS account, license, per-source restrictions and attribution |
| Europe PMC / PubMed | Literature discovery and source verification | Index records are not evidence grading |

## Architecture
1. Terminology ingestion adapters per source with explicit license, version, language, checksum and update schedule.
2. Canonical concept store (stable source IDs), multilingual labels, synonyms, semantic types and provenance.
3. Typed relationship graph; separate disease-phenotype association from causation and diagnostic criteria. Record evidence strength when available.
4. Query pipeline: detect language → normalize spelling → lookup aliases → rank candidate concepts → disambiguate → compose transparent search query. Fallback to original query if unmatched.
5. Retrieval layer: Europe PMC API and PubMed E-utilities where permitted; deduplicate by DOI/PMID/PMCID; retain links to all origins. Google Scholar opens externally, no scraping.
6. Clinical education layer: differential possibilities only when sufficient case context exists, references and uncertainty; red-flag advice; clinician review.
7. Monitoring: unknown-query rate, mapping errors, source version drift, localization coverage, latency, and safety review.
8. Validation: physician-reviewed gold set including Arabic, Hebrew, English and German ambiguity cases; sensitivity/specificity by use case, negative tests, rare-disease coverage, and adversarial symptoms.

## Release gates
- No diagnostic claims before independent clinical validation, risk assessment and applicable regulatory review.
- Do not activate a terminology connector without verified license, data lineage, versioned update and tests.
- Display exact coverage/source status in UI.
- Do not merge into main or deploy to primary site without user approval.

## Immediate prototype
The existing 15-concept synonym resolver remains **search-only**. The UI must say so. It is not a medical diagnosis dictionary. The next engineering milestone is a versioned MeSH/HPO data adapter with documented permissions and validation, then PubMed retrieval and source-level deduplication.
