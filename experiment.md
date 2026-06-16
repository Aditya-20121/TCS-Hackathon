# Experimental Report: AI-Led Medical Review Assistant

## 1. Overview

This document describes the full experimental setup, methodology, design rationale, bottlenecks encountered, known limitations, and the path from prototype to production for the AI-Led Medical Review Assistant. The project trains Qwen3-14B with LoRA adapters to perform four pharmacovigilance sub-tasks — seriousness assessment, MedDRA coding, labelling status determination, and WHO-UMC causality classification — from a single unstructured clinical safety narrative.

The experiment is structured as a teacher-student pipeline: Qwen3-32B generates synthetic training labels, and Qwen3-14B is fine-tuned to replicate that behaviour at lower inference cost.

---

## 2. Experimental Setup

### 2.1 Hardware

| Component | Specification |
| --- | --- |
| GPU | AMD Instinct MI300X |
| VRAM | 192 GB HBM3 (unified) |
| Platform | AMD ROCm 6.x |
| Environment | AMD cloud (notebooks.amd.com) |

The MI300X's unified memory pool is the single most important hardware characteristic for this experiment. It allows Qwen3-32B (~64 GB BF16) and the fine-tuning job for Qwen3-14B (~35 GB BF16 + LoRA) to run on the same machine sequentially without re-provisioning, substantially reducing iteration time.

### 2.2 Software Stack

| Component | Library / Tool |
| --- | --- |
| Data generation | vLLM (OpenAI-compatible server) |
| Fine-tuning | TRL `Trainer` + PEFT `LoraConfig` |
| Model loading | HuggingFace `transformers` + `accelerate` |
| Loss masking | `DataCollatorForCompletionOnlyLM` (implemented locally — removed from TRL ≥0.14) |
| MedDRA lookup | JSON dict (`meddra_lookup.json`, 24,820 entries); codes attached post-inference |
| Demo interface | Gradio |
| Precision | BF16 throughout |

Unsloth was explicitly excluded — it is CUDA-only and does not support AMD ROCm. TRL + PEFT provide equivalent functionality with full ROCm compatibility.

### 2.3 Data Sources

| Source | Role | Volume |
| --- | --- | --- |
| openFDA Drug Event API (FAERS) | Structured AE records | 10,000 records |
| openFDA Drug Label API | Adverse reactions section text for labelling grounding | 344 unique drugs (24% coverage) |
| MEDDRA.xlsx (Kaggle) | PT → SOC hierarchy dictionary | Full MedDRA hierarchy |

### 2.4 Models

| Model | Role | VRAM |
| --- | --- | --- |
| Qwen3-32B (BF16) via vLLM | Teacher — generates narratives and labelling assessments | ~64 GB |
| Qwen3-14B (BF16) + LoRA | Student — fine-tuned for inference | ~35 GB (train), ~30 GB (infer) |

---

## 3. Methodology

### 3.1 Pipeline Overview

The pipeline has six stages:

```
FAERS fetch → MedDRA lookup → FDA label fetch → Data generation → Dataset formatting → LoRA fine-tuning
```

Each stage produces an intermediate file that can be inspected and resumed from, making the pipeline tolerant of interruptions and easy to debug.

### 3.2 Stage 1 — Data Collection (openFDA FAERS)

10,000 Individual Case Safety Reports (ICSRs) are fetched from the openFDA Drug Event API. For each record, structured fields are extracted: suspect drug, reaction PT (MedDRA Preferred Term), seriousness flag, seriousness criteria, patient demographics, reaction outcome, drug action taken (withdrawn/reduced/unchanged), and rechallenge result.

All records in the openFDA export have empty narrative fields. This is a known characteristic of FAERS public data — verbatim narratives are redacted for privacy. This means 100% of narratives must be generated synthetically, which is both a challenge (no real narrative supervision) and an opportunity (complete control over narrative style and ICH E2D compliance).

**WHO-UMC is computed here deterministically**, directly from the FAERS structured fields: `reaction_outcome`, `drug_additional` (dechallenge), and `drug_recurrence` (rechallenge). The mapping follows the WHO-UMC causality definition:

| Condition | WHO-UMC Category |
| --- | --- |
| Positive rechallenge + positive dechallenge | Certain |
| Positive dechallenge + resolved outcome | Probable/Likely |
| Resolved outcome + serious event | Probable/Likely |
| Resolved outcome | Possible |
| Fatal outcome | Possible |
| Not recovered | Unlikely |
| Otherwise | Unassessable |

### 3.3 Stage 2 — MedDRA Lookup Construction

The MEDDRA.xlsx file (Kaggle) is parsed to build a dictionary from PT name to the PT code and SOC name/code. FAERS reports use PT names that occasionally deviate from the MedDRA dictionary (differences in capitalisation, British/American spelling, minor wording variations). `rapidfuzz` with a similarity threshold of 80 is used to resolve these mismatches. The resulting lookup table has near-100% coverage for the 10,000 fetched records.

### 3.4 Stage 3 — FDA Drug Label Fetch

For each unique drug name in the dataset, the openFDA Drug Label API is queried for the `adverse_reactions` section (truncated to 2,500 characters) or the `warnings` section as fallback (1,500 characters). Results are cached to `data/drug_labels.json` so subsequent runs do not re-fetch.

344 out of 1,444 unique drug names returned a valid label (24%). The remaining 76% fall back to the LLM's parametric knowledge during labelling generation. This cache forms the grounding context for Stage 4.

### 3.5 Stage 4 — Synthetic Data Generation (Qwen3-32B)

Qwen3-32B runs via vLLM in non-thinking mode (`enable_thinking: false`). Two generation tasks run per record:

**Narrative generation:** A structured FAERS record (drug, reaction, demographics, outcome, action taken) is formatted into a prompt with one ICH E2D-formatted example. The model generates a 3–5 sentence clinical narrative in third-person clinical register. Temperature is set to 0.5 for stylistic variation across similar records while maintaining clinical accuracy. The one-shot example anchors the model to ICH E2D conventions (temporal relationship, action taken, outcome described).

**Labelling status determination:** When a drug label is available in the cache, the `adverse_reactions` section text is injected directly into the prompt, and the model is asked whether the reaction is listed (Expected) or not (Unexpected), with evidence. When no label is available, the model falls back to its parametric knowledge with a note that label verification was unavailable. Two separate prompt templates handle these two paths.

The generation loop uses `asyncio` with a semaphore limiting concurrency to 8 requests. Checkpoints are written every 200 records. On MI300X, throughput is approximately 800–1,200 records per hour.

### 3.6 Stage 5 — Dataset Formatting

Generated records are validated: narratives must be ≥ 100 characters, seriousness must be one of the two valid values, WHO-UMC must be one of the 6 standard categories, labelling status must be Expected or Unexpected. Records failing validation are discarded (typical rejection rate: 5–15%, mostly malformed JSON from the model).

Passing records are formatted into Qwen3's ChatML template with `DataCollatorForCompletionOnlyLM` applied at training time to mask the system and user tokens. Only the assistant's JSON output receives gradient signal.

### 3.7 Stage 6 — LoRA Fine-Tuning

Qwen3-14B is fine-tuned with LoRA adapters injected into all 7 linear layers (q, k, v, o projections in attention; gate, up, down projections in the SwiGLU MLP). Rank is set to 64 with alpha 128 (effective scaling factor 2.0). This trains approximately 160M parameters out of 14.8B total (~1.1%).

Training uses TRL's `SFTTrainer` with cosine LR scheduling, 5% linear warmup, effective batch size 32 (per-device batch 2, gradient accumulation 16), and a 2048-token maximum sequence length. Three epochs over the full dataset. BF16 precision throughout.

---

## 4. Key Design Decisions and Justifications

### 4.1 Dropping Naranjo, Keeping WHO-UMC

**Decision:** The experiment originally included Naranjo score generation as a causality measure. It was removed early and replaced with deterministic WHO-UMC computation.

**Justification:** The Naranjo algorithm requires 10 clinical data points, several of which are not present in FAERS records: previous conclusive reports of the reaction, patient's response to placebo, laboratory confirmation, prior exposure without the reaction, and dose-response evidence. Because narratives are generated synthetically from the same FAERS record, any Naranjo score the LLM produces is circular — it scores its own output. During testing, this produced near-uniform "Possible" ratings (score 1–4) across all records, making the field statistically useless for training.

WHO-UMC, by contrast, is defined in terms of exactly the fields FAERS captures: outcome, dechallenge, and rechallenge. The computation is deterministic, reproducible, and grounded in observed fact rather than inferred from narrative text.

### 4.2 FDA Drug Label API for Labelling

**Decision:** Instead of relying entirely on the model's parametric memory for whether an ADR is labelled, the actual `adverse_reactions` section from the FDA prescribing information is fetched and injected into the prompt.

**Justification:** LLM parametric knowledge about drug labels is unreliable in three specific ways: (1) labels are updated frequently and training data has a knowledge cutoff; (2) brand-name/generic-name disambiguation is imperfect; (3) rare ADRs with limited literature representation are prone to hallucination. Grounding on real label text converts an inference task into a document-reading task, which LLMs perform substantially better. The 24% coverage rate is acceptable for a training dataset — 2,400+ out of 10,000 records have grounded labels, providing a strong supervision signal for the student model.

### 4.3 Qwen3-32B as Teacher Model

**Decision:** Use Qwen3-32B for generation rather than a closed API (GPT-4, Claude) or a smaller open model.

**Justification:** Three reasons. First, **privacy**: FAERS data, while public, contains drug names and demographic information that should not leave the research environment. Running on-premise avoids any data transfer to third-party APIs. Second, **cost**: at 10,000 records × 2 generation calls per record, a closed API would incur substantial per-token charges. vLLM on MI300X is effectively free after hardware allocation. Third, **reproducibility**: the same model weights and generation parameters produce consistent outputs across runs. Qwen3-32B is the largest dense model in the Qwen3 family and produces narrative quality comparable to proprietary models on structured clinical prompts.

### 4.4 One-Shot Prompting for Narratives

**Decision:** Include one ICH E2D-formatted example in every narrative generation prompt.

**Justification:** Zero-shot narrative generation produced inconsistent output — some responses were too short (1–2 sentences), others too long (8–10 sentences), and tone varied between clinical and informal register. Adding one high-quality example anchors the model's output distribution to the desired format without limiting expressiveness. The one-shot example is fixed (atorvastatin/myopathy case) because it demonstrates temporal relationship, action taken, and resolved outcome — the three structural elements required by ICH E2D.

### 4.5 LoRA Rank 64 Across All 7 Linear Layers

**Decision:** Set LoRA rank to 64 (alpha=128) and apply to all 7 linear layers, rather than a lower rank on fewer layers.

**Justification:** The fine-tuning task requires the model to learn four distinct clinical classification schemas simultaneously (seriousness, MedDRA taxonomy, labelling vocabulary, WHO-UMC categories) and to output them in consistent JSON structure. Lower ranks (r=16 or r=32) on attention-only layers were found in ablation studies on similar instruction-tuning tasks to underfit multi-task instruction following. Applying adapters to MLP layers as well as attention is important because the MLP layers carry a significant portion of factual knowledge retrieval, which is central to MedDRA coding. Rank 64 uses approximately 13 GB of additional memory during training — well within the MI300X budget.

### 4.6 Response-Only Loss Masking

**Decision:** Use `DataCollatorForCompletionOnlyLM` to compute loss only over the assistant's JSON output.

**Justification:** A Qwen3-14B model pre-trained on ChatML data already handles system and user prompts well — computing cross-entropy loss over those tokens wastes gradient capacity on tokens the model handles correctly by default and can cause the model to overfit to the specific prompt wording. Masking the prompt means 100% of training signal flows through the JSON output, accelerating convergence and improving structural reliability of the output format.

---

## 5. Bottlenecks and How They Were Resolved

### 5.1 FAERS Narrative Coverage

**Bottleneck:** All 10,000 fetched FAERS records had empty `narrative` fields. This was not anticipated initially.

**Resolution:** Pivoted to 100% synthetic narrative generation using Qwen3-32B. Rather than viewing this as a limitation, it was reframed as an advantage: all narratives are ICH E2D compliant, consistently formatted, and free of real patient-identifiable information.

### 5.2 Circular Causality Scoring

**Bottleneck:** Initial pipeline included Naranjo scoring via LLM, which produced unreliable, near-uniform "Possible" scores regardless of the case details because the model was effectively scoring its own output.

**Resolution:** Dropped Naranjo entirely. Implemented deterministic WHO-UMC computation from FAERS structured fields. This required adding three new fields to the FAERS fetch (`drug_additional`, `drug_recurrence`, `reaction_outcome`) and a backfill step for already-fetched records.

### 5.3 Labelling Hallucination

**Bottleneck:** When asked whether an ADR is listed in a drug's label, the LLM sometimes confidently hallucinated label content, particularly for less-common drugs or brand-name products.

**Resolution:** Integrated the openFDA Drug Label API. The model now receives the actual `adverse_reactions` section text as context when available. This converts a knowledge retrieval problem (prone to hallucination) into a reading comprehension problem (which LLMs handle more reliably). A separate prompt template handles the fallback case where no label is available.

### 5.4 AMD ROCm Compatibility

**Bottleneck:** Several popular fine-tuning libraries (Unsloth, BitsAndBytes quantization, some Flash Attention implementations) are CUDA-only and fail on AMD ROCm.

**Resolution:** Constrained the stack to TRL + PEFT + HuggingFace `transformers`, all of which support ROCm. Quantization (QLoRA) was not used — BF16 full-precision LoRA was used instead, which is sufficient given the 192 GB VRAM budget. This actually simplifies the pipeline by eliminating quantization-related precision issues.

### 5.5 openFDA Rate Limiting

**Bottleneck:** The openFDA API enforces rate limits (1,000 requests/day without an API key, approximately 240 requests/minute with one). Fetching 10,000 records required pagination management and retry logic.

**Resolution:** Checkpointed fetching saves progress every 500 records. A configurable `SLEEP_BETWEEN` parameter spaces out requests. For users hitting rate limits, the documentation notes how to add an API key to double throughput. The 10,000 records were fetched across multiple sessions using the checkpoint resume capability.

### 5.6 FDA Label Coverage Gap

**Bottleneck:** Only 344 out of 1,444 unique drug names (24%) returned valid FDA labels. FAERS drug names often appear as abbreviations, misspellings, or informal trade names that don't match the FDA label database keys.

**Resolution:** Three search strategies are tried in order for each drug name: brand name, generic name (INN), and active substance name. Despite this, coverage remains at 24%. The 76% fallback to parametric knowledge is accepted as a pragmatic limitation for a hackathon-scale experiment. In production, a drug name normalisation layer (e.g., RxNorm lookup before FDA label query) would substantially improve coverage.

---

## 6. Quality Assessment

After data generation, a quality check cell in notebook 03 samples 50 records and evaluates:

- **Narrative coherence:** Length distribution, presence of temporal markers ("after", "following", "days later"), third-person construction
- **Labelling consistency:** Ratio of Expected vs. Unexpected across the dataset; evidence string presence
- **WHO-UMC distribution:** Checks that all 6 categories are represented and that the distribution is plausible (Possible should be the plurality)
- **JSON validity:** All records must parse cleanly; malformed records are reported

Typical dataset statistics after quality filtering:
- ~8,500–9,000 valid records from 10,000 generated (5–15% rejection)
- Training / validation split: 90/10
- WHO-UMC distribution: Possible (~45%), Probable/Likely (~30%), Unlikely (~10%), Unassessable (~10%), Certain/Conditional (~5%)
- Labelling: approximately 60% Expected, 40% Unexpected (varies by drug cohort)

---

## 7. Evaluation Results

### 7.1 Setup

Evaluation uses `infer/eval.py` run via `run_eval.sh`. Both base and fine-tuned models receive the same prompt (identical system prompt + user instruction) to ensure a fair, reportable comparison. Thinking mode is disabled (`enable_thinking=False` via `apply_chat_template`) for both configurations.

- **Evaluation set:** `data/val_chatml.jsonl` — 50 records (held-out from training)
- **Metrics:** Exact-match accuracy on four fields: `seriousness`, `meddra_pt` (case-insensitive), `meddra_soc` (case-insensitive), `labelling_status`
- **Excluded from eval:** `who_umc_category` (computed deterministically from FAERS fields — not a measure of narrative understanding) and MedDRA codes (looked up post-inference, not predicted)

### 7.2 Fine-Tuned Model Results

| Metric | Result |
| --- | --- |
| Records evaluated | 50 |
| JSON parse rate | 100.0% (50/50) |
| Seriousness | 94.0% |
| MedDRA PT | 94.0% |
| MedDRA SOC | 98.0% |
| Labelling status | 92.0% |
| **All four fields correct** | **80.0%** |

### 7.3 Notes on Metrics

**Why "All fields correct" is the primary metric:** Simple per-field averages (94+94+98+92)/4 = 94.5% overstate practical utility. A case that gets seriousness wrong but MedDRA right is still incorrect from a regulatory standpoint. The 80% all-correct rate reflects real-world usefulness — 4 in 5 records need no human correction on the evaluated fields.

**SOC vs PT accuracy:** SOC (98%) is consistently higher than PT (94%) because SOC is a coarser classification (27 top-level categories vs. ~80,000 MedDRA terms). PT accuracy at 94% is strong given the size of the MedDRA vocabulary.

**Base model results:** Not yet recorded. The `run_eval.sh` script saves both runs to `data/eval_results/base.txt` and `data/eval_results/finetuned.txt` for side-by-side comparison.

---

## 8. Limitations of the Current Experiment

### 7.1 Synthetic Training Data

All clinical narratives are generated by Qwen3-32B rather than written by human medical reviewers. This introduces a **teacher-student knowledge transfer bottleneck**: Qwen3-14B cannot exceed the quality ceiling set by Qwen3-32B. More critically, the fine-tuned model learns the writing style and clinical reasoning patterns of Qwen3-32B, not those of actual pharmacovigilance professionals. If Qwen3-32B has systematic biases in how it describes adverse events (e.g., consistently over-estimating event severity in narrative language), Qwen3-14B will inherit those biases.

Real clinical narratives from regulatory submissions would substantially improve signal quality, but are subject to privacy regulations (21 CFR, GDPR) and are not publicly available at scale.

### 7.2 FDA Label Coverage

Only 24% of drug-event pairs have grounded labelling decisions. The remaining 76% are assessed from LLM parametric memory, which is subject to knowledge cutoff and hallucination for newer or less-documented drugs. This limits the reliability of the `labelling_status` field for less-common drugs.

### 7.3 WHO-UMC Simplification

The deterministic WHO-UMC computation uses a simplified mapping from three FAERS fields. The actual WHO-UMC guidelines also consider the time relationship between exposure and event, biological plausibility, and alternative explanations — none of which are derivable from FAERS structured fields without a narrative. As a result, the WHO-UMC labels are systematically biased toward "Possible" (the default when little structured information is available) and may not reflect the nuanced causality assessment a trained reviewer would produce.

### 7.4 Single-Drug, Single-Reaction Assumption

FAERS records can contain multiple suspect drugs and multiple reactions. The pipeline selects the primary suspect drug and the first reported reaction. This simplification means the model is not trained on polypharmacy cases, drug-drug interaction scenarios, or multi-event reports — which are common in real regulatory submissions.

### 7.5 English-Only

FAERS is predominantly English, and all prompts and training data are in English. The model is not validated for use with narratives in other languages, despite WHO ICSR guidelines permitting reports in any language.

### 7.6 No Uncertainty Quantification

The model outputs a single point prediction (e.g., "Serious") with no associated confidence score. A regulatory context requires knowing not just what the model predicts but how certain that prediction is. Records where the model is uncertain should be flagged for mandatory human review, but the current architecture provides no mechanism for this.

### 7.7 MedDRA Version Sensitivity

MedDRA is versioned (updated twice yearly). The MEDDRA.xlsx used for lookup corresponds to a specific version. If the model is deployed against a newer MedDRA version, PT names and codes may have changed, causing lookup failures or incorrect coding.

---

## 9. Path to Production Deployment

Moving this prototype to a real pharmacovigilance workflow requires addressing the following:

### 8.1 Data Quality

Replace synthetic narratives with real de-identified ICSRs. Pharmaceutical companies and CROs maintain large repositories of historical case narratives that could serve as fine-tuning data under appropriate data sharing agreements. The current pipeline would remain unchanged — only the data source changes.

### 8.2 Drug Name Normalisation

Add an RxNorm lookup layer before the FDA label query. RxNorm maps brand names, generic names, and INNs to a canonical identifier, enabling reliable label retrieval. This would substantially improve the 24% label coverage rate.

### 8.3 Regulatory Validation

Deploying AI in the regulatory pharmacovigilance workflow triggers 21 CFR Part 11 requirements (for FDA submissions), GxP validation requirements, and in Europe, Article 61 of Regulation (EU) No 536/2014. This requires:

- Prospective validation study against expert human reviewer decisions
- Documented IQ/OQ/PQ (Installation/Operational/Performance Qualification)
- Audit trail for all AI-assisted decisions
- Change control procedures for model updates

### 8.4 Human-in-the-Loop Architecture

The model output should be positioned as a **triage recommendation**, not a final decision. A deployment architecture would surface the model's structured assessment alongside the narrative, allow the reviewer to confirm or override each field, and log the final decision for both regulatory compliance and model retraining.

### 8.5 Uncertainty Thresholding

Implement confidence scoring (e.g., Monte Carlo dropout, temperature scaling, or ensemble over LoRA adapters) to identify records where the model is uncertain. Records below a confidence threshold route automatically to senior human reviewers.

### 8.6 Serving Infrastructure

Replace the interactive `infer.py` script with a production serving stack:

- **Model serving:** vLLM or HuggingFace TGI with batching and autoscaling
- **API layer:** FastAPI REST endpoint accepting narrative text, returning structured JSON
- **Integration:** Webhook connectors for Oracle Argus Safety, Veeva Vault Safety, or other case management systems
- **Monitoring:** Latency, throughput, output distribution drift detection

### 8.7 MedDRA Lifecycle Management

Establish a quarterly update process tied to MedDRA version releases. Each release requires:

1. Regenerating `meddra_lookup.json` from the new MEDDRA.xlsx
2. Auditing model outputs for changed PT/SOC mappings
3. Retraining or fine-tuning if the distribution shift is material

### 8.8 Multilingual Extension

ICSR regulations require accepting reports in all languages. The current model would need either multilingual fine-tuning data or an upstream translation layer. Qwen3 natively supports multiple languages, which reduces the cost of multilingual extension compared to English-only base models.

### 8.9 Compliance and Privacy

- All patient narrative data must be de-identified before ingestion (HIPAA Safe Harbor or Expert Determination method)
- Model weights trained on de-identified data must not be extractable in ways that could re-identify patients
- GDPR Articles 22 and 35 (automated decision-making and DPIA) apply in EU deployments
