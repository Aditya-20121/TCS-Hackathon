# Survey: AI-Assisted Pharmacovigilance and Medical Case Review

## 1. Introduction

Pharmacovigilance (PV) is the science of detecting, assessing, understanding, and preventing adverse effects or any other drug-related problems. The central workflow artifact is the Individual Case Safety Report (ICSR), a structured document that captures a patient's adverse drug reaction (ADR) experience. Each ICSR requires a trained medical reviewer to perform several sub-tasks: seriousness determination, medical terminology coding (MedDRA), labelling assessment, and causality evaluation.

The scale of the problem is significant. The FDA's FAERS database receives over one million new ICSRs per year. The WHO's VigiBase contains over 40 million reports as of 2024. The European EudraVigilance system processes hundreds of thousands of reports annually. Manual review of this volume by trained medical writers and pharmacovigilance scientists is expensive, slow, and introduces inter-reviewer variability. Automating any portion of the review workflow offers material efficiency gains.

This survey covers the landscape of existing approaches — from traditional rule-based case management systems, through statistical signal detection and NLP-based ADE extraction, to modern LLM-based end-to-end systems — and positions the AI-Led Medical Review Assistant within this space.

---

## 2. Regulatory and Clinical Context

### 2.1 ICSR Workflow Requirements

ICH E2D (Post-Approval Safety Data Management) defines the minimal data elements of an ICSR and the timelines for expedited reporting (15 calendar days for serious unexpected ADRs). ICH E2A defines seriousness criteria. MedDRA (Medical Dictionary for Regulatory Activities) is mandated by all major regulatory authorities (FDA, EMA, PMDA, Health Canada) for ADR coding. These standards define the ground truth that any automated system must replicate.

### 2.2 The Four Core Sub-Tasks

Every ICSR review involves:

1. **Seriousness Assessment** — Binary classification (Serious/Non-serious) against six ICH E2A criteria: death, life-threatening, hospitalisation, disability/incapacity, congenital anomaly, medically significant event.

2. **MedDRA Coding** — Mapping the verbatim patient report to a standardized term in the MedDRA hierarchy, at minimum to Preferred Term (PT) and System Organ Class (SOC) level.

3. **Labelling Assessment** — Checking whether the coded ADR is listed in the drug's current Reference Safety Information (RSI), which determines whether an expedited report is required.

4. **Causality Assessment** — Evaluating the strength of the causal relationship between the drug and the event, using structured algorithms (Naranjo, WHO-UMC) or reviewer judgment.

---

## 3. Traditional Pharmacovigilance Case Management Systems

### 3.1 Oracle Argus Safety

Oracle Argus Safety is the dominant commercial ICSR processing platform globally, used by the majority of large pharmaceutical companies. It provides end-to-end case management: data entry, MedDRA coding (via manual lookup), workflow routing, regulatory submission formatting (E2B R3 XML), and reporting. Argus is fundamentally a workflow system, not an AI system — coding and assessment remain manual, with the platform providing structured data entry templates and reference data.

**Limitation:** No AI-driven assessment. Every sub-task requires human input. Argus automates the administrative workflow (routing, submission, tracking) but not the clinical decision-making.

### 3.2 IQVIA Vigilance (formerly Empirica Signal)

IQVIA Vigilance is a signal detection and analysis platform that applies statistical disproportionality methods to FAERS and VigiBase data. It does not assist with individual case review but with population-level signal detection across large datasets.

### 3.3 Veeva Vault Safety

Veeva Vault Safety is a newer cloud-based ICSR management system that competes with Oracle Argus. Like Argus, it provides workflow automation but relies on human reviewers for clinical decisions. Recent versions have begun incorporating AI-assisted MedDRA suggestions, though published performance benchmarks are not publicly available.

### 3.4 WHO VigiBase and VigiLyze

VigiBase is the WHO's global ICSR database (maintained by the Uppsala Monitoring Centre, UMC). VigiLyze is the web interface for accessing it. The UMC developed the BCPNN (Bayesian Confidence Propagation Neural Network) — a statistical disproportionality method that identifies drug-event combinations that occur more frequently than expected by chance. This is a signal detection tool, not an ICSR review tool.

**Distinction from our work:** All of these systems address database management and signal detection at population level. None assist the individual case reviewer in making per-ICSR clinical decisions.

---

## 4. Statistical Signal Detection Methods

Signal detection operates on aggregated ICSR databases to identify drug-event pairs that warrant further investigation. These methods are well-established and orthogonal to individual case review automation.

### 4.1 Disproportionality Analysis

The standard toolbox includes:

- **PRR (Proportional Reporting Ratio)** — Evans et al., used by MHRA and EMA
- **ROR (Reporting Odds Ratio)** — analogous to the odds ratio in case-control studies
- **EBGM (Empirical Bayes Geometric Mean)** — DuMouchel's multi-item Gamma Poisson Shrinker (MGPS), used by FDA
- **BCPNN (IC, Information Component)** — Bayesian approach used by WHO UMC

These methods identify statistical associations in large databases but provide no assistance with individual case assessment, and they are not competitive with or complementary to per-ICSR LLM-based review. They operate on coded, aggregated data rather than unstructured narratives.

---

## 5. NLP-Based Adverse Drug Event Detection

A large academic research community has focused on extracting ADR information from unstructured text — primarily EHR clinical notes, social media, and published case reports. This is closest in spirit to our task, but the goal is typically ADE *extraction* from text, not *assessment* of a structured report.

### 5.1 ADE Extraction from Electronic Health Records

Clinical NLP systems have been developed to identify adverse drug events in EHR clinical notes using:

- **Named Entity Recognition (NER)** — tagging drug names, adverse event mentions, dosages, and temporal markers in free text
- **Relation Extraction** — linking drug entities to adverse event entities within a sentence or document

Seminal datasets include the ADE corpus (Gurulingappa et al., 2012) with ~23,000 sentences from MEDLINE case reports annotated for ADR-drug pairs, and the CADEC corpus (Karimi et al., 2015) with social media posts annotated for ADE mentions and MedDRA mappings.

Standard approaches progressed from SVM + handcrafted features → BiLSTM-CRF → BERT-based sequence labelling (BioBERT, PubMedBERT, ClinicalBERT). State-of-the-art NER on the ADE corpus achieves F1 scores above 0.85 for drug-ADE pair identification.

**How this differs from our work:** ADE extraction identifies whether an ADR mention exists in text. Our task is richer: given a known drug-ADR pair, we assess seriousness, map to MedDRA, evaluate labelling, and determine causality. ADE extraction is an upstream task; our model operates downstream.

### 5.2 Social Media Pharmacovigilance

A distinct sub-field applies NLP to Twitter, Reddit, and patient forum posts to mine spontaneous ADR reports. Tools like MedWatcher (FDA's project) and academic systems trained on tweets demonstrate the feasibility of detecting ADR signals from informal text. These systems focus on identification and normalisation of lay ADR terminology to MedDRA PTs, not on clinical assessment.

### 5.3 The TAC 2017 Adverse Drug Reaction Shared Task

The Text Analysis Conference (TAC) 2017 ADR shared task provided a benchmark for extracting ADR information from FDA drug labels. Participating systems achieved strong performance on identifying ADR mentions in structured label sections using BERT-based models. This is most relevant to our labelling grounding strategy — we use the `adverse_reactions` section of FDA labels as context, a direct application of the insight from this shared task.

---

## 6. MedDRA Coding Automation

MedDRA coding is a natural candidate for automation — it is a classification task over a large (approximately 80,000 term) hierarchical vocabulary, with a clear ground truth.

### 6.1 Rule-Based and Dictionary Approaches

Early MedDRA coding tools used lexical lookup: match the verbatim report term against the MedDRA dictionary using exact match, then fuzzy match. Products like MedDRA AutoCoder (now part of various vendor platforms) implement this approach. Performance is high when verbatim terms closely match MedDRA LLT names, but degrades significantly for patient-reported lay terms, abbreviations, or multi-event reports.

### 6.2 Machine Learning Classification

Several academic groups have framed MedDRA PT assignment as a multi-class classification problem. SVM classifiers trained on TF-IDF features from verbatim terms achieved early baselines. Convolutional and recurrent neural networks improved on these, particularly for informal terminology.

### 6.3 BERT-Based Approaches

BioBERT and PubMedBERT substantially improved MedDRA coding accuracy by representing verbatim terms in a rich biomedical embedding space. Fine-tuning on company-proprietary annotated datasets (verbatim term → PT pairs from historical cases) is the standard approach for production systems. Reported accuracy for PT-level coding from verbatim terms is 85–92% in controlled settings with known vocabulary.

**Key limitation of all existing coding automation:** These systems code a single verbatim term in isolation. They do not integrate seriousness, labelling, or causality — the tasks remain siloed. Our model performs all four tasks jointly from a single narrative, enabling cross-task consistency.

### 6.4 Large-Scale Mapping Tools

The WHO's UMC maintains the WHOART-to-MedDRA bridging dictionary, and several vendors offer automated term migration tools when MedDRA is version-updated. These are lookup-based, not model-based.

---

## 7. Seriousness Classification

### 7.1 Rule-Based Systems

Seriousness determination is naturally rule-based: an event is serious if it meets any one of the six ICH E2A criteria. Modern ICSR platforms implement these rules directly as structured data fields — the reporter checks boxes for "hospitalisation required", "fatal", etc. When these fields are populated, seriousness is computed automatically.

The challenge is that in practice, seriousness criteria are sometimes buried in narrative text rather than checked in structured fields. This is where NLP adds value.

### 7.2 NLP Approaches to Seriousness Detection

Several systems have been trained to identify seriousness indicators in narrative text using keyword spotting, rule-based parsers, and latterly BERT-based classifiers. Published academic work on seriousness classification from FAERS narratives typically reports accuracy above 90%, but these results are often on filtered subsets where seriousness indicators are clearly stated.

**Our approach:** Given that seriousness criteria are available as structured fields from FAERS, we treat seriousness as a factual label extracted from the record rather than inferred from narrative. This eliminates the classification problem entirely for the training data generation phase, and the fine-tuned model learns to assess seriousness from the narrative the same way a human reviewer would — by identifying mentions of hospitalisation, death, or other criteria.

---

## 8. Causality Assessment Automation

### 8.1 The Naranjo Algorithm

The Naranjo Adverse Drug Reaction Probability Scale (Naranjo et al., 1981) is a 10-question questionnaire that produces a numeric score mapped to four probability categories: Definite (≥9), Probable (5–8), Possible (1–4), Doubtful (≤0). It remains the most widely cited causality algorithm in clinical and academic settings.

Several groups have attempted to automate Naranjo scoring from case narratives using NLP: extracting mentions of dechallenge, rechallenge, time to onset, and alternative explanations. Published systems achieve reasonable recall on individual question items but low precision — the clinical nuances required by several questions (e.g., "were there objective findings to confirm the reaction") are difficult to extract from informal text.

**Our decision to drop Naranjo:** In our setting, all narratives are synthetically generated from the same FAERS record. Scoring a generated narrative with Naranjo produces circular output — the score reflects what we put in the narrative, not independent clinical evidence. This is a fundamental problem specific to synthetic data pipelines, not a limitation of Naranjo in general.

### 8.2 WHO-UMC Causality Categories

The WHO-UMC causality classification (Certain, Probable/Likely, Possible, Unlikely, Conditional/Unclassified, Unassessable) is widely used in spontaneous reporting databases. It is qualitative rather than algorithmic, making it harder to automate from text but easier to compute from structured fields.

Our deterministic WHO-UMC computation from FAERS fields (outcome, dechallenge, rechallenge) is a novel aspect of our pipeline — to our knowledge, no existing system computes WHO-UMC deterministically from FAERS structured fields at scale in a training data generation context. Most published work either assigns WHO-UMC manually for small case series or uses simplified heuristics.

### 8.3 BCPNN for Population-Level Causality

The Bayesian Confidence Propagation Neural Network (BCPNN) developed at WHO UMC operates at population level — it measures disproportionality of a drug-event pair in a database rather than causality for an individual case. It is not a case-level causality tool and is not directly comparable to case-level algorithms.

---

## 9. Large Language Models for Pharmacovigilance

The most relevant and recent body of work applies LLMs to PV tasks. This section surveys the approaches most directly comparable to our project.

### 9.1 GPT-based Systems for ICSR Processing

Several groups have demonstrated GPT-3.5 and GPT-4's capability on individual PV tasks:

**Seriousness classification:** GPT-4 achieves strong performance (>90% accuracy) on seriousness classification from narrative text in zero-shot prompting, suggesting that general-purpose LLMs already have substantial capability on this sub-task without fine-tuning.

**MedDRA coding:** GPT-4 in zero-shot achieves approximately 60–70% PT-level accuracy on standard test sets, rising to 75–80% with few-shot examples. This is below the performance of fine-tuned BERT-based coding systems (~85–92%), indicating that MedDRA coding benefits substantially from specialised training.

**Labelling and causality:** Zero-shot GPT performance on labelling and causality tasks is less well-studied. Published work is sparse, partly because validated ground truth datasets for these tasks are not publicly available at scale.

**Limitation of GPT-based approaches:** Data privacy is a critical concern in pharmaceutical contexts. Sending ICSR data to third-party API endpoints violates HIPAA and many pharmaceutical company data governance policies. Additionally, GPT-4 API costs at ICSR-processing scale are prohibitive for continuous operation.

### 9.2 Biomedical Foundation Models

**BioBERT** (Lee et al., 2020) and **PubMedBERT** (Gu et al., 2021) are BERT-variant models pre-trained on biomedical literature. They represent the previous state of the art for biomedical NLP tasks including ADE extraction, clinical NER, and biomedical relation extraction. Their fixed-length input (512 tokens) limits their utility for full ICSR narrative processing, which may exceed this length. They also produce classification labels rather than structured outputs, limiting their applicability to multi-task assessment.

**BioGPT** (Microsoft, 2022) is an autoregressive GPT-2 style model pre-trained on PubMed abstracts. It performs well on biomedical QA and entity linking but was not designed for instruction following or structured output generation.

**Med-PaLM** (Google, 2023) and **Med-PaLM 2** achieve physician-level performance on USMLE-style medical QA. However, they are proprietary, access-restricted, and not fine-tunable for custom tasks. Their primary application domain is clinical QA, not pharmacovigilance workflow tasks.

### 9.3 Open-Source LLM Fine-Tuning for Clinical Tasks

The most relevant direct precedent for our approach is the growing body of work fine-tuning open-source LLMs on clinical tasks using LoRA:

**Clinical Camel** (2023) fine-tunes LLaMA-2-13B on clinical reasoning datasets using QLoRA. It demonstrates that 13B-class open models can approach GPT-4 performance on medical reasoning with targeted fine-tuning. Our approach is structurally similar but applied to PV-specific structured output tasks rather than general medical reasoning.

**MedAlpaca** fine-tunes LLaMA on medical QA datasets. Similarly demonstrates the general principle without PV-specific application.

**ClinicalGPT** (2023) applies RLHF and SFT to LLaMA specifically for clinical applications in Chinese healthcare. Demonstrates the applicability of instruction tuning for structured clinical output in non-English contexts.

None of these systems address pharmacovigilance sub-tasks specifically, and none address the problem of generating training data synthetically when labelled clinical data is unavailable.

### 9.4 Multi-Task Clinical NLP

The fundamental challenge in clinical NLP is that the four PV sub-tasks are typically solved by separate specialised models. Production systems typically involve:

- A BERT-based NER model for ADE extraction
- A classification head for seriousness
- A separate coding model for MedDRA assignment
- Manual review for labelling and causality

Each model is trained independently with no shared representation. Errors in upstream tasks (incorrect ADE extraction) propagate to downstream tasks (MedDRA coding) without any mechanism for cross-task correction.

End-to-end multi-task models that produce all four assessments from a single pass through a single model are rare. The closest approaches are prompt-chained GPT-4 systems where each sub-task is handled by a separate API call — but this is sequential processing of independent outputs, not joint prediction.

---

## 10. Comparison with the Proposed Approach

| Dimension | Traditional Systems | Specialised ML/NLP | GPT-4 Based | Our Approach |
| --- | --- | --- | --- | --- |
| Seriousness | Manual / rules | BERT classifier | Zero-shot | Fine-tuned, narrative-based |
| MedDRA coding | Manual / lexical | BERT fine-tuned | ~70% zero-shot | Fine-tuned, multi-task |
| Labelling | Manual | Not automated | Parametric | Grounded in FDA labels |
| Causality | Manual / Naranjo | Not automated | Zero-shot | Deterministic (WHO-UMC) |
| Multi-task | No | No | Separate calls | Single inference pass |
| Data privacy | Platform-level | On-premise | API (data leaves org) | On-premise, open weights |
| Infrastructure | Commercial SaaS | Custom deployment | Closed API | Open, ROCm compatible |
| Structured output | Yes (forms) | Classification labels | Prompt-dependent | Consistent JSON |
| Training data | N/A | Proprietary annotated | N/A | Synthetically generated |

### 10.1 Key Differentiators

**End-to-end single-model assessment:** Existing commercial systems and research approaches treat the four sub-tasks as independent problems. Our model produces all four assessments in a single inference pass, with shared representation across tasks. This enables cross-task coherence — for example, a "Fatal" outcome in the narrative is consistently linked to "Serious" seriousness, "Possible" WHO-UMC, and appropriate labelling, because the model sees all tasks simultaneously.

**Grounded labelling via FDA API:** No published PV NLP system, to our knowledge, integrates real-time FDA drug label text as context for labelling assessment. The typical approach is either manual lookup or parametric LLM knowledge. Our two-prompt system (with label context vs. without) is a novel grounding strategy specifically for the labelling sub-task.

**Synthetic training data pipeline at scale:** The teacher-student pipeline — using a large generative model to create training labels for a smaller student model — is a well-established approach in NLP (knowledge distillation). However, applying this specifically to the pharmacovigilance domain, where labelled training data is scarce and privacy-restricted, is a novel contribution. We generate 10,000 labelled ICSRs from public FAERS data without any access to proprietary case data.

**On-premise, open weights, AMD ROCm:** All existing LLM-based PV tools either rely on closed API access (OpenAI, Anthropic) or are built on CUDA-only frameworks. Our stack runs entirely on open-weight models with a ROCm-compatible training stack, enabling deployment in pharmaceutical environments where data cannot leave on-premise infrastructure.

**Deterministic causality from structured fields:** Rather than attempting to infer causality from narrative text (circular when the narrative is generated), we compute WHO-UMC from FAERS's structured dechallenge and rechallenge fields. This is a pragmatic and reproducible approach that has not been systematically applied in published PV NLP work.

---

## 11. Novelty Assessment

### 11.1 Confirmed Novel Contributions

1. **Grounded labelling via real-time FDA label API:** The two-template prompting strategy (LABELLING_PROMPT_WITH_LABEL / LABELLING_PROMPT_NO_LABEL) using live-fetched adverse reactions text as context for labelling determination has not been described in published PV NLP literature.

2. **Synthetic PV training dataset from FAERS + teacher-student pipeline:** Generating a large-scale pharmacovigilance training dataset using a generative LLM teacher from public FAERS data, with one-shot prompted ICH E2D narrative generation, is a novel dataset construction methodology.

3. **Deterministic WHO-UMC labelling at scale as training signal:** Using structured FAERS fields (outcome, dechallenge, rechallenge) to compute WHO-UMC causality labels deterministically, and incorporating these into LLM fine-tuning, has not been described in prior published work.

### 11.2 Incremental Contributions

4. **Multi-task fine-tuning for all four PV sub-tasks in one model:** Prior work fine-tunes models for individual PV sub-tasks. Combining all four into a single fine-tuned model with shared representation is a natural extension that has not been demonstrated at this task combination.

5. **ROCm-compatible PV fine-tuning stack:** The literature demonstrates PV fine-tuning on CUDA. Demonstrating the same on AMD ROCm with TRL + PEFT is novel in terms of platform but not conceptually.

---

## 12. Open Challenges and Future Directions

### 12.1 Ground Truth and Benchmarking

The absence of a public, large-scale, expert-annotated ICSR dataset is the single biggest obstacle to progress in this field. Pharmaceutical company data is proprietary. FAERS public data lacks narratives and expert labels. Future work should prioritize collaborative dataset creation with regulatory agencies or pharmaceutical consortia under appropriate de-identification and data sharing agreements.

### 12.2 Multimodal ICSR Processing

ICSRs increasingly include attachments — hospital discharge summaries, laboratory reports, imaging reports — that contain critical clinical information not captured in the structured fields. A complete ICSR review assistant would need to process these documents alongside the narrative, requiring multimodal or long-context architectures.

### 12.3 Regulatory-Grade Uncertainty Quantification

A model that says "Serious" is only as useful as its confidence in that prediction. Regulatory contexts require knowing when to defer to human review. Conformal prediction, calibrated probability estimation, or ensemble methods applied to LoRA adapters are promising directions for uncertainty quantification without significant inference overhead.

### 12.4 Multilingual ICSR Processing

ICH E2B(R3) accepts narratives in any language. A production-grade system would need to handle French, German, Japanese, Spanish, and Portuguese at minimum (major pharmaceutical market languages). Qwen3's multilingual pre-training is an asset here that has not yet been exploited.

### 12.5 Continuous Learning from Reviewer Feedback

A deployed system generates a stream of human feedback — reviewer acceptances, overrides, and corrections. This data is high-quality, task-specific supervision that should be used to continually improve the model. Designing a compliant, auditable continuous learning loop (where model updates are versioned and validated) is both a technical and regulatory challenge.

### 12.6 Signal Integration

A per-ICSR review assistant and a population-level signal detection system are complementary. Future systems could integrate the two: case-level structured assessments feed into signal detection algorithms (more reliably coded data improves signal quality), and population-level signals inform case-level priors (if a drug-event pair is a known signal, that context could improve causality assessment for individual cases).
