# AI-Led Medical Review Assistant

A fine-tuned LLM (Qwen3-14B + LoRA) that assists pharmacovigilance reviewers with four clinical tasks from a single unstructured clinical safety narrative:

1. **Seriousness Assessment** — Classifies events as Serious/Non-serious per ICH E2D criteria
2. **MedDRA Coding** — Maps adverse events to PT and SOC with 8-digit codes
3. **Labelling Status** — Determines if an event is Expected or Unexpected (grounded in FDA drug label text)
4. **Causality Assessment** — WHO-UMC category (computed deterministically from FAERS structured fields)

---

## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1: Fetch FAERS Data](#step-1-fetch-faers-data)
  - [Step 2: Build MedDRA Lookup](#step-2-build-meddra-lookup)
  - [Step 3: Generate Dataset with Qwen3-32B](#step-3-generate-dataset-with-qwen3-32b)
  - [Step 4: Format Training Data](#step-4-format-training-data)
  - [Step 5: Fine-tune Qwen3-14B](#step-5-fine-tune-qwen3-14b)
  - [Step 6: Run Inference](#step-6-run-inference)
  - [Step 7: Demo Interface](#step-7-demo-interface)
- [Input / Output Format](#input--output-format)
- [Design Decisions](#design-decisions)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)

---

## Background

### What is Pharmacovigilance?

Pharmacovigilance is the science of monitoring, detecting, and preventing adverse drug reactions (ADRs). When a patient experiences an unexpected reaction to a drug, a safety report called an **Individual Case Safety Report (ICSR)** is filed. Medical reviewers must process each report through a multi-step workflow:

1. Read the unstructured clinical narrative
2. Decide if the event meets **seriousness criteria** (death, hospitalization, life-threatening, disability, congenital anomaly, or medically significant)
3. Assign a standardized **MedDRA code** to the adverse event
4. Check if the event is **listed in the drug's label** (Expected) or not (Unexpected)
5. Assess the **causal relationship** between the drug and the event

This is repetitive, expert-intensive work. This project trains an LLM to assist reviewers by producing a structured JSON assessment from a plain-text narrative.

### What is MedDRA?

MedDRA (Medical Dictionary for Regulatory Activities) is the international standard terminology for adverse events. It organizes terms in a five-level hierarchy:

```
SOC   (System Organ Class)         ← broadest, e.g., "Hepatobiliary disorders"
  └─ HLGT (High Level Group Term)
       └─ HLT  (High Level Term)
            └─ PT   (Preferred Term)  ← standard reporting level
                 └─ LLT  (Lowest Level Term)  ← most specific, verbatim-level
```

Each term has a unique 8-digit numeric code. FAERS stores reactions at the PT level. When a verbatim LLT is not available, LLT = PT (standard practice in regulatory reporting).

---

## Architecture

The project is a two-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATASET CREATION                     │
│                                                                   │
│  openFDA FAERS API          MEDDRA.xlsx (Kaggle)                 │
│  (10,000 AE reports)   +    (full hierarchy dict)                │
│       │                           │                              │
│       ▼                           ▼                              │
│  Structured records          PT → SOC lookup                     │
│  (drug, reaction PT,         (fuzzy-matched)                     │
│   seriousness, demographics,      │                              │
│   outcome, rechallenge fields)    │                              │
│       │                           │                              │
│       │    openFDA Drug Label API │                              │
│       │    (adverse_reactions     │                              │
│       │     section per drug) ────┤                              │
│       └───────────────┬───────────┘                              │
│                       ▼                                          │
│              WHO-UMC computed deterministically                  │
│              from FAERS outcome/rechallenge fields               │
│                       │                                          │
│                       ▼                                          │
│              Qwen3-32B via vLLM                                  │
│              Generates (per record):                             │
│              • Clinical narrative  (100% generated — FAERS       │
│                narratives are empty)                             │
│              • Labelling status + evidence (grounded in          │
│                real FDA label text where available)              │
│                       │                                          │
│                       ▼                                          │
│              ChatML instruction-tuning JSONL                     │
│              (train.jsonl / val.jsonl)                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: FINE-TUNING + INFERENCE                │
│                                                                   │
│  Qwen3-14B (base)                                                │
│  + LoRA adapters (PEFT)                                          │
│  + TRL SFTTrainer                                                │
│  + Response-only loss masking                                    │
│       │                                                          │
│       ▼                                                          │
│  Fine-tuned Medical Review Assistant                             │
│       │                                                          │
│       ▼                                                          │
│  Input:  clinical narrative (plain text)                         │
│  Output: structured JSON with all 4 assessments                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Qwen3-32B for data generation?

Qwen3-32B is used as a **teacher model** to generate training labels — specifically, the fields FAERS does not provide (narrative text and labelling status). It runs in **non-thinking mode** (`enable_thinking: false`) for structured JSON output and consistent throughput. On the MI300X (192 GB), Qwen3-32B uses ~64 GB in BF16, leaving ample headroom for the fine-tuning job.

> **Note:** The Qwen3 model lineup is: 0.6B / 1.7B / 4B / 8B / 14B / 30B-A3B / **32B** / 235B-A22B. There is no Qwen3-72B. Qwen3-32B is the largest dense model in the family.

### Why LoRA and not full fine-tuning?

Full fine-tuning of a 14B-parameter model requires storing weights, gradients, and two optimizer momentum tensors, pushing memory above 160 GB. LoRA injects small trainable matrices (rank-64) into attention and MLP layers — only ~1–2% of parameters are trained. For Qwen3-14B in BF16, this uses approximately 35 GB of VRAM.

### Why response-only loss masking?

Computing cross-entropy loss over the entire sequence — including the system prompt and user instruction — wastes gradient updates on tokens the model already handles well. Masking the prompt tokens focuses 100% of training signal on the assistant's JSON output, improving convergence speed and output structure quality.

---

## Project Structure

```
TCS-Hackathon/
│
├── MEDDRA.xlsx                  ← MedDRA hierarchy dictionary (Kaggle dataset, gitignored)
│
├── 01_fetch_faers.ipynb         ← Fetch 10,000 AE reports from openFDA FAERS API
├── 02_explore_meddra.ipynb      ← Inspect MEDDRA.xlsx and build PT → SOC lookup
├── 03_generate_dataset.ipynb    ← Qwen3-32B data generation (async, checkpointed)
├── 04_format_dataset.ipynb      ← Validate, format, and split training data
├── 05_demo.py                   ← Gradio demo interface for the fine-tuned model
│
├── train/
│   └── train.py                 ← LoRA fine-tuning script (TRL + PEFT)
│
├── infer/
│   ├── infer.py                 ← Inference: interactive, single, or batch mode
│   └── eval.py                  ← Evaluation script: base vs fine-tuned comparison
│
├── run_eval.sh                  ← Shell script: runs eval.py for both configs, saves results
│
├── data/                        ← Created automatically during pipeline
│   ├── faers_raw.json           ← Raw FAERS records (output of notebook 01)
│   ├── drug_labels.json         ← FDA drug label cache, keyed by drug name
│   ├── meddra_lookup.json       ← PT name → hierarchy dict (output of notebook 02)
│   ├── dataset_raw.json         ← Generated records with all fields (output of notebook 03)
│   ├── train.jsonl              ← Training set in ChatML text format (for training loss)
│   ├── train_chatml.jsonl       ← Training set in messages format
│   ├── val.jsonl                ← Validation set in ChatML text format
│   └── val_chatml.jsonl         ← Validation set in messages format (used by eval.py)
│
├── train/output/                ← Created by train.py (gitignored)
│   ├── checkpoint-*/            ← Intermediate checkpoints
│   └── final_adapter/           ← Final LoRA adapter weights
│
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | AMD Instinct MI300X |
| VRAM | 192 GB HBM3 |
| Software | ROCm 6.1+ |

**VRAM usage by phase:**

| Phase | Task | VRAM |
|-------|------|------|
| Data generation | Qwen3-32B BF16 | ~64 GB |
| Fine-tuning | Qwen3-14B BF16 + LoRA r=64 | ~35 GB |
| Inference | Qwen3-14B BF16 + LoRA adapter | ~30 GB |

> **Note:** Unsloth is CUDA-only and **will not work on AMD ROCm**. This project uses `transformers` + `peft` + `trl` which are fully ROCm-compatible.

---

## Installation

### 1. Install PyTorch for ROCm

This must be installed separately before the other requirements. Check the ROCm version with `rocminfo | grep "ROCm Version"`, then:

```bash
# For ROCm 6.2
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2
```

Verify:
```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # AMD Instinct MI300X
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install vLLM (for Qwen3-32B data generation)

```bash
pip install vllm
```

### 4. Download Qwen3 models from HuggingFace

```bash
# Teacher model for data generation (used in notebook 03)
huggingface-cli download Qwen/Qwen3-32B

# Base model for fine-tuning (used in train.py and infer.py)
huggingface-cli download Qwen/Qwen3-14B
```

### 5. Place the MedDRA dataset

The `MEDDRA.xlsx` file (from [Kaggle: e0xextazy/meddra](https://www.kaggle.com/datasets/e0xextazy/meddra)) should be in the project root:

```
TCS-Hackathon/
└── MEDDRA.xlsx   ← must be here (gitignored — not tracked in the repo)
```

---

## Pipeline Walkthrough

### Step 1: Fetch FAERS Data

**Notebook:** `01_fetch_faers.ipynb`

Opens the [openFDA Drug Event API](https://open.fda.gov/apis/drug/event/) and fetches adverse event reports. No API key is required. The notebook paginates through results and saves a clean JSON record for each report.

**What it extracts per record:**

| Field | Description |
|-------|-------------|
| `drug_name` | Suspect drug (characterization = "1") |
| `reaction_pt` | MedDRA Preferred Term of the adverse event |
| `serious` | `true` if the event meets any seriousness criterion |
| `seriousness_criteria` | List: Death, Hospitalization, Life-threatening, Disability, Congenital Anomaly, Medically Significant |
| `patient_age` | Age at onset |
| `patient_sex` | male / female / unknown |
| `reaction_outcome` | Recovered, Recovering, Not Recovered, Fatal, Unknown |
| `drug_additional` | Action taken: 1=withdrawn, 2=reduced, 3=unchanged |
| `drug_recurrence` | Rechallenge result: 1=positive rechallenge |
| `concomitant_drugs` | List of co-administered drugs |
| `who_umc_preliminary` | WHO-UMC category computed deterministically (see below) |

**WHO-UMC computed from structured fields:**

WHO-UMC causality is derived deterministically from `reaction_outcome`, `drug_additional`, and `drug_recurrence` — no LLM is involved:

```
rechallenge positive + dechallenge positive → Certain
dechallenge positive + resolved             → Probable/Likely
resolved + serious event                    → Probable/Likely
resolved                                    → Possible
fatal outcome                               → Possible
not recovered                               → Unlikely
otherwise                                   → Unassessable
```

> **Why deterministic?** Naranjo-style scoring was considered but dropped — it requires clinical information (lab tests, alternative explanations, time to onset) that is absent from synthetic narratives. Assigning Naranjo from the same data used to generate the narrative produces circular, unreliable results. WHO-UMC maps cleanly to the FAERS fields that are factually grounded.

**Configuration:**

```python
TARGET_RECORDS = 10000   # total records to fetch
BATCH_SIZE     = 100     # records per API request
SLEEP_BETWEEN  = 0.5     # seconds between requests
```

**Checkpointing:** `data/faers_raw.json` is saved every 500 records. Re-running the notebook resumes from the last saved state.

**Expected output:** `data/faers_raw.json`

---

### Step 2: Build MedDRA Lookup

**Notebook:** `02_explore_meddra.ipynb`

Reads `MEDDRA.xlsx` and builds a dictionary mapping every PT name to its full MedDRA hierarchy (PT code, SOC name, SOC code).

**How column detection works:**

The notebook inspects every sheet in `MEDDRA.xlsx` and auto-detects columns by matching keywords:

```
pt_code      → PT_CODE        ✓
pt_name      → PT_NAME        ✓
soc_code     → SOC_CODE       ✓
soc_name     → SOC_NAME       ✓
```

If any required column shows `✗ NOT FOUND`, set it manually before proceeding:

```python
COL["pt_name"] = "Preferred Term"   # exact column header from the file
```

**Fuzzy matching:** FAERS PT names sometimes differ slightly from MedDRA dictionary entries. `rapidfuzz` is used with a threshold of 80. A test cell shows match results for common terms.

**Expected output:** `data/meddra_lookup.json`

---

### Step 3: Generate Dataset with Qwen3-32B

**Notebook:** `03_generate_dataset.ipynb`

This is the most compute-intensive step. Qwen3-32B generates the two fields FAERS does not provide: **clinical narratives** and **labelling status**. WHO-UMC is already computed in step 1 and passed through.

> **All 10,000 FAERS records have empty narratives** — the `narrative` field is blank in every fetched record. Qwen3-32B generates 100% of narratives.

#### Start vLLM first

In a **separate terminal**, before running this notebook:

```bash
vllm serve Qwen/Qwen3-32B \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --port 8000
```

Wait for `Application startup complete.` before proceeding. The notebook health-check cell confirms vLLM is reachable.

#### Cell execution order

Run cells in this order:

1. **Config** — sets paths, model URL, concurrency
2. **Imports**
3. **Load FAERS records**
4. **MedDRA lookup**
5. **FDA label fetch** — fetches the `adverse_reactions` section from openFDA drug label API for each unique drug; results cached to `data/drug_labels.json`
6. **Prompts** — defines `NARRATIVE_PROMPT`, `LABELLING_PROMPT_WITH_LABEL`, `LABELLING_PROMPT_NO_LABEL`
7. **`process_record()`** — async function that generates narrative + labelling for one record
8. **Quick test** — single record end-to-end test (run this before the main loop)
9. **Load checkpoint** — loads `dataset_raw.json` to resume partial runs
10. **Main generation loop** — processes all records with semaphore-limited concurrency, checkpoints every 200 records

#### What Qwen3-32B generates per record

**Narrative generation:**

The model writes a 3–5 sentence clinical narrative in ICH E2D format from FAERS structured fields. A one-shot example is included in every prompt:

```
Drug: Atorvastatin 40 mg | Reaction: Myopathy | Patient: 58y male | Outcome: Recovered
→ "A 58-year-old male patient with hypercholesterolaemia was prescribed Atorvastatin
   40 mg once daily. Approximately six weeks following initiation of therapy, he
   developed myopathy characterised by proximal muscle weakness and elevated serum
   creatine kinase. The drug was discontinued and symptoms resolved within four weeks."
```

**Labelling status:**

When an FDA drug label is available (fetched in the FDA label step), the prompt provides the actual `adverse_reactions` section text as context. When unavailable, a fallback prompt uses the model's parametric knowledge. This grounds labelling decisions in real prescribing information rather than model memory.

```json
{
  "labelling_status": "Expected",
  "labelling_evidence": "Myopathy is listed as an adverse reaction in the Atorvastatin prescribing information under Musculoskeletal adverse reactions."
}
```

> Approximately 24% of drugs (344/1,444 unique names) have FDA labels available via the openFDA label API. The remaining 76% fall back to LLM parametric knowledge.

#### Concurrency and checkpointing

```python
CONCURRENCY = 8    # concurrent requests to vLLM; increase to 12-16 if GPU allows
```

Checkpoints are written to `data/dataset_raw.json` every 200 records. Re-running the notebook skips already-processed report IDs automatically.

**Expected throughput:** ~800–1,200 records/hour on MI300X. Full 10,000 records: ~8–12 hours.

**Expected output:** `data/dataset_raw.json`

---

### Step 4: Format Training Data

**Notebook:** `04_format_dataset.ipynb`

Validates each generated record and formats it into the ChatML template Qwen3 was pre-trained on.

#### Validation rules

Records are dropped if they fail any of these checks:

| Check | Rule |
|-------|------|
| Narrative length | Must be ≥ 100 characters |
| MedDRA PT | Must be non-empty |
| Seriousness | Must be `"Serious"` or `"Non-serious"` |
| WHO-UMC category | Must be one of the 6 standard categories |
| Labelling status | Must be `"Expected"` or `"Unexpected"` |

Typical rejection rate is 5–15%, mostly from malformed JSON responses.

#### ChatML format

Each training example is formatted as:

```
<|im_start|>system
You are an expert pharmacovigilance medical reviewer...<|im_end|>
<|im_start|>user
Analyze the following clinical safety narrative and return a structured JSON assessment.

Narrative:
A 54-year-old male was admitted to the hospital...<|im_end|>
<|im_start|>assistant
{
  "seriousness": "Serious",
  "seriousness_criteria": ["Hospitalization"],
  "meddra_pt": "Jaundice",
  "meddra_pt_code": 10023126,
  "meddra_soc": "Hepatobiliary disorders",
  "meddra_soc_code": 10019805,
  "labelling_status": "Unexpected",
  "labelling_evidence": "...",
  "who_umc_category": "Probable/Likely"
}<|im_end|>
```

**Loss masking:** `DataCollatorForCompletionOnlyLM` masks the system and user tokens so gradients flow only through the assistant's JSON output.

**Expected outputs:**

| File | Format | Use |
| --- | --- | --- |
| `data/train.jsonl` | `{"text": "..."}` (full ChatML string) | SFTTrainer input |
| `data/val.jsonl` | Same | Validation during training |

---

### Step 5: Fine-tune Qwen3-14B

**Script:** `train/train.py`

```bash
cd train
python train.py
```

To resume from a checkpoint:
```bash
python train.py --resume_from_checkpoint ./output/checkpoint-500
```

#### LoRA configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (rank) | 64 | Sufficient capacity for multi-task clinical JSON |
| `alpha` | 128 | Scaling factor = alpha/r = 2.0; standard for task adaptation |
| `dropout` | 0.05 | Light regularization |
| Target modules | All 7 linear layers | Covers both attention and feed-forward layers |
| Trainable params | ~1–2% of total | ~160M of 14.8B |

**Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) + `gate_proj`, `up_proj`, `down_proj` (SwiGLU MLP).

#### Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size (per device) | 2 |
| Gradient accumulation | 16 |
| Effective batch size | 32 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup | 5% of steps |
| Max sequence length | 2048 |
| Precision | BF16 |

**VRAM:** ~35 GB. **Training time:** ~8–12 hours for 10,000 examples × 3 epochs on MI300X.

#### Output

```
train/output/final_adapter/
├── adapter_config.json          ← LoRA configuration
├── adapter_model.safetensors    ← Trained adapter weights (~500 MB)
├── tokenizer.json
└── tokenizer_config.json
```

The base model weights are not duplicated. Both are loaded together at inference time.

---

### Step 6: Run Inference

**Script:** `infer/infer.py`

#### Interactive mode (default)

```bash
cd infer
python infer.py
```

Paste a narrative, press Enter twice:

```
===================================================================
MEDICAL REVIEW ASSESSMENT
===================================================================
  Seriousness         : Serious
  Criteria            : Hospitalization
  MedDRA PT           : Jaundice [10023126]
  MedDRA SOC          : Hepatobiliary disorders [10019805]
  Labelling Status    : Unexpected
  Labelling Evidence  : Review of Drug X prescribing information shows no documentation of jaundice.
  WHO-UMC             : Probable/Likely

  [2.4s | 187 tokens]
===================================================================
```

#### Single narrative

```bash
python infer.py --narrative "A 54-year-old male developed jaundice three weeks after starting Drug X 500mg daily."
```

#### Batch mode

```bash
python infer.py --input_file narratives.txt --output_file results.jsonl
```

One narrative per line in `narratives.txt`. Results written as JSONL.

---

### Step 7: Demo Interface

**Script:** `05_demo.py`

A Gradio web UI for demonstrating the fine-tuned model.

```bash
python 05_demo.py           # local URL only
python 05_demo.py --share   # public shareable link via Gradio tunnel
```

**Features:**

- Six pre-loaded example narratives (amoxicillin anaphylaxis, warfarin ICH, atorvastatin myalgia, adalimumab injection site, ibuprofen GI bleed, clozapine agranulocytosis)
- Color-coded result cards: Seriousness, MedDRA PT + SOC (with 8-digit codes), Labelling Status
- MedDRA PT and SOC codes looked up post-inference from `data/meddra_lookup.json`
- Same prompt as evaluation (no few-shot examples) for consistent, reportable results

### Step 8: Evaluate — Base vs Fine-Tuned

**Script:** `infer/eval.py` | **Runner:** `run_eval.sh`

Evaluates 50 records from `data/val_chatml.jsonl` using the same prompt for both configurations, enabling a fair comparison.

```bash
bash run_eval.sh
```

Results are saved to `data/eval_results/` as `base.txt`, `finetuned.txt`, and `comparison_summary.txt`.

**Fine-tuned model results (50 records):**

| Metric | Base Model | Fine-Tuned |
|--------|-----------|------------|
| JSON parse rate | — | 100.0% |
| Seriousness | — | 94.0% |
| MedDRA PT | — | 94.0% |
| MedDRA SOC | — | 98.0% |
| Labelling status | — | 92.0% |
| **All fields correct** | — | **80.0%** |

---

## Input / Output Format

### Input

Any unstructured clinical safety narrative in plain English. Recommended length: 2–10 sentences. Very short inputs (< 2 sentences) may produce lower-quality assessments.

### Output

```json
{
  "seriousness": "Serious",
  "seriousness_criteria": ["Hospitalization", "Life-threatening"],

  "meddra_pt": "Anaphylactic reaction",
  "meddra_pt_code": 10002198,
  "meddra_soc": "Immune system disorders",
  "meddra_soc_code": 10021428,

  "labelling_status": "Expected",
  "labelling_evidence": "Anaphylaxis is listed as a known serious adverse reaction in the prescribing information.",

  "who_umc_category": "Probable/Likely"
}
```

> **Note on MedDRA codes:** The model predicts `meddra_pt` and `meddra_soc` as names. The 8-digit `meddra_pt_code` and `meddra_soc_code` are looked up post-inference from `data/meddra_lookup.json` (24,820 entries). This is more reliable than asking the model to memorize 8-digit codes.

**Field definitions:**

| Field | Type | Values |
|-------|------|--------|
| `seriousness` | string | `"Serious"` or `"Non-serious"` |
| `seriousness_criteria` | array | Subset of: Death, Hospitalization, Life-threatening, Disability/Incapacity, Congenital Anomaly, Medically Significant |
| `meddra_pt` | string | MedDRA Preferred Term name |
| `meddra_pt_code` | integer | 8-digit MedDRA PT code |
| `meddra_soc` | string | MedDRA System Organ Class name |
| `meddra_soc_code` | integer | 8-digit MedDRA SOC code |
| `labelling_status` | string | `"Expected"` or `"Unexpected"` |
| `labelling_evidence` | string | One-sentence rationale |
| `who_umc_category` | string | `"Certain"`, `"Probable/Likely"`, `"Possible"`, `"Unlikely"`, `"Conditional/Unclassified"`, `"Unassessable"` |

---

## Design Decisions

### Why no Naranjo score?

The Naranjo algorithm requires specific clinical information: previous conclusive reports of the reaction, objective confirmation by lab testing, known placebo response rate, and several others. Generating a synthetic narrative and then scoring that same narrative with Naranjo produces a circular result where the score simply reflects what the LLM put in the narrative, not an independent clinical judgment. Testing confirmed this: Naranjo outputs were near-uniformly "Possible" (score 1–4) across all generated records, making the field uninformative. WHO-UMC was retained because it maps cleanly to FAERS fields that are factually grounded (outcome, dechallenge, rechallenge), not to narrative text.

### Why FDA drug label API for labelling?

Using only the model's parametric knowledge for labelling status risks hallucination — especially for brand-name drugs, recently approved drugs, or drugs with rare ADRs. By fetching the actual `adverse_reactions` section from the openFDA drug label API and injecting it into the prompt, labelling decisions are grounded in the drug's official prescribing information. Records where no label is available fall back to parametric knowledge with an explicit note in `labelling_evidence`.

### Why few-shot prompting for narratives?

Zero-shot narrative generation produced variable output lengths and writing styles. Adding one ICH E2D-formatted example to every narrative prompt anchors the model to third-person clinical register, temporal structure, and appropriate length (3–5 sentences), making the training labels more consistent and easier for Qwen3-14B to learn.

---

## Configuration Reference

### Fetch size (notebook 01)

```python
TARGET_RECORDS = 10000   # Recommended: 5000–10000 for a good training set
```

Increasing beyond 10,000 may hit openFDA's daily rate limit. With a free API key the limit is higher — add `api_key=YOUR_KEY` to the params dict.

### Generation concurrency (notebook 03)

```python
CONCURRENCY = 8   # Concurrent requests to vLLM
```

Safe range: 4–16. If vLLM returns timeout errors, reduce. If GPU utilization is below 80%, increase.

### LoRA rank (train.py)

```python
LORA_R     = 64    # Higher rank = more capacity but more VRAM
LORA_ALPHA = 128   # Keep alpha = 2 × r
```

For a smaller dataset (< 2,000 examples), consider `r=32, alpha=64` to reduce overfitting risk.

### Learning rate (train.py)

```python
LEARNING_RATE = 2e-4
```

If training loss oscillates, lower to `1e-4`. If convergence is very slow after 100 steps, try `3e-4`.

---

## Troubleshooting

### vLLM not reachable in notebook 03

```
✗ vLLM not reachable: [Errno 111] Connection refused
```

Start vLLM in a separate terminal and wait for `Application startup complete.`:
```bash
vllm serve Qwen/Qwen3-32B --dtype bfloat16 --max-model-len 4096 --port 8000
```

### `drug_labels` not defined (test cell fails)

The FDA label fetch cell must be run before the test cell and the main generation loop. If `drug_labels` is missing, re-run the FDA label fetch cell. It loads from `data/drug_labels.json` cache if the file exists, so it completes in seconds on subsequent runs.

### MEDDRA.xlsx column not found

```text
⚠ REQUIRED columns not found: ['pt_code', 'soc_name']
```

Find the actual column header in the sheet inspection output and set it manually:

```python
COL["pt_code"] = "Preferred Term Code"
COL["soc_name"] = "System Organ Class"
```

### Out of memory during training

1. Reduce `PER_DEVICE_BATCH_SIZE` from 2 to 1
2. Increase `GRADIENT_ACCUMULATION` from 16 to 32 to keep effective batch size constant
3. Reduce `MAX_SEQ_LENGTH` from 2048 to 1024

### Training loss not decreasing

- Verify `DataCollatorForCompletionOnlyLM` is finding the response template. The response template for Qwen3 ChatML is `<|im_start|>assistant\n`.
- Verify the dataset format is `{"text": "..."}`, not `{"messages": [...]}`.
- Try lowering learning rate to `1e-4`.

### JSON parse error at inference

The model occasionally outputs text before the opening brace on ambiguous narratives. The `extract_json` function in `infer.py` searches for the first `{...}` block. If this fails, the raw output is returned under `"raw_output"` for inspection.

### openFDA API rate limit

```text
HTTP 429 Too Many Requests
```

Increase `SLEEP_BETWEEN` to 1.0–2.0 seconds, or register for a free API key at the openFDA website.

---

## Glossary

| Term | Definition |
| --- | --- |
| ADR | Adverse Drug Reaction |
| AE | Adverse Event |
| FAERS | FDA Adverse Event Reporting System |
| HLGT | High Level Group Term (MedDRA level 4) |
| HLT | High Level Term (MedDRA level 3) |
| ICSR | Individual Case Safety Report |
| LLT | Lowest Level Term (MedDRA level 1, most specific) |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning method |
| MedDRA | Medical Dictionary for Regulatory Activities |
| PT | Preferred Term (MedDRA level 2, standard reporting level) |
| PV | Pharmacovigilance |
| ROCm | AMD's open compute platform (GPU software stack, analogous to CUDA) |
| SOC | System Organ Class (MedDRA level 5, broadest) |
| SFT | Supervised Fine-Tuning |
| WHO-UMC | World Health Organization Uppsala Monitoring Centre causality classification system |
