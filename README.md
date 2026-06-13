# AI-Led Medical Review Assistant

A fine-tuned LLM (Qwen3-14B + LoRA) that assists pharmacovigilance reviewers with four clinical tasks from a single unstructured clinical safety narrative:

1. **Seriousness Assessment** — Classifies events as Serious/Non-serious per ICH E2D criteria
2. **MedDRA Coding** — Maps adverse events to LLT, PT, and SOC with 8-digit codes
3. **Labelling Status** — Determines if an event is Expected or Unexpected
4. **Causality Assessment** — Computes Naranjo score + WHO-UMC category

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
- [Input / Output Format](#input--output-format)
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
5. Score the **causal relationship** between the drug and the event

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

Each term has a unique 8-digit numeric code. FAERS (the FDA adverse event database) stores reactions at the PT level.

---

## Architecture

The project is a two-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATASET CREATION                     │
│                                                                   │
│  openFDA FAERS API          MEDDRA.xlsx (Kaggle)                 │
│  (5000 AE reports)     +    (full hierarchy dict)                │
│       │                           │                              │
│       ▼                           ▼                              │
│  Structured records          PT → LLT/SOC lookup                 │
│  (drug, reaction PT,         (fuzzy-matched)                     │
│   seriousness, demographics)      │                              │
│       │                           │                              │
│       └───────────────┬───────────┘                              │
│                       ▼                                          │
│              Qwen3-32B via vLLM                                  │
│              Generates (per record):                             │
│              • Clinical narrative  (if not in FAERS)             │
│              • Naranjo score + category                          │
│              • WHO-UMC causality category                        │
│              • Labelling status + evidence                       │
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

Qwen3-32B is used as a **teacher model** to generate training labels — specifically, the fields FAERS does not provide (narrative text, causality scores, labelling status). It runs in **non-thinking mode** for structured JSON output. On the MI300X (192 GB), it uses ~64 GB in BF16, leaving headroom for the fine-tuning job.

### Why LoRA and not full fine-tuning?

Full fine-tuning of a 14B-parameter model requires tracking model weights, gradients, and two optimizer momentum tensors, pushing memory requirements above 160 GB. LoRA injects small trainable matrices (rank-64) into the attention and MLP layers — only ~1–2% of parameters are trained. For Qwen3-14B in BF16, this uses approximately 35 GB of VRAM, well within the MI300X budget.

### Why response-only loss masking?

During instruction tuning, computing the cross-entropy loss over the entire sequence — including the system prompt and user instruction — wastes gradient updates on tokens the model already handles well. Masking the prompt tokens focuses 100% of training signal on the assistant's JSON output, improving convergence speed and output structure quality.

---

## Project Structure

```
TCS-Hackathon/
│
├── MEDDRA.xlsx                  ← MedDRA hierarchy dictionary (Kaggle dataset)
│
├── 01_fetch_faers.ipynb         ← Fetch adverse event reports from openFDA API
├── 02_explore_meddra.ipynb      ← Inspect MEDDRA.xlsx and build PT → hierarchy lookup
├── 03_generate_dataset.ipynb    ← Qwen3-32B data generation (async, checkpointed)
├── 04_format_dataset.ipynb      ← Validate, format, and split training data
│
├── train/
│   └── train.py                 ← LoRA fine-tuning script (TRL + PEFT)
│
├── infer/
│   └── infer.py                 ← Inference: interactive, single, or batch mode
│
├── data/                        ← Created automatically during pipeline
│   ├── faers_raw.json           ← Raw FAERS records (output of notebook 01)
│   ├── meddra_lookup.json       ← PT name → hierarchy dict (output of notebook 02)
│   ├── dataset_raw.json         ← Generated records with all fields (output of notebook 03)
│   ├── train.jsonl              ← Final training set in ChatML format
│   ├── val.jsonl                ← Validation set
│   ├── train_chatml.jsonl       ← Training set in messages format (for evaluation)
│   └── val_chatml.jsonl         ← Val set in messages format
│
├── train/output/                ← Created by train.py
│   ├── checkpoint-*/            ← Intermediate checkpoints
│   └── final_adapter/           ← Final LoRA adapter weights
│
└── requirements.txt
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

This must be installed separately before the other requirements. Check the exact ROCm version on your machine with `rocminfo | grep "ROCm Version"`, then:

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

vLLM is installed separately because it has its own ROCm wheel:

```bash
pip install vllm
```

Verify:
```bash
vllm --version
```

### 4. Download Qwen3 models from HuggingFace

```bash
# Teacher model for data generation (used in notebook 03)
huggingface-cli download Qwen/Qwen3-32B

# Base model for fine-tuning (used in train.py and infer.py)
huggingface-cli download Qwen/Qwen3-14B
```

Or set `HF_HUB_OFFLINE=0` and let the scripts download automatically on first run.

### 5. Place the MedDRA dataset

The `MEDDRA.xlsx` file (from [Kaggle: e0xextazy/meddra](https://www.kaggle.com/datasets/e0xextazy/meddra)) should be in the project root:

```
TCS-Hackathon/
└── MEDDRA.xlsx   ← must be here
```

---

## Pipeline Walkthrough

### Step 1: Fetch FAERS Data

**Notebook:** `01_fetch_faers.ipynb`

Opens the [openFDA Drug Event API](https://open.fda.gov/apis/drug/event/) and fetches adverse event reports. No API key is required. The notebook paginates through results and saves a clean JSON record for each report.

**What it extracts per record:**

| Field | Description |
|-------|-------------|
| `drug_name` | Suspect drug (first drug with characterization = "1") |
| `reaction_pt` | MedDRA Preferred Term name of the adverse event |
| `serious` | `true` if the event meets any seriousness criterion |
| `seriousness_criteria` | List: Death, Hospitalization, Life-threatening, Disability, Congenital Anomaly, Medically Significant |
| `patient_age` | Patient age at onset |
| `patient_sex` | male / female / unknown |
| `reaction_outcome` | Recovered, Recovering, Not Recovered, Fatal, Unknown |
| `narrative` | Free-text clinical narrative (present in ~30–40% of records) |
| `has_narrative` | Boolean flag — records with `false` get narratives generated in step 3 |

**Configuration (top of notebook):**

```python
TARGET_RECORDS = 5000    # total records to fetch; increase for larger dataset
BATCH_SIZE     = 100     # records per API request
SLEEP_BETWEEN  = 0.5     # seconds between requests (respect rate limits)
```

**Checkpointing:** The file `data/faers_raw.json` is saved every 500 records. If the cell is interrupted and re-run, it resumes from the last saved state automatically.

**Expected output:** `data/faers_raw.json`

---

### Step 2: Build MedDRA Lookup

**Notebook:** `02_explore_meddra.ipynb`

Reads `MEDDRA.xlsx` and builds a dictionary mapping every PT name to its full MedDRA hierarchy (LLT, HLT, HLGT, SOC with codes).

**How the column detection works:**

The notebook inspects every sheet in `MEDDRA.xlsx` and auto-detects columns by matching keywords. The cell prints a table like:

```
llt_code     → LLT_CODE       ✓
llt_name     → LLT_NAME       ✓
pt_code      → PT_CODE        ✓
pt_name      → PT_NAME        ✓
soc_code     → SOC_CODE       ✓
soc_name     → SOC_NAME       ✓
```

If any required column shows `✗ NOT FOUND`, manually set the column name in the `COL` dictionary before proceeding. For example:

```python
COL["pt_name"] = "Preferred Term"   # exact column header from the file
```

**Fuzzy matching:**

FAERS PT names sometimes differ slightly from MedDRA dictionary entries (e.g., different capitalization, British vs. American spelling). The notebook uses `rapidfuzz` for fuzzy matching with a default threshold of 80. A test cell shows match results for common terms like "Bronchospasm", "Anaphylactic reaction", and "Dyspnoea".

**Expected output:** `data/meddra_lookup.json`

---

### Step 3: Generate Dataset with Qwen3-32B

**Notebook:** `03_generate_dataset.ipynb`

This is the most compute-intensive step. Qwen3-32B generates the fields that FAERS and the MedDRA dictionary do not provide.

#### Start vLLM first

In a **separate terminal**, before running this notebook:

```bash
vllm serve Qwen/Qwen3-32B \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --port 8000
```

Wait for this message before proceeding:
```
INFO:     Application startup complete.
```

The notebook includes a health-check cell that confirms vLLM is reachable and lists the loaded model.

#### What Qwen3-32B generates per record

**Narrative generation** (for records where `has_narrative = false`):

The model writes a 3–6 sentence clinical narrative from structured FAERS fields (drug, reaction, demographics, outcome). Temperature is set to 0.5 for some variation across similar cases.

Example prompt → output:
```
Drug: METFORMIN, Reaction: Lactic acidosis, Patient: 67y female
→ "A 67-year-old female patient with type 2 diabetes mellitus was prescribed
   Metformin 1000 mg twice daily. Approximately three weeks after initiating
   therapy, she presented to the emergency department with dyspnoea, generalized
   weakness, and abdominal pain..."
```

**Causality and labelling assessment** (for all records):

A single structured prompt asks the model to return:

```json
{
  "naranjo_score": 6,
  "naranjo_category": "Probable",
  "who_umc_category": "Probable/Likely",
  "labelling_status": "Expected",
  "labelling_evidence": "Lactic acidosis is listed as a serious warning in the Metformin prescribing information."
}
```

> **Thinking mode is disabled** (`enable_thinking: false`) for all Qwen3-32B calls. Thinking mode produces verbose reasoning chains which are incompatible with JSON-only output requirements and wastes inference compute.

#### Concurrency and performance

The notebook uses `asyncio` with a semaphore to run 8 concurrent requests to vLLM. On the MI300X, this typically processes 800–1200 records per hour.

**Concurrency setting:**
```python
CONCURRENCY = 8   # increase to 12-16 if vLLM throughput allows
```

**Checkpointing:** Results are written to `data/dataset_raw.json` every 200 records. Re-running the notebook skips already-processed records automatically (matching by FAERS report ID).

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
| Naranjo category | Must be one of: `Definite`, `Probable`, `Possible`, `Doubtful` |
| WHO-UMC category | Must be one of the 6 standard categories |
| Labelling status | Must be `"Expected"` or `"Unexpected"` |
| Naranjo score | Must be an integer between -5 and 13 |

Typical rejection rate is 5–15%, mostly from malformed JSON responses.

#### ChatML format

Each training example is formatted as:

```
<|im_start|>system
You are an expert pharmacovigilance medical reviewer...<|im_end|>
<|im_start|>user
Analyze the following clinical safety narrative...

Narrative:
A 54-year-old male was admitted to the hospital...<|im_end|>
<|im_start|>assistant
{
  "seriousness": "Serious",
  "seriousness_criteria": ["Hospitalization"],
  "meddra_llt": "Jaundice",
  "meddra_llt_code": 10023126,
  ...
}<|im_end|>
```

**Expected outputs:**

| File | Format | Use |
|------|--------|-----|
| `data/train.jsonl` | `{"text": "<full ChatML string>"}` | SFTTrainer input |
| `data/val.jsonl` | Same | Validation during training |
| `data/train_chatml.jsonl` | `{"messages": [...]}` | Evaluation scripts |
| `data/val_chatml.jsonl` | Same | Evaluation scripts |

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
| `r` (rank) | 64 | Controls adapter capacity; 64 is sufficient for multi-task clinical JSON |
| `alpha` | 128 | Scaling factor = alpha/r = 2.0; accelerates convergence |
| `dropout` | 0.05 | Light regularization |
| Target modules | All 7 linear layers | Ensures both attention and feed-forward layers learn the task |
| Trainable params | ~1–2% of total | ~160M parameters out of 14.8B |

**Target modules explained:**
- `q_proj`, `k_proj`, `v_proj`, `o_proj` — query, key, value, output projections in self-attention
- `gate_proj`, `up_proj`, `down_proj` — the three linear layers in each SwiGLU MLP block

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

#### Expected VRAM and time

- **VRAM:** ~35 GB (model weights ~28 GB + gradients + optimizer states for LoRA params only)
- **Training time:** ~1.5–2 hours for 3,000 examples × 3 epochs on MI300X

#### Output

The final LoRA adapter is saved to `train/output/final_adapter/`. This directory contains:

```
final_adapter/
├── adapter_config.json          ← LoRA configuration
├── adapter_model.safetensors    ← Trained adapter weights (~500 MB)
├── tokenizer.json               ← Tokenizer (copy from base model)
├── tokenizer_config.json
└── training_config.json         ← Hyperparameter record
```

The base model weights are not duplicated. At inference time, both the base model and this adapter are loaded together.

---

### Step 6: Run Inference

**Script:** `infer/infer.py`

The inference script loads Qwen3-14B + the LoRA adapter and runs the medical review.

#### Interactive mode (default)

```bash
cd infer
python infer.py
```

Paste a narrative, press Enter twice, and receive the assessment:

```
===================================================================
MEDICAL REVIEW ASSESSMENT
===================================================================
  Seriousness         : Serious
  Criteria            : Hospitalization
  MedDRA LLT          : Jaundice [10023126]
  MedDRA PT           : Jaundice [10023126]
  MedDRA SOC          : Hepatobiliary disorders [10019805]
  Labelling Status    : Unexpected
  Labelling Evidence  : Review of Drug X prescribing information shows no documentation of jaundice.
  Naranjo Score       : 6 (Probable)
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

Where `narratives.txt` has one narrative per line. Results are written as JSONL, one JSON object per line.

---

## Input / Output Format

### Input

Any unstructured clinical safety narrative in plain English. Examples:

- Spontaneous reports from patients or healthcare providers
- Clinical trial adverse event narratives
- Case reports from medical literature

Recommended length: 2–10 sentences. Very short inputs (< 2 sentences) may produce lower-quality assessments.

### Output

The model returns a JSON object with the following schema:

```json
{
  "seriousness": "Serious",
  "seriousness_criteria": ["Hospitalization", "Life-threatening"],

  "meddra_llt": "Anaphylactic reaction",
  "meddra_llt_code": 10002198,
  "meddra_pt": "Anaphylactic reaction",
  "meddra_pt_code": 10002198,
  "meddra_soc": "Immune system disorders",
  "meddra_soc_code": 10021428,

  "labelling_status": "Expected",
  "labelling_evidence": "Anaphylaxis is listed as a known serious adverse reaction in the Drug Y prescribing information.",

  "naranjo_score": 7,
  "naranjo_category": "Probable",
  "who_umc_category": "Probable/Likely"
}
```

**Field definitions:**

| Field | Type | Values |
|-------|------|--------|
| `seriousness` | string | `"Serious"` or `"Non-serious"` |
| `seriousness_criteria` | array | Subset of: Death, Hospitalization, Life-threatening, Disability/Incapacity, Congenital Anomaly, Medically Significant |
| `meddra_llt` | string | MedDRA Lowest Level Term name |
| `meddra_llt_code` | integer | 8-digit MedDRA LLT code |
| `meddra_pt` | string | MedDRA Preferred Term name |
| `meddra_pt_code` | integer | 8-digit MedDRA PT code |
| `meddra_soc` | string | MedDRA System Organ Class name |
| `meddra_soc_code` | integer | 8-digit MedDRA SOC code |
| `labelling_status` | string | `"Expected"` or `"Unexpected"` |
| `labelling_evidence` | string | One-sentence rationale |
| `naranjo_score` | integer | 0–13 (higher = stronger causal evidence) |
| `naranjo_category` | string | `"Definite"` (≥9), `"Probable"` (5–8), `"Possible"` (1–4), `"Doubtful"` (≤0) |
| `who_umc_category` | string | `"Certain"`, `"Probable/Likely"`, `"Possible"`, `"Unlikely"`, `"Conditional/Unclassified"`, `"Unassessable"` |

---

## Configuration Reference

### Fetch size (notebook 01)

```python
TARGET_RECORDS = 5000   # Recommended: 5000-10000 for a good training set
```

Increasing beyond 10,000 may hit openFDA's daily rate limit without an API key (1,000 requests/day). With a free API key (`api_key` parameter), the limit is much higher.

### Generation concurrency (notebook 03)

```python
CONCURRENCY = 8   # Concurrent requests to vLLM
```

Safe range: 4–16. If vLLM returns timeout errors, reduce this. If GPU utilization is below 80%, increase it.

### LoRA rank (train.py)

```python
LORA_R     = 64    # Higher rank = more capacity but more VRAM
LORA_ALPHA = 128   # Keep alpha = 2 × r for this task
```

For a smaller dataset (< 2,000 examples), consider `r=32, alpha=64` to reduce overfitting risk.

### Learning rate (train.py)

```python
LEARNING_RATE = 2e-4   # Standard for chat/instruct models
```

If training loss oscillates instead of decreasing smoothly, lower to `1e-4`. If convergence is very slow (loss barely moving after 100 steps), try `3e-4`.

---

## Troubleshooting

### vLLM not reachable in notebook 03

```
✗ vLLM not reachable: [Errno 111] Connection refused
```

vLLM is not running. Open a separate terminal and start it:
```bash
vllm serve Qwen/Qwen3-32B --dtype bfloat16 --max-model-len 4096 --port 8000
```
Wait for `Application startup complete.` before retrying.

### MEDDRA.xlsx column not found

```
⚠ REQUIRED columns not found: ['pt_code', 'soc_name']
```

The auto-detector did not recognize the column names. Look at the output of the sheet inspection cells, find the actual column header, and set it manually:
```python
COL["pt_code"] = "Preferred Term Code"   # use the exact header string from the file
COL["soc_name"] = "System Organ Class"
```

### Out of memory during training

If `train.py` raises a CUDA/ROCm OOM error:

1. Reduce `PER_DEVICE_BATCH_SIZE` from 2 to 1
2. Increase `GRADIENT_ACCUMULATION` from 16 to 32 to keep effective batch size the same
3. Reduce `MAX_SEQ_LENGTH` from 2048 to 1024

### Training loss not decreasing

- Check that `DataCollatorForCompletionOnlyLM` is finding the response template. Add a debug print of the first batch's `labels` tensor — non-masked tokens should be non-(-100).
- Verify the dataset has `{"text": "..."}` format, not `{"messages": [...]}`.
- Try lowering learning rate to `1e-4`.

### JSON parse error at inference

The model occasionally outputs text before the JSON opening brace, especially on short or ambiguous narratives. The `extract_json` function in `infer.py` handles this by searching for the first `{...}` block. If this still fails, the raw output is returned under the key `"raw_output"` for inspection.

### openFDA API rate limit

If requests start returning `HTTP 429`:
- Reduce `SLEEP_BETWEEN` to 1.0–2.0 seconds
- Or register for a free API key at https://open.fda.gov/apis/authentication/ and add `api_key=YOUR_KEY` to the params dict in `fetch_faers_batch`

---

## Glossary

| Term | Definition |
|------|------------|
| ADR | Adverse Drug Reaction |
| AE | Adverse Event |
| FAERS | FDA Adverse Event Reporting System |
| HLGT | High Level Group Term (MedDRA level 4) |
| HLT | High Level Term (MedDRA level 3) |
| ICSR | Individual Case Safety Report |
| LLT | Lowest Level Term (MedDRA level 1, most specific) |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning method |
| MedDRA | Medical Dictionary for Regulatory Activities |
| Naranjo Scale | 10-question algorithm producing a probability score for ADR causality |
| PT | Preferred Term (MedDRA level 2, standard reporting level) |
| PV | Pharmacovigilance |
| QLoRA | Quantized LoRA — LoRA with 4-bit base model quantization |
| ROCm | AMD's open compute platform (GPU software stack, analogous to CUDA) |
| SOC | System Organ Class (MedDRA level 5, broadest) |
| SFT | Supervised Fine-Tuning |
| WHO-UMC | World Health Organization Uppsala Monitoring Centre causality classification system |
