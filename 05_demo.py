"""
05_demo.py — Medical Review Assistant Interactive Demo

Loads the fine-tuned Qwen3-14B + LoRA adapter and provides a Gradio web interface
for demonstrating AI-assisted pharmacovigilance case assessment.

Usage:
    python 05_demo.py
    python 05_demo.py --adapter train/output/final_adapter
    python 05_demo.py --share          # public Gradio link for remote demo
    python 05_demo.py --cpu            # force CPU inference
"""

import json
import re
import time
import argparse
from pathlib import Path

import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths & generation config ──────────────────────────────────────────────────
ROOT          = Path(__file__).parent
ADAPTER_PATH  = ROOT / "train" / "output" / "final_adapter"
BASE_MODEL_ID = "Qwen/Qwen3-14B"
MEDDRA_LOOKUP = ROOT / "data" / "meddra_lookup.json"

MAX_NEW_TOKENS = 600
TEMPERATURE    = 0.1
TOP_P          = 0.9

# ── Prompts — must match training format exactly ───────────────────────────────
SYSTEM_PROMPT = """You are an expert pharmacovigilance medical reviewer with deep knowledge of ICH E2D guidelines and MedDRA terminology.

Given a clinical safety narrative, return ONLY a valid JSON object with exactly these fields:

{
  "seriousness": <"Serious" | "Non-serious">,
  "seriousness_criteria": <array of zero or more from: "Death", "Hospitalization", "Life-threatening", "Disability/Incapacity", "Congenital Anomaly", "Medically Significant">,
  "meddra_pt": <MedDRA Preferred Term name as a string>,
  "meddra_soc": <MedDRA System Organ Class name as a string>,
  "labelling_status": <"Expected" | "Unexpected">,
  "labelling_evidence": <one sentence explaining why the event is listed or not listed in the drug label>,
}

Rules:
- seriousness is "Serious" if the narrative mentions death, hospitalisation, life-threatening condition, permanent disability, congenital anomaly, or a medically significant event requiring intervention.
- seriousness_criteria must be an empty array [] for Non-serious events.
- meddra_pt and meddra_soc must use exact MedDRA dictionary names (title case).
- labelling_status is "Expected" if the adverse event is listed in the drug's current prescribing information, otherwise "Unexpected".
- Output JSON only. No explanation, no markdown, no code block."""

INSTRUCTION = (
    "Analyze the following clinical safety narrative and return a structured JSON assessment "
    "covering seriousness, MedDRA coding, and labelling status."
)

# ── Few-shot examples ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "narrative": (
            "A 72-year-old male patient with type 2 diabetes mellitus was receiving metformin "
            "1000 mg twice daily. He was admitted to the emergency department presenting with "
            "severe nausea, vomiting, abdominal pain, and confusion. Laboratory results revealed "
            "elevated serum lactate consistent with lactic acidosis. Metformin was discontinued "
            "and the patient required hospitalization for supportive care, recovering fully after "
            "five days."
        ),
        "output": {
            "seriousness": "Serious",
            "seriousness_criteria": ["Hospitalization"],
            "meddra_pt": "Lactic acidosis",
            "meddra_soc": "Metabolism and nutrition disorders",
            "labelling_status": "Expected",
            "labelling_evidence": (
                "Lactic acidosis is listed as a black box warning in the metformin "
                "prescribing information due to risk of fatal and non-fatal cases."
            ),
        },
    },
    {
        "narrative": (
            "A 45-year-old female patient was initiated on lisinopril 10 mg once daily for "
            "essential hypertension. Approximately two weeks after starting treatment she "
            "developed a persistent dry, non-productive cough that was bothersome but did not "
            "require hospitalisation. Lisinopril was continued at the physician's discretion."
        ),
        "output": {
            "seriousness": "Non-serious",
            "seriousness_criteria": [],
            "meddra_pt": "Cough",
            "meddra_soc": "Respiratory, thoracic and mediastinal disorders",
            "labelling_status": "Expected",
            "labelling_evidence": (
                "Dry cough is a very common, well-documented adverse effect of ACE inhibitors "
                "including lisinopril and is explicitly listed in the adverse reactions section "
                "of the prescribing information."
            ),
        },
    },
]

# ── Pre-loaded demo narratives for the dropdown ───────────────────────────────
DEMO_NARRATIVES = {
    "── Select an example ──": "",
    "Amoxicillin — Anaphylaxis (Serious / Unexpected)": (
        "A 34-year-old female patient with no known drug allergies was prescribed amoxicillin "
        "500 mg three times daily for a community-acquired respiratory infection. Within "
        "15 minutes of taking the first dose she developed urticaria, angioedema of the lips "
        "and throat, severe bronchospasm, and hypotension (BP 80/50 mmHg). Emergency services "
        "were called and she received intramuscular epinephrine, intravenous antihistamines, "
        "and corticosteroids. She was admitted to the intensive care unit for 48 hours and "
        "discharged with full recovery."
    ),
    "Warfarin — Intracranial Haemorrhage (Serious / Expected)": (
        "A 78-year-old male patient with atrial fibrillation was receiving warfarin 5 mg daily "
        "with an INR target of 2.0–3.0. He was found unresponsive at home. Neuroimaging "
        "revealed a large left hemispheric intracranial haemorrhage with midline shift. "
        "His INR on admission was 4.8. Warfarin was reversed with vitamin K and prothrombin "
        "complex concentrate. Despite neurosurgical intervention, the patient sustained "
        "permanent neurological disability."
    ),
    "Atorvastatin — Myalgia (Non-serious / Expected)": (
        "A 58-year-old male patient with hypercholesterolaemia was treated with atorvastatin "
        "40 mg once daily. After three months of therapy he reported bilateral proximal muscle "
        "aching and weakness in the thighs and upper arms, particularly after physical activity. "
        "Creatine kinase levels were mildly elevated at 320 U/L (reference < 200 U/L). "
        "The event was non-serious. Atorvastatin dose was reduced to 20 mg with subsequent "
        "improvement in symptoms."
    ),
    "Adalimumab — Injection Site Reaction (Non-serious / Expected)": (
        "A 42-year-old female patient with moderately severe plaque psoriasis was initiated on "
        "adalimumab 80 mg subcutaneously at week 0 followed by 40 mg every other week. "
        "Following the second injection she noticed mild erythema, induration, and pruritus "
        "at the injection site measuring approximately 3 cm in diameter. Symptoms resolved "
        "within 48 hours without intervention. Adalimumab was continued without modification."
    ),
    "Ibuprofen — GI Haemorrhage (Serious / Expected)": (
        "A 65-year-old male patient with osteoarthritis was self-medicating with ibuprofen "
        "600 mg three times daily without a proton pump inhibitor for six weeks. He presented "
        "to the emergency department with haematemesis and melaena. Upper GI endoscopy revealed "
        "a large gastric ulcer with active oozing requiring endoscopic haemostasis and blood "
        "transfusion. He was hospitalised for five days. Ibuprofen was permanently discontinued."
    ),
    "Clozapine — Agranulocytosis (Serious / Expected)": (
        "A 29-year-old male patient with treatment-resistant schizophrenia had been receiving "
        "clozapine 300 mg daily for eight months. During routine weekly haematological "
        "monitoring his absolute neutrophil count was found to be 400 cells/μL, consistent "
        "with agranulocytosis. He was immediately hospitalised, clozapine was discontinued, "
        "and granulocyte colony-stimulating factor was administered. The neutrophil count "
        "recovered to normal range after ten days."
    ),
}


# ── MedDRA lookup (loaded once at startup) ────────────────────────────────────
_meddra_db: dict | None = None


def _ensure_meddra():
    global _meddra_db
    if _meddra_db is None:
        with open(MEDDRA_LOOKUP) as f:
            _meddra_db = json.load(f)


def lookup_meddra(pt_name: str) -> dict:
    _ensure_meddra()
    entry = _meddra_db.get(pt_name.lower(), {})
    return {
        "pt_code":  entry.get("meddra_pt_code"),
        "soc_name": entry.get("meddra_soc"),
        "soc_code": entry.get("meddra_soc_code"),
    }


# ── Global model handles ───────────────────────────────────────────────────────
_model     = None
_tokenizer = None


def load_model(
    base_model_id: str = BASE_MODEL_ID,
    adapter_path: Path = ADAPTER_PATH,
    force_cpu: bool = False,
) -> None:
    global _model, _tokenizer

    print(f"\n[1/3] Loading tokenizer from: {adapter_path}")
    tok = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype      = torch.float32 if force_cpu else torch.bfloat16
    device_map = "cpu"         if force_cpu else "auto"

    print(f"[2/3] Loading base model: {base_model_id}  (dtype={dtype})")
    mdl = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"[3/3] Loading LoRA adapter from: {adapter_path}")
    mdl = PeftModel.from_pretrained(mdl, str(adapter_path))
    mdl.eval()

    if torch.cuda.is_available() and not force_cpu:
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Model ready on GPU. VRAM used: {vram_gb:.1f} GB\n")
    else:
        print("Model ready (CPU).\n")

    _model, _tokenizer = mdl, tok


# ── Prompt building ────────────────────────────────────────────────────────────

def build_prompt(narrative: str) -> str:
    """Same prompt as eval.py — fair and consistent across all inference paths."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{INSTRUCTION}\n\nNarrative: {narrative.strip()}\n\nOutput:"},
    ]
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_output(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<|im_end|>", "").strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def run_inference(narrative: str) -> tuple[dict | None, float, int]:
    """Returns (parsed_result, elapsed_seconds, new_token_count)."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model is not loaded.")

    prompt = build_prompt(narrative)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=(TEMPERATURE > 0),
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    result   = parse_output(raw_text)

    if result is not None:
        codes = lookup_meddra(result.get("meddra_pt", ""))
        result["meddra_pt_code"]  = codes["pt_code"]
        result["meddra_soc_code"] = codes["soc_code"]
        if codes["soc_name"]:
            result["meddra_soc"] = codes["soc_name"]

    return result, elapsed, len(new_tokens)


# ── HTML rendering helpers ─────────────────────────────────────────────────────

def _pill(text: str, fg: str, bg: str, bold: bool = True, size: str = "0.78rem") -> str:
    weight = "700" if bold else "500"
    return (
        f'<span style="display:inline-block;padding:3px 11px;border-radius:9999px;'
        f'background:{bg};color:{fg};font-weight:{weight};font-size:{size};'
        f'letter-spacing:0.03em;white-space:nowrap;">{text}</span>'
    )


def _section_label(text: str) -> str:
    return (
        f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;'
        f'color:#94a3b8;text-transform:uppercase;margin-bottom:10px;">{text}</div>'
    )


def _card(title: str, body: str, accent: str = "#e2e8f0") -> str:
    return (
        f'<div style="background:#ffffff;border:1px solid {accent};border-radius:14px;'
        f'padding:18px 20px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
        f'{_section_label(title)}'
        f'{body}'
        f'</div>'
    )


def _monobadge(code) -> str:
    if not code:
        return '<span style="font-size:0.78rem;color:#cbd5e1;">—</span>'
    return (
        f'<span style="font-family:\'JetBrains Mono\',\'Fira Code\',monospace;font-size:0.78rem;'
        f'color:#475569;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;'
        f'padding:3px 8px;white-space:nowrap;">{code}</span>'
    )


# ── Main assessment renderer ───────────────────────────────────────────────────

def render_assessment(result: dict, elapsed: float, n_tokens: int) -> str:
    serious  = result.get("seriousness", "Unknown")
    criteria = result.get("seriousness_criteria", [])
    pt       = result.get("meddra_pt", "—")
    pt_code  = result.get("meddra_pt_code")
    soc      = result.get("meddra_soc", "—")
    soc_code = result.get("meddra_soc_code")
    label    = result.get("labelling_status", "—")
    evidence = result.get("labelling_evidence", "")

    # ── Seriousness card ──────────────────────────────────────────────────────
    if serious == "Serious":
        serious_pill = _pill("SERIOUS", "#991b1b", "#fee2e2", size="0.88rem")
        serious_border = "#fca5a5"
        serious_icon = (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
            'style="margin-right:8px;flex-shrink:0;" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 '
            '001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="#991b1b" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
    else:
        serious_pill = _pill("NON-SERIOUS", "#166534", "#dcfce7", size="0.88rem")
        serious_border = "#86efac"
        serious_icon = (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
            'style="margin-right:8px;flex-shrink:0;" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M22 11.08V12a10 10 0 11-5.93-9.14" stroke="#166534" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
            '<polyline points="22 4 12 14.01 9 11.01" stroke="#166534" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )

    if criteria:
        criteria_html = "".join(
            _pill(c, "#1e40af", "#dbeafe", bold=False) + "&ensp;"
            for c in criteria
        )
    else:
        criteria_html = (
            '<span style="color:#94a3b8;font-size:0.8rem;font-style:italic;">'
            'No ICH E2D criteria met</span>'
        )

    serious_header = (
        f'<div style="display:flex;align-items:center;margin-bottom:10px;">'
        f'{serious_icon}{serious_pill}'
        f'</div>'
    )
    serious_body = (
        f'{serious_header}'
        f'<div style="font-size:0.72rem;font-weight:600;color:#94a3b8;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">ICH E2D Criteria</div>'
        f'<div>{criteria_html}</div>'
    )

    # ── MedDRA card ───────────────────────────────────────────────────────────
    meddra_body = (
        # SOC row
        f'<div style="margin-bottom:10px;">'
        f'<div style="font-size:0.7rem;font-weight:600;color:#94a3b8;text-transform:uppercase;'
        f'letter-spacing:0.07em;margin-bottom:4px;">System Organ Class (SOC)</div>'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-weight:600;color:#1e293b;font-size:0.92rem;">{soc}</span>'
        f'{_monobadge(soc_code)}'
        f'</div>'
        f'</div>'
        # divider arrow
        f'<div style="display:flex;align-items:center;gap:6px;margin:8px 0;">'
        f'<div style="height:1px;flex:1;background:#e2e8f0;"></div>'
        f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M12 5v14M19 12l-7 7-7-7" stroke="#94a3b8" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'<div style="height:1px;flex:1;background:#e2e8f0;"></div>'
        f'</div>'
        # PT row
        f'<div>'
        f'<div style="font-size:0.7rem;font-weight:600;color:#94a3b8;text-transform:uppercase;'
        f'letter-spacing:0.07em;margin-bottom:4px;">Preferred Term (PT)</div>'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-weight:700;color:#1e293b;font-size:0.96rem;">{pt}</span>'
        f'{_monobadge(pt_code)}'
        f'</div>'
        f'</div>'
    )

    # ── Labelling card ────────────────────────────────────────────────────────
    if label == "Expected":
        label_pill   = _pill("EXPECTED", "#166534", "#dcfce7")
        label_border = "#86efac"
    else:
        label_pill   = _pill("UNEXPECTED", "#92400e", "#fef3c7")
        label_border = "#fcd34d"

    labelling_body = (
        f'<div style="margin-bottom:10px;">{label_pill}</div>'
        f'<div style="font-size:0.84rem;color:#475569;line-height:1.6;">{evidence}</div>'
    )

    # ── Card layout ───────────────────────────────────────────────────────────
    top_row = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">'
        + _card("Seriousness Assessment — ICH E2D", serious_body, accent=serious_border)
        + _card("Labelling Status", labelling_body, accent=label_border)
        + '</div>'
    )

    stats_bar = (
        f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:16px;'
        f'margin-top:10px;padding-top:8px;border-top:1px solid #f1f5f9;">'
        f'<span style="font-size:0.74rem;color:#94a3b8;">'
        f'<span style="font-weight:600;color:#64748b;">{elapsed:.1f}s</span> inference</span>'
        f'<span style="font-size:0.74rem;color:#94a3b8;">'
        f'<span style="font-weight:600;color:#64748b;">{n_tokens}</span> tokens generated</span>'
        f'<span style="font-size:0.74rem;color:#94a3b8;">Qwen3-14B + LoRA</span>'
        f'</div>'
    )

    return (
        '<div style="font-family:\'Inter\',system-ui,sans-serif;max-width:100%;">'
        + top_row
        + _card("MedDRA Coding", meddra_body)
        + stats_bar
        + '</div>'
    )


def render_error(message: str) -> str:
    return (
        '<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:14px;'
        'padding:20px 24px;color:#991b1b;font-family:\'Inter\',system-ui,sans-serif;">'
        '<div style="font-weight:700;font-size:0.9rem;margin-bottom:6px;">Assessment failed</div>'
        f'<div style="font-size:0.84rem;line-height:1.6;">{message}</div></div>'
    )


# ── Gradio callbacks ───────────────────────────────────────────────────────────

def analyze_narrative(narrative: str) -> tuple[str, str]:
    narrative = narrative.strip()
    if not narrative:
        return render_error("Please enter a clinical narrative."), ""
    if len(narrative) < 60:
        return render_error(
            "Narrative is too short. Please provide a complete clinical case description "
            "(patient demographics, drug, adverse event, outcome)."
        ), ""

    try:
        result, elapsed, n_tokens = run_inference(narrative)
    except RuntimeError as e:
        return render_error(str(e)), ""
    except Exception as e:
        return render_error(f"Inference error: {type(e).__name__}: {e}"), ""

    if result is None:
        return render_error(
            "The model did not return valid JSON. This may happen on very short or "
            "ambiguous narratives — try adding more clinical detail."
        ), ""

    html     = render_assessment(result, elapsed, n_tokens)
    raw_json = json.dumps(result, indent=2)
    return html, raw_json


def load_example(name: str) -> str:
    return DEMO_NARRATIVES.get(name, "")


def clear_inputs() -> tuple[str, str, str]:
    return "", _empty_state_html(), ""


# ── Static HTML snippets ───────────────────────────────────────────────────────

def _empty_state_html() -> str:
    fields = [
        ("Patient", "Age, sex, relevant medical history"),
        ("Drug", "Name, dose, frequency, route, indication"),
        ("Event", "Adverse event description, onset timing"),
        ("Outcome", "Resolution, hospitalisation, intervention"),
    ]
    rows = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;">'
        f'<div style="min-width:6px;height:6px;border-radius:50%;background:#cbd5e1;margin-top:6px;"></div>'
        f'<div><span style="font-size:0.8rem;font-weight:600;color:#64748b;">{k}:</span>'
        f'<span style="font-size:0.8rem;color:#94a3b8;"> {v}</span></div>'
        f'</div>'
        for k, v in fields
    )
    return (
        '<div style="font-family:\'Inter\',system-ui,sans-serif;padding:32px 24px;'
        'text-align:center;">'
        '<div style="width:48px;height:48px;border-radius:12px;background:#f1f5f9;'
        'display:flex;align-items:center;justify-content:center;margin:0 auto 16px;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M9 12h6M9 16h6M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 '
        '5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        '</div>'
        '<div style="font-size:0.9rem;font-weight:600;color:#64748b;margin-bottom:4px;">'
        'Enter a narrative and click Analyze</div>'
        '<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:20px;">'
        'A good narrative includes:</div>'
        f'<div style="text-align:left;background:#f8fafc;border-radius:10px;padding:14px 16px;">'
        f'{rows}</div>'
        '</div>'
    )


HEADER_HTML = """
<div style="text-align:center;padding:16px 0 8px;font-family:'Inter',system-ui,sans-serif;">
  <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:8px;">
    <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#1e40af,#1d4ed8);
         display:flex;align-items:center;justify-content:center;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2a10 10 0 100 20A10 10 0 0012 2zM12 8v4l3 3" stroke="white" stroke-width="2"
              stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M8 12h.01M12 16h.01M16 12h.01" stroke="white" stroke-width="2.5"
              stroke-linecap="round"/>
      </svg>
    </div>
    <h1 style="font-size:1.6rem;font-weight:800;color:#0f172a;margin:0;letter-spacing:-0.02em;">
      Medical Review Assistant
    </h1>
  </div>
  <p style="color:#64748b;font-size:0.85rem;margin:0 0 14px;">
    AI-assisted pharmacovigilance &nbsp;&middot;&nbsp; Qwen3-14B + LoRA fine-tuning &nbsp;&middot;&nbsp; TCS Hackathon 2026
  </p>
  <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
    <span style="background:#fee2e2;color:#991b1b;padding:4px 13px;border-radius:9999px;
         font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">SERIOUSNESS &middot; ICH E2D</span>
    <span style="background:#dcfce7;color:#166534;padding:4px 13px;border-radius:9999px;
         font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">MEDDRA PT + SOC</span>
    <span style="background:#fef3c7;color:#92400e;padding:4px 13px;border-radius:9999px;
         font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">LABELLING STATUS</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;margin-top:8px;font-size:0.74rem;color:#94a3b8;
     border-top:1px solid #e2e8f0;padding-top:12px;font-family:'Inter',system-ui,sans-serif;">
  For demonstration purposes only &nbsp;&middot;&nbsp; Not for clinical or regulatory use
  &nbsp;&middot;&nbsp; Outputs require qualified medical reviewer validation
</div>
"""

CUSTOM_CSS = """
.gr-button { font-weight: 600 !important; }
.gr-textbox textarea {
    font-size: 0.87rem !important;
    line-height: 1.65 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
.analyze-btn { font-size: 1rem !important; }
footer { display: none !important; }
"""


def create_interface() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="Medical Review Assistant",
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── Left panel: Input ──────────────────────────────────────────────
            with gr.Column(scale=5):
                gr.Markdown("#### Clinical Narrative")
                narrative_box = gr.Textbox(
                    label="",
                    placeholder=(
                        "Paste or type an adverse event narrative here.\n\n"
                        "Include: patient demographics · drug name + dose · "
                        "adverse event description · clinical outcome."
                    ),
                    lines=14,
                    max_lines=24,
                )
                with gr.Row():
                    example_dd = gr.Dropdown(
                        choices=list(DEMO_NARRATIVES.keys()),
                        value="── Select an example ──",
                        label="Load example case",
                        scale=4,
                        interactive=True,
                    )
                with gr.Row():
                    analyze_btn = gr.Button(
                        "Analyze Case",
                        variant="primary",
                        size="lg",
                        scale=3,
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

            # ── Right panel: Output ────────────────────────────────────────────
            with gr.Column(scale=6):
                gr.Markdown("#### Assessment Results")
                with gr.Tabs():
                    with gr.Tab("Visual Assessment"):
                        output_html = gr.HTML(value=_empty_state_html())
                    with gr.Tab("Raw JSON"):
                        raw_json_box = gr.Code(
                            language="json",
                            label="",
                            lines=22,
                            interactive=False,
                        )

        gr.HTML(FOOTER_HTML)

        # ── Event wiring ───────────────────────────────────────────────────────
        example_dd.change(
            fn=load_example,
            inputs=example_dd,
            outputs=narrative_box,
        )
        analyze_btn.click(
            fn=analyze_narrative,
            inputs=narrative_box,
            outputs=[output_html, raw_json_box],
        )
        narrative_box.submit(
            fn=analyze_narrative,
            inputs=narrative_box,
            outputs=[output_html, raw_json_box],
        )
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[narrative_box, output_html, raw_json_box],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Review Assistant Demo")
    parser.add_argument("--base_model", default=BASE_MODEL_ID,    help="HuggingFace model ID")
    parser.add_argument("--adapter",    default=str(ADAPTER_PATH), help="Path to LoRA adapter directory")
    parser.add_argument("--share",      action="store_true",       help="Create public Gradio share link")
    parser.add_argument("--port",       type=int, default=7860,    help="Local server port")
    parser.add_argument("--cpu",        action="store_true",       help="Force CPU (no GPU)")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"\nERROR: Adapter not found at: {adapter_path.resolve()}")
        print("Run  train/train.py  first to produce the fine-tuned adapter.")
        raise SystemExit(1)

    load_model(
        base_model_id=args.base_model,
        adapter_path=adapter_path,
        force_cpu=args.cpu,
    )

    demo = create_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None,
    )
