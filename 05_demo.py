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

MAX_NEW_TOKENS = 600
TEMPERATURE    = 0.1
TOP_P          = 0.9

# ── Prompts — must match training format exactly ───────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert pharmacovigilance medical reviewer. "
    "Analyze clinical safety narratives and provide structured assessments "
    "according to ICH E2D guidelines and MedDRA coding standards."
)

INSTRUCTION = (
    "Analyze the following clinical safety narrative. "
    "Provide a structured medical review covering:\n"
    "1. Seriousness assessment with specific ICH E2D criteria\n"
    "2. MedDRA coding: Preferred Term (PT) and System Organ Class (SOC) with 8-digit codes\n"
    "3. Labelling status: Expected or Unexpected, with brief evidence\n"
    "4. Causality: WHO-UMC category\n"
    "Return your assessment as a JSON object."
)

# ── Few-shot examples ──────────────────────────────────────────────────────────
# Two representative cases bracketing the output space.
# Placed inside the user message so the fine-tuned model sees them as demonstrations
# and anchors its output format before generating the actual assessment.
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
            "meddra_pt_code": 10023525,
            "meddra_soc": "Metabolism and nutrition disorders",
            "meddra_soc_code": 10027433,
            "labelling_status": "Expected",
            "labelling_evidence": (
                "Lactic acidosis is listed as a black box warning in the metformin "
                "prescribing information due to risk of fatal and non-fatal cases."
            ),
            "who_umc_category": "Probable/Likely",
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
            "meddra_pt_code": 10011224,
            "meddra_soc": "Respiratory, thoracic and mediastinal disorders",
            "meddra_soc_code": 10038738,
            "labelling_status": "Expected",
            "labelling_evidence": (
                "Dry cough is a very common, well-documented adverse effect of ACE inhibitors "
                "including lisinopril and is explicitly listed in the adverse reactions section "
                "of the prescribing information."
            ),
            "who_umc_category": "Possible",
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

# ── WHO-UMC display colours ────────────────────────────────────────────────────
WHO_UMC_COLOUR = {
    "Certain":                  ("#166534", "#dcfce7"),
    "Probable/Likely":          ("#1e40af", "#dbeafe"),
    "Possible":                 ("#92400e", "#fef3c7"),
    "Unlikely":                 ("#991b1b", "#fee2e2"),
    "Conditional/Unclassified": ("#5b21b6", "#ede9fe"),
    "Unassessable":             ("#374151", "#f3f4f6"),
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
    """
    ChatML prompt with two few-shot demonstrations before the actual case.
    The few-shot block is placed inside the user message so the fine-tuned model
    sees it as context guidance before generating its JSON assessment.
    """
    shot_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        shot_block += (
            f"\n### Example {i}\n"
            f"Narrative:\n{ex['narrative']}\n\n"
            f"Assessment:\n{json.dumps(ex['output'], indent=2)}\n"
        )

    user_content = (
        f"{INSTRUCTION}"
        f"\n\n## Demonstration Examples\n"
        f"{shot_block}"
        f"\n---\n"
        f"## Case to Analyze\n\n"
        f"Narrative:\n{narrative.strip()}"
    )

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_output(text: str) -> dict | None:
    text = text.strip().replace("<|im_end|>", "").strip()
    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First JSON object anywhere in the text
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
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
    raw_text   = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    result     = parse_output(raw_text)

    return result, elapsed, len(new_tokens)


# ── HTML rendering helpers ─────────────────────────────────────────────────────

def _pill(text: str, fg: str, bg: str, bold: bool = True) -> str:
    weight = "700" if bold else "500"
    return (
        f'<span style="display:inline-block;padding:3px 12px;border-radius:9999px;'
        f'background:{bg};color:{fg};font-weight:{weight};font-size:0.82rem;">{text}</span>'
    )


def _card(title: str, body: str) -> str:
    return (
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'
        f'padding:16px 20px;margin-bottom:12px;">'
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:.09em;color:#94a3b8;'
        f'text-transform:uppercase;margin-bottom:10px;">{title}</div>'
        f'{body}'
        f'</div>'
    )


def render_assessment(result: dict, elapsed: float, n_tokens: int) -> str:
    serious  = result.get("seriousness", "Unknown")
    criteria = result.get("seriousness_criteria", [])
    pt       = result.get("meddra_pt", "—")
    pt_code  = result.get("meddra_pt_code", "—")
    soc      = result.get("meddra_soc", "—")
    soc_code = result.get("meddra_soc_code", "—")
    label    = result.get("labelling_status", "—")
    evidence = result.get("labelling_evidence", "")
    who_umc  = result.get("who_umc_category", "—")

    # ── Seriousness card ──────────────────────────────────────────────────────
    if serious == "Serious":
        serious_pill = _pill("⚠ SERIOUS", "#991b1b", "#fee2e2")
    else:
        serious_pill = _pill("✓ NON-SERIOUS", "#166534", "#dcfce7")

    if criteria:
        criteria_html = "".join(
            _pill(c, "#1e40af", "#dbeafe", bold=False) + "&ensp;"
            for c in criteria
        )
    else:
        criteria_html = (
            '<span style="color:#94a3b8;font-size:0.82rem;">'
            'No ICH E2D criteria met</span>'
        )

    seriousness_body = (
        f'<div style="margin-bottom:8px;">{serious_pill}</div>'
        f'<div style="font-size:0.82rem;color:#64748b;margin-top:6px;">'
        f'Criteria:&nbsp; {criteria_html}</div>'
    )

    # ── MedDRA card ───────────────────────────────────────────────────────────
    meddra_body = f"""
<table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
  <tr>
    <td style="padding:5px 0;color:#64748b;width:38%;">Preferred Term</td>
    <td style="padding:5px 0;font-weight:600;color:#1e293b;">{pt}</td>
    <td style="padding:5px 0;color:#94a3b8;font-family:monospace;font-size:0.78rem;text-align:right;">{pt_code}</td>
  </tr>
  <tr>
    <td style="padding:5px 0;color:#64748b;">System Organ Class</td>
    <td style="padding:5px 0;font-weight:600;color:#1e293b;">{soc}</td>
    <td style="padding:5px 0;color:#94a3b8;font-family:monospace;font-size:0.78rem;text-align:right;">{soc_code}</td>
  </tr>
</table>"""

    # ── Labelling card ────────────────────────────────────────────────────────
    if label == "Expected":
        label_pill = _pill("● EXPECTED", "#166534", "#dcfce7")
    else:
        label_pill = _pill("● UNEXPECTED", "#92400e", "#fef3c7")

    labelling_body = (
        f'<div style="margin-bottom:8px;">{label_pill}</div>'
        f'<div style="font-size:0.84rem;color:#475569;line-height:1.55;margin-top:6px;">'
        f'{evidence}</div>'
    )

    # ── WHO-UMC card ──────────────────────────────────────────────────────────
    fg, bg = WHO_UMC_COLOUR.get(who_umc, ("#374151", "#f3f4f6"))
    who_body = (
        f'<div style="font-size:1.05rem;font-weight:700;color:{fg};">{who_umc}</div>'
        f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">'
        f'Based on reported outcome and dechallenge data</div>'
    )

    return (
        '<div style="font-family:\'Inter\',system-ui,sans-serif;max-width:100%;">'
        + _card("Seriousness Assessment — ICH E2D", seriousness_body)
        + _card("MedDRA Coding", meddra_body)
        + _card("Labelling Status", labelling_body)
        + _card("Causality — WHO-UMC", who_body)
        + f'<div style="text-align:right;font-size:0.76rem;color:#94a3b8;margin-top:2px;">'
        f'⏱ {elapsed:.1f}s &nbsp;|&nbsp; {n_tokens} tokens generated</div>'
        + "</div>"
    )


def render_error(message: str) -> str:
    return (
        '<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:12px;'
        'padding:20px;color:#991b1b;font-family:sans-serif;">'
        f'<strong>Assessment failed</strong><br><br>{message}</div>'
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


# ── Gradio interface definition ────────────────────────────────────────────────

HEADER_HTML = """
<div style="text-align:center;padding:8px 0 4px;">
  <h1 style="font-size:1.75rem;font-weight:800;color:#1e293b;margin:0 0 4px;">
    Medical Review Assistant
  </h1>
  <p style="color:#64748b;font-size:0.88rem;margin:0 0 10px;">
    AI-assisted pharmacovigilance &nbsp;·&nbsp; Qwen3-14B + LoRA fine-tuning
  </p>
  <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
    <span style="background:#dbeafe;color:#1e40af;padding:2px 10px;border-radius:9999px;font-size:0.75rem;font-weight:600;">Seriousness · ICH E2D</span>
    <span style="background:#dcfce7;color:#166534;padding:2px 10px;border-radius:9999px;font-size:0.75rem;font-weight:600;">MedDRA Coding</span>
    <span style="background:#fef3c7;color:#92400e;padding:2px 10px;border-radius:9999px;font-size:0.75rem;font-weight:600;">Labelling Status</span>
    <span style="background:#ede9fe;color:#5b21b6;padding:2px 10px;border-radius:9999px;font-size:0.75rem;font-weight:600;">WHO-UMC Causality</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;margin-top:16px;font-size:0.76rem;color:#94a3b8;border-top:1px solid #e2e8f0;padding-top:12px;">
  For demonstration purposes only. Not for clinical or regulatory use. &nbsp;·&nbsp; TCS Hackathon 2026
</div>
"""


def create_interface() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="Medical Review Assistant",
        css=(
            ".gr-button { font-weight: 600; }"
            ".gr-textbox textarea { font-size: 0.87rem; line-height: 1.6; }"
        ),
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
                        "adverse event · outcome."
                    ),
                    lines=14,
                    max_lines=24,
                )
                with gr.Row():
                    example_dd = gr.Dropdown(
                        choices=list(DEMO_NARRATIVES.keys()),
                        value="── Select an example ──",
                        label="Load example case",
                        scale=3,
                        interactive=True,
                    )
                analyze_btn = gr.Button(
                    "Analyze Case  ▶",
                    variant="primary",
                    size="lg",
                )

            # ── Right panel: Output ────────────────────────────────────────────
            with gr.Column(scale=5):
                gr.Markdown("#### Assessment Results")
                output_html = gr.HTML(
                    value=(
                        '<div style="color:#94a3b8;font-family:sans-serif;'
                        'padding:60px 0;text-align:center;font-size:0.9rem;">'
                        'Enter a clinical narrative and click <strong>Analyze Case</strong>'
                        '</div>'
                    )
                )
                with gr.Accordion("Raw JSON", open=False):
                    raw_json_box = gr.Code(
                        language="json",
                        label="",
                        lines=18,
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
        # Also trigger on Shift+Enter in the textbox
        narrative_box.submit(
            fn=analyze_narrative,
            inputs=narrative_box,
            outputs=[output_html, raw_json_box],
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
