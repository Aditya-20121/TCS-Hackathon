"""
Medical Review Assistant — Inference Script

Loads Qwen3-14B with the LoRA adapter and analyzes clinical safety narratives.

Usage:
    cd infer

    # Interactive mode (type narratives one by one)
    python infer.py

    # Single narrative from argument
    python infer.py --narrative "A 54-year-old male developed jaundice after starting Drug X."

    # Batch mode from a text file (one narrative per line)
    python infer.py --input_file narratives.txt --output_file results.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
ADAPTER_PATH = ROOT / "train" / "output" / "final_adapter"
BASE_MODEL   = "Qwen/Qwen3-14B"
MEDDRA_LOOKUP = ROOT / "data" / "meddra_lookup.json"

# ── Generation config ─────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 600
TEMPERATURE    = 0.1    # low = deterministic structured output
TOP_P          = 0.9

SYSTEM_PROMPT = """You are an expert pharmacovigilance medical reviewer with deep knowledge of ICH E2D guidelines, MedDRA terminology, and WHO-UMC causality assessment.

Given a clinical safety narrative, return ONLY a valid JSON object with exactly these fields:

{
  "seriousness": <"Serious" | "Non-serious">,
  "seriousness_criteria": <array of zero or more from: "Death", "Hospitalization", "Life-threatening", "Disability/Incapacity", "Congenital Anomaly", "Medically Significant">,
  "meddra_pt": <MedDRA Preferred Term name as a string>,
  "meddra_soc": <MedDRA System Organ Class name as a string>,
  "labelling_status": <"Expected" | "Unexpected">,
  "labelling_evidence": <one sentence explaining why the event is listed or not listed in the drug label>,
  "who_umc_category": <"Certain" | "Probable/Likely" | "Possible" | "Unlikely" | "Conditional/Unclassified" | "Unassessable">
}

Rules:
- seriousness is "Serious" if the narrative mentions death, hospitalisation, life-threatening condition, permanent disability, congenital anomaly, or a medically significant event requiring intervention.
- seriousness_criteria must be an empty array [] for Non-serious events.
- meddra_pt and meddra_soc must use exact MedDRA dictionary names (title case).
- labelling_status is "Expected" if the adverse event is listed in the drug's current prescribing information, otherwise "Unexpected".
- who_umc_category follows WHO-UMC 2012 causality definitions.
- Output JSON only. No explanation, no markdown, no code block."""

INSTRUCTION = (
    "Analyze the following clinical safety narrative and return a structured JSON assessment "
    "covering seriousness, MedDRA coding, labelling status, and WHO-UMC causality."
)


# ── MedDRA lookup ─────────────────────────────────────────────────────────────
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


def load_model(base_model: str = BASE_MODEL, adapter_path: Path = ADAPTER_PATH):
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        padding_side="left",   # left-padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Model ready. VRAM: {vram:.1f} GB")
    else:
        print("Model ready (CPU mode).")

    return model, tokenizer


def build_prompt(narrative: str, tokenizer) -> str:
    """Build prompt using apply_chat_template with thinking disabled."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{INSTRUCTION}\n\nNarrative: {narrative.strip()}\n\nOutput:"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _extract_json(text: str) -> dict | None:
    import re
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


def analyze(narrative: str, model, tokenizer) -> dict:
    """Run inference and return parsed JSON output."""
    prompt = build_prompt(narrative.strip(), tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=(TEMPERATURE > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Decode only the new tokens (skip prompt)
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip thinking blocks and extract JSON robustly
    import re
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    raw_text = raw_text.replace("<|im_end|>", "").strip()

    result = _extract_json(raw_text)
    if result is not None:
        codes = lookup_meddra(result.get("meddra_pt", ""))
        result["meddra_pt_code"]  = codes["pt_code"]
        result["meddra_soc_code"] = codes["soc_code"]
        if codes["soc_name"]:
            result["meddra_soc"] = codes["soc_name"]
        result["_meta"] = {"inference_time_s": round(elapsed, 2), "raw_tokens": len(new_tokens)}
        return result
    return {"error": "JSON parse failed", "raw_output": raw_text, "_meta": {"inference_time_s": round(elapsed, 2)}}


def pretty_print(result: dict):
    """Print the result in a readable format."""
    print("\n" + "=" * 65)
    print("MEDICAL REVIEW ASSESSMENT")
    print("=" * 65)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        print(f"Raw output:\n{result.get('raw_output', '')}")
        return

    fields = [
        ("Seriousness",        result.get("seriousness", "—")),
        ("Criteria",           ", ".join(result.get("seriousness_criteria", [])) or "None"),
        ("MedDRA PT",          f"{result.get('meddra_pt','—')} [{result.get('meddra_pt_code','—')}]"),
        ("MedDRA SOC",         f"{result.get('meddra_soc','—')} [{result.get('meddra_soc_code','—')}]"),
        ("Labelling Status",   result.get("labelling_status", "—")),
        ("Labelling Evidence", result.get("labelling_evidence", "—")),
        ("WHO-UMC",            result.get("who_umc_category", "—")),
    ]

    for label, value in fields:
        print(f"  {label:<22}: {value}")

    meta = result.get("_meta", {})
    print(f"\n  [{meta.get('inference_time_s','?')}s | {meta.get('raw_tokens','?')} tokens]")
    print("=" * 65)


def interactive_mode(model, tokenizer):
    """Run interactive CLI loop."""
    print("\n" + "=" * 65)
    print("Medical Review Assistant — Interactive Mode")
    print("Type or paste a clinical narrative, then press Enter twice.")
    print("Type 'exit' to quit.")
    print("=" * 65)

    while True:
        print("\nNarrative (press Enter twice when done):")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().lower() == "exit":
                print("Goodbye.")
                return
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        narrative = "\n".join(lines).strip()
        if not narrative:
            continue

        result = analyze(narrative, model, tokenizer)
        pretty_print(result)


def batch_mode(input_file: str, output_file: str, model, tokenizer):
    """Process multiple narratives from a file."""
    input_path  = Path(input_file)
    output_path = Path(output_file)

    narratives = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Processing {len(narratives)} narratives from {input_path}...")

    with open(output_path, "w") as f:
        for i, narrative in enumerate(narratives, 1):
            print(f"  [{i}/{len(narratives)}] {narrative[:60]}...")
            result = analyze(narrative, model, tokenizer)
            result["_narrative"] = narrative
            f.write(json.dumps(result) + "\n")

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Medical Review Assistant Inference")
    parser.add_argument("--narrative",   type=str,  default=None, help="Single narrative to analyze")
    parser.add_argument("--input_file",  type=str,  default=None, help="File with one narrative per line")
    parser.add_argument("--output_file", type=str,  default="results.jsonl")
    parser.add_argument("--base_model",  type=str,  default=BASE_MODEL)
    parser.add_argument("--adapter",     type=str,  default=str(ADAPTER_PATH))
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run train/train.py first.")
        sys.exit(1)

    model, tokenizer = load_model(args.base_model, adapter_path)

    if args.narrative:
        result = analyze(args.narrative, model, tokenizer)
        pretty_print(result)
        print("\nJSON output:")
        print(json.dumps({k: v for k, v in result.items() if k != "_meta"}, indent=2))

    elif args.input_file:
        batch_mode(args.input_file, args.output_file, model, tokenizer)

    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
