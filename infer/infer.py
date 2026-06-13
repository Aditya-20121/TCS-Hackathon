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

# ── Generation config ─────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 600
TEMPERATURE    = 0.1    # low = deterministic structured output
TOP_P          = 0.9

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


def build_prompt(narrative: str) -> str:
    """Build the ChatML prompt for a clinical narrative."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{INSTRUCTION}\n\nNarrative:\n{narrative}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def analyze(narrative: str, model, tokenizer) -> dict:
    """Run inference and return parsed JSON output."""
    prompt = build_prompt(narrative.strip())
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

    # Remove trailing <|im_end|> if present
    raw_text = raw_text.replace("<|im_end|>", "").strip()

    # Parse JSON
    try:
        result = json.loads(raw_text)
        result["_meta"] = {
            "inference_time_s": round(elapsed, 2),
            "raw_tokens": len(new_tokens),
        }
        return result
    except json.JSONDecodeError:
        return {
            "error": "JSON parse failed",
            "raw_output": raw_text,
            "_meta": {"inference_time_s": round(elapsed, 2)},
        }


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
