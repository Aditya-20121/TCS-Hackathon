"""
Evaluate the fine-tuned Qwen3-14B Medical Review Assistant on a held-out validation set.

Compares two configurations:
  base model   (no LoRA adapter)
  fine-tuned   (with LoRA adapter)

Usage (run_eval.sh handles both automatically):
    python eval.py --no_adapter --results_txt base.txt
    python eval.py              --results_txt finetuned.txt
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DEFAULT_VAL = ROOT / "data" / "val_chatml.jsonl"
ADAPTER_DIR = ROOT / "train" / "output" / "final_adapter"
BASE_MODEL  = "Qwen/Qwen3-14B"

LIMIT         = 50   # default samples to evaluate; override with --limit
PRINT_SAMPLES = 5    # print raw model output for the first N samples

# ── Base model prompt — explicit JSON schema for zero-shot structured output ───
BASE_SYSTEM_PROMPT = """You are an expert pharmacovigilance medical reviewer with deep knowledge of ICH E2D guidelines, MedDRA terminology, and WHO-UMC causality assessment.

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

def build_prompt(narrative: str, tokenizer) -> str:
    """Same prompt for both base and fine-tuned — ensures a fair, reportable comparison.
    enable_thinking=False suppresses Qwen3 <think> blocks for both models consistently."""
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Analyze the following clinical safety narrative and return a structured JSON assessment "
            "covering seriousness, MedDRA coding, labelling status, and WHO-UMC causality.\n"
            f"Narrative: {narrative.strip()}\n\nOutput:"
        )},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from model output robustly.

    Handles: plain JSON, markdown fences anywhere in text, preamble/postamble,
    Qwen3 <think> blocks, and nested braces via depth tracking.
    """
    import re

    # Strip Qwen3 thinking blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try markdown code fence first (handles ```json\n{...}\n``` anywhere)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back: find outermost { } using brace-depth tracking
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


def extract_narrative(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            for marker in ("Narrative:\n", "Narrative: "):
                idx = content.find(marker)
                if idx != -1:
                    return content[idx + len(marker):].strip()
    return ""


def extract_gold(record: dict) -> dict | None:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return extract_json(msg.get("content", ""))
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",   type=str, default=None,
                        help="Path to eval JSONL (default: data/val_chatml.jsonl)")
    parser.add_argument("--adapter",     type=str, default=str(ADAPTER_DIR))
    parser.add_argument("--base_model",  type=str, default=BASE_MODEL)
    parser.add_argument("--limit",       type=int, default=LIMIT,
                        help=f"Evaluate only the first N examples (default: {LIMIT})")
    parser.add_argument("--output",      type=str, default=None,
                        help="Save per-record results to this JSONL file")
    parser.add_argument("--results_txt", type=str, default=None,
                        help="Save the summary table to a plain-text file")
    parser.add_argument("--no_adapter",  action="store_true",
                        help="Run base model only (no LoRA adapter)")
    return parser.parse_args()


def main():
    args = parse_args()

    test_path = Path(args.test_file) if args.test_file else DEFAULT_VAL
    assert test_path.exists(), f"Eval file not found: {test_path}"

    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    mode = "Base model" if args.no_adapter else "Fine-tuned"

    print(f"Eval file:  {test_path}")
    print(f"Records:    {len(records)}")
    print(f"Mode:       {mode}")
    if not args.no_adapter:
        print(f"Adapter:    {args.adapter}")

    print("\nLoading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = base if args.no_adapter else PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print(f"Model ready — {mode}\n")

    counters = {
        "total":             0,
        "json_parse_ok":     0,
        "seriousness_match": 0,
        "labelling_match":   0,
        "meddra_pt_match":   0,
        "meddra_soc_match":  0,
        "all_fields_match":  0,
    }
    results = []
    t0 = time.time()

    for i, record in enumerate(records):
        narrative = extract_narrative(record)
        gold      = extract_gold(record)

        if not narrative or not gold:
            print(f"[{i+1}] Skipped — could not parse narrative or gold label")
            continue

        prompt = build_prompt(narrative, tokenizer)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Print raw output for the first PRINT_SAMPLES records
        if counters["total"] < PRINT_SAMPLES:
            print(f"\n{'─'*55}")
            print(f"Sample {counters['total']+1} — narrative (truncated):")
            print(f"  {narrative[:200]}...")
            print(f"Model output:")
            print(generated.strip())
            print(f"{'─'*55}\n")

        pred = extract_json(generated)
        counters["total"] += 1

        if pred is None:
            print(f"[{i+1}] JSON parse FAILED")
            results.append({"index": i, "parse_ok": False, "narrative": narrative[:80]})
            continue

        counters["json_parse_ok"] += 1

        s_match   = pred.get("seriousness")      == gold.get("seriousness")
        l_match   = pred.get("labelling_status") == gold.get("labelling_status")
        pt_match  = (pred.get("meddra_pt")  or "").lower() == (gold.get("meddra_pt")  or "").lower()
        soc_match = (pred.get("meddra_soc") or "").lower() == (gold.get("meddra_soc") or "").lower()
        all_match = s_match and l_match and pt_match and soc_match

        if s_match:   counters["seriousness_match"] += 1
        if l_match:   counters["labelling_match"]   += 1
        if pt_match:  counters["meddra_pt_match"]   += 1
        if soc_match: counters["meddra_soc_match"]  += 1
        if all_match: counters["all_fields_match"]  += 1

        results.append({
            "index":       i,
            "parse_ok":    True,
            "seriousness": {"pred": pred.get("seriousness"),      "gold": gold.get("seriousness"),      "match": s_match},
            "labelling":   {"pred": pred.get("labelling_status"), "gold": gold.get("labelling_status"), "match": l_match},
            "meddra_pt":   {"pred": pred.get("meddra_pt"),        "gold": gold.get("meddra_pt"),        "match": pt_match},
            "meddra_soc":  {"pred": pred.get("meddra_soc"),       "gold": gold.get("meddra_soc"),       "match": soc_match},
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(records)}]  "
                  f"seriousness={counters['seriousness_match']}/{counters['total']}  "
                  f"labelling={counters['labelling_match']}/{counters['total']}  "
                  f"meddra_pt={counters['meddra_pt_match']}/{counters['total']}  "
                  f"({(i+1)/elapsed:.1f} rec/s)")

    n = counters["total"]
    if n == 0:
        print("No records evaluated.")
        return

    elapsed = time.time() - t0
    p = counters["json_parse_ok"] or 1

    lines = [
        "",
        "=" * 60,
        f"EVALUATION RESULTS — {mode}",
        "=" * 60,
        f"  Records evaluated  : {n}",
        f"  Time               : {elapsed:.1f}s  ({n/elapsed:.1f} rec/s)",
        f"  JSON parse rate    : {counters['json_parse_ok']/n*100:.1f}%  ({counters['json_parse_ok']}/{n})",
        "",
        f"  Field accuracy (on {counters['json_parse_ok']} parsed records):",
        f"    Seriousness       : {counters['seriousness_match']/p*100:.1f}%",
        f"    MedDRA PT         : {counters['meddra_pt_match']/p*100:.1f}%",
        f"    MedDRA SOC        : {counters['meddra_soc_match']/p*100:.1f}%",
        f"    Labelling status  : {counters['labelling_match']/p*100:.1f}%",
        f"    All fields correct: {counters['all_fields_match']/p*100:.1f}%",
        "=" * 60,
    ]

    for line in lines:
        print(line)

    if args.results_txt:
        txt_path = Path(args.results_txt)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSummary saved to: {txt_path}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Per-record results saved to: {out_path}")


if __name__ == "__main__":
    main()
