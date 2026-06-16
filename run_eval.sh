#!/bin/bash
set -e

INFER_DIR="/workspace/shared/TCS-Hackathon/infer"
VAL_FILE="/workspace/shared/TCS-Hackathon/data/val_chatml.jsonl"
ADAPTER="/workspace/shared/TCS-Hackathon/train/output/final_adapter"
OUT_DIR="/workspace/shared/TCS-Hackathon/data/eval_results"
LIMIT=50

mkdir -p "$OUT_DIR"

echo ""
echo "############################################################"
echo "  Medical Review Assistant — Evaluation (${LIMIT} records)"
echo "  2 configurations: base model vs fine-tuned"
echo "############################################################"

# ── 1. Base model ─────────────────────────────────────────────────────────────
echo ""
echo "[1/2] Base model"
echo "------------------------------------------------------------"
python "$INFER_DIR/eval.py" \
  --test_file  "$VAL_FILE" \
  --no_adapter \
  --limit      $LIMIT \
  --output     "$OUT_DIR/base.jsonl" \
  --results_txt "$OUT_DIR/base.txt"

# ── 2. Fine-tuned model ───────────────────────────────────────────────────────
echo ""
echo "[2/2] Fine-tuned"
echo "------------------------------------------------------------"
python "$INFER_DIR/eval.py" \
  --test_file  "$VAL_FILE" \
  --adapter    "$ADAPTER" \
  --limit      $LIMIT \
  --output     "$OUT_DIR/finetuned.jsonl" \
  --results_txt "$OUT_DIR/finetuned.txt"

# ── Comparison summary ────────────────────────────────────────────────────────
SUMMARY="$OUT_DIR/comparison_summary.txt"

echo ""
echo "############################################################"
echo "  COMPARISON SUMMARY"
echo "############################################################"

{
  echo "============================================================"
  echo "  COMPARISON SUMMARY — Medical Review Assistant Evaluation"
  echo "  Records per run : $LIMIT"
  echo "  Val file        : $VAL_FILE"
  echo "============================================================"
  echo ""
  echo "── Base model ───────────────────────────────────────────────"
  cat "$OUT_DIR/base.txt"
  echo ""
  echo "── Fine-tuned ───────────────────────────────────────────────"
  cat "$OUT_DIR/finetuned.txt"
  echo ""
  echo "============================================================"
  echo "  Files saved to: $OUT_DIR/"
  echo "============================================================"
} | tee "$SUMMARY"

echo ""
echo "All done. Summary saved to: $SUMMARY"
