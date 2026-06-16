"""
Fine-tune Qwen3-14B with LoRA for Medical Review Assistant.

Stack: TRL SFTTrainer + PEFT LoRA + HuggingFace Transformers
Hardware: AMD MI300X (192GB) — ROCm, BF16
Runtime: ~1.5-2 hours for 3K examples x 3 epochs

Usage:
    cd train
    python train.py

    # To resume from checkpoint:
    python train.py --resume_from_checkpoint ./output/checkpoint-500
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
import trl
from transformers import Trainer  # used instead of SFTTrainer to avoid TRL API instability

# DataCollatorForCompletionOnlyLM was removed from TRL ~0.14+.
# Implement it locally so we don't depend on TRL for this class.
class DataCollatorForCompletionOnlyLM:
    """
    Masks every token that comes before (and including) the response template so
    that cross-entropy loss is computed only on the assistant's JSON output.
    """

    def __init__(self, response_template, tokenizer, mlm=False, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.response_template_ids = (
            tokenizer.encode(response_template, add_special_tokens=False)
            if isinstance(response_template, str)
            else list(response_template)
        )

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        tlen = len(self.response_template_ids)

        for i in range(len(labels)):
            seq = labels[i].tolist()
            mask_until = len(seq)  # default: mask everything if template not found
            for j in range(len(seq) - tlen, -1, -1):
                if seq[j : j + tlen] == self.response_template_ids:
                    mask_until = j + tlen  # keep tokens AFTER the template
                    break
            labels[i, :mask_until] = self.ignore_index

        labels[batch["attention_mask"] == 0] = self.ignore_index
        batch["labels"] = labels
        return batch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
TRAIN_FILE = ROOT / "data" / "train.jsonl"
VAL_FILE   = ROOT / "data" / "val.jsonl"
OUTPUT_DIR = ROOT / "train" / "output"

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-14B"

# ── LoRA hyperparameters ──────────────────────────────────────────────────────
LORA_R           = 64
LORA_ALPHA       = 128     # scaling = alpha/r = 2.0
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [    # all linear layers in Qwen3
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyperparameters ──────────────────────────────────────────────────
# With MI300X (192GB), Qwen3-14B BF16 uses ~35GB → plenty of headroom
NUM_EPOCHS               = 3
PER_DEVICE_BATCH_SIZE    = 2
GRADIENT_ACCUMULATION    = 16     # effective batch = 2 * 16 = 32
LEARNING_RATE            = 2e-4
LR_SCHEDULER             = "cosine"
WARMUP_RATIO             = 0.05
MAX_SEQ_LENGTH           = 2048
SAVE_STEPS               = 200
LOGGING_STEPS            = 10
MAX_GRAD_NORM            = 1.0

# ChatML response template — loss is computed ONLY on tokens after this
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    assert TRAIN_FILE.exists(), f"Training file not found: {TRAIN_FILE}"
    assert VAL_FILE.exists(),   f"Val file not found: {VAL_FILE}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model:   {args.base_model}")
    print(f"Train file:   {TRAIN_FILE} ({TRAIN_FILE.stat().st_size / 1e3:.0f} KB)")
    print(f"Val file:     {VAL_FILE}")
    print(f"Output dir:   {output_dir}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"TRL:          {trl.__version__}")
    print(f"CUDA/ROCm:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",   # right-padding for SFT
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")

    # ── Load model in BF16 ────────────────────────────────────────────────────
    print("\nLoading model in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"Model loaded. VRAM used: {vram_gb:.1f} GB")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    print("\nApplying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    val_dataset   = load_dataset("json", data_files=str(VAL_FILE),   split="train")
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val:   {len(val_dataset)} examples")

    # ── Response-only loss: mask instruction tokens ───────────────────────────
    # Only compute loss on the assistant's JSON output, not on the prompt.
    # This focuses gradient updates entirely on medical review accuracy.
    response_template_ids = tokenizer.encode(
        RESPONSE_TEMPLATE,
        add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    # ── Tokenize datasets ─────────────────────────────────────────────────────
    # Pre-tokenize so we use stable transformers.Trainer instead of SFTTrainer,
    # whose API changes every minor TRL release.
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,   # the collator handles per-batch padding
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_dataset   = val_dataset.map(tokenize,   batched=True, remove_columns=["text"])

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        fp16=False,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        run_name="qwen3-14b-medical-review",
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training...")
    print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Steps per epoch: {len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION)}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save final adapter ────────────────────────────────────────────────────
    print("\nSaving final adapter...")
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training config for reproducibility
    config = {
        "base_model":    args.base_model,
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "lora_dropout":  LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "num_epochs":    NUM_EPOCHS,
        "batch_size":    PER_DEVICE_BATCH_SIZE,
        "grad_accum":    GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    with open(final_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete. Adapter saved to: {final_dir}")
    print("Next step: run infer/infer.py to test the model.")


if __name__ == "__main__":
    main()
