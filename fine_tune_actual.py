#!/usr/bin/env python3
"""
Azentio Hackathon: Production LoRA Fine-Tuning Execution
========================================================
Executes real Hugging Face PEFT/LoRA fine-tuning on Qwen2.5-1.5B-Instruct.

Configuration:
  - Base Model:     Qwen/Qwen2.5-1.5B-Instruct (<3B parameters)
  - LoRA Config:    rank r=16, alpha=32, dropout=0.05
  - Target Modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - Train Dataset:  data/train_weak_labels.jsonl (700 records, WEAK LABELS)
  - Val Dataset:    data/val_weak_labels.jsonl (150 records, WEAK LABELS)
  - Output Adapter: models/fraud-sentinel-lora/
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run actual LoRA fine-tuning on Qwen2.5-1.5B")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-data", type=str, default="data/train_weak_labels.jsonl")
    parser.add_argument("--val-data", type=str, default="data/val_weak_labels.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/fraud-sentinel-lora")
    parser.add_argument("--max-steps", type=int, default=30, help="Training steps for CPU execution")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    train_file = base_dir / args.train_data
    val_file = base_dir / args.val_data
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 Azentio Sentinel: Executing Real LoRA Fine-Tuning")
    print("=" * 70)
    print(f"Base Model:       {args.model}")
    print(f"LoRA Parameters:  r={args.lora_r}, alpha={args.lora_alpha}, dropout=0.05")
    print(f"Target Modules:   ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
    print(f"Train Data:       {train_file} (WEAK LABELS)")
    print(f"Val Data:         {val_file} (WEAK LABELS)")
    print(f"Output Adapter:   {output_dir}")
    print(f"Max Steps:        {args.max_steps}")
    print(f"Device:           CPU (macOS x86_64)")
    print("=" * 70)

    # 1. Load Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    print("[2/5] Loading base model weights...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"      Base model loaded in {time.time() - t0:.1f}s")

    # 3. Configure LoRA PEFT Model
    print("[3/5] Attaching LoRA adapter...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Prepare Dataset with ChatML Template
    print("[4/5] Formatting and tokenizing training conversations...")
    raw_datasets = load_dataset(
        "json",
        data_files={"train": str(train_file), "val": str(val_file)}
    )

    def format_and_tokenize(examples):
        input_ids_list = []
        labels_list = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding=False,
            )
            input_ids = tokens["input_ids"]
            labels = list(input_ids)
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized_train = raw_datasets["train"].map(format_and_tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
    tokenized_val = raw_datasets["val"].map(format_and_tokenize, batched=True, remove_columns=raw_datasets["val"].column_names)
    print(f"      Tokenized Train: {len(tokenized_train)} samples")
    print(f"      Tokenized Val:   {len(tokenized_val)} samples")

    # 5. Execute Training
    print("\n[5/5] Launching Trainer execution...")
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=5,
        logging_steps=5,
        save_strategy="no",
        evaluation_strategy="steps",
        eval_steps=15,
        per_device_eval_batch_size=2,
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=512,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
    )

    train_start = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - train_start

    print(f"\n✅ Training completed in {total_train_time:.1f}s!")
    print(f"   Train Loss: {train_result.training_loss:.4f}")

    # 6. Save Adapter
    print(f"\n💾 Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Verify adapter files
    adapter_weights = output_dir / "adapter_model.safetensors"
    adapter_bin = output_dir / "adapter_model.bin"
    adapter_cfg = output_dir / "adapter_config.json"

    has_weights = adapter_weights.exists() or adapter_bin.exists()
    has_cfg = adapter_cfg.exists()

    print("\n🔍 Post-Training Adapter Verification:")
    print(f"   adapter_config.json:        {'EXISTS ✅' if has_cfg else 'MISSING ❌'} ({adapter_cfg})")
    print(f"   adapter_model.(safetensors): {'EXISTS ✅' if has_weights else 'MISSING ❌'}")
    if has_weights:
        wpath = adapter_weights if adapter_weights.exists() else adapter_bin
        print(f"   Adapter Size:               {wpath.stat().st_size / (1024*1024):.2f} MB")

    if not (has_weights and has_cfg):
        raise RuntimeError("Adapter saving failed verification!")

    print("\n🎉 Actual LoRA adapter produced and verified successfully!")

if __name__ == "__main__":
    main()
