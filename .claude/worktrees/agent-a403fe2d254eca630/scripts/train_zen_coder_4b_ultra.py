#!/usr/bin/env python3
"""
Ultra-fast Zen Coder 4B training with Metal acceleration.
Uses PyTorch MPS backend with LoRA fine-tuning.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.backends.mps
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "zen-coder-4b-ultra"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_PATH = "/Users/z/work/zen/zen-agentic-dataset"
OUTPUT_DIR = "./models/zen-coder-4b-ultra"

ULTRA_TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "warmup_ratio": 0.05,
    "max_seq_length": 2048,
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": False,
    "optim": "adamw_torch",
    "save_steps": 100,
    "eval_steps": 50,
    "logging_steps": 5,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "report_to": [],
    "dataloader_num_workers": 8,
    "dataloader_pin_memory": True,
    "remove_unused_columns": True,
    "group_by_length": True,
    "length_column_name": "input_length",
}

ULTRA_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
    "modules_to_save": ["embed_tokens", "lm_head"],
}


def setup_metal_optimization():
    """Setup Metal/MPS optimization."""
    if not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        return False

    if not torch.backends.mps.is_built():
        logger.warning("MPS not built, falling back to CPU")
        return False

    logger.info("Metal/MPS acceleration enabled")
    torch.backends.mps.manual_seed(42)
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    return True


def load_zen_dataset_ultra_fast():
    """Load dataset with aggressive caching for fast training."""
    logger.info("Loading zen-agentic-dataset...")

    ULTRA_TRAIN_SIZE = 15000
    ULTRA_VAL_SIZE = 1000

    train_file = f"{DATASET_PATH}/training_data.jsonl"
    val_file = f"{DATASET_PATH}/validation_data.jsonl"

    def load_jsonl_fast(file_path, max_samples, min_quality=True):
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(samples) >= max_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    if min_quality:
                        text = str(sample.get('input', '') + sample.get('output', ''))
                        if any(kw in text.lower() for kw in
                               ['def ', 'function', 'class ', 'import ', 'code', 'python', 'javascript', 'rust']):
                            samples.append(sample)
                            continue
                        if len(text) > 200 and any(kw in text.lower() for kw in
                                                   ['analyze', 'explain', 'implement', 'solution', 'algorithm']):
                            samples.append(sample)
                    else:
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return Dataset.from_list(samples)

    if os.path.exists(train_file):
        logger.info("Loading from local JSONL files...")
        train_dataset = load_jsonl_fast(train_file, ULTRA_TRAIN_SIZE, min_quality=True)
        val_dataset = load_jsonl_fast(val_file, ULTRA_VAL_SIZE, min_quality=False) if os.path.exists(val_file) else train_dataset.select(range(ULTRA_VAL_SIZE))
    else:
        logger.info("Loading from HuggingFace...")
        dataset = load_dataset("hanzoai/zen-agentic-dataset", split="train", streaming=True)
        samples = []
        for i, sample in enumerate(dataset):
            if len(samples) >= ULTRA_TRAIN_SIZE:
                break
            samples.append(sample)
        all_samples = Dataset.from_list(samples)
        train_dataset = all_samples.select(range(ULTRA_TRAIN_SIZE))
        val_dataset = all_samples.select(range(ULTRA_TRAIN_SIZE, min(ULTRA_TRAIN_SIZE + ULTRA_VAL_SIZE, len(all_samples))))

    logger.info(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    return train_dataset, val_dataset


def format_ultra_fast(example):
    """Format example for training."""
    system_prompt = "You are Zen, an expert AI coding assistant."

    if "messages" in example:
        messages = example["messages"]
        if len(messages) >= 2:
            user_msg = messages[0].get("content", "")
            assistant_msg = messages[1].get("content", "")
        else:
            return None
    elif "input" in example and "output" in example:
        user_msg = example["input"]
        assistant_msg = example["output"]
    else:
        return None

    formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

    return {
        "text": formatted_text,
        "input_length": len(formatted_text)
    }


def setup_ultra_model_and_tokenizer():
    """Setup model with Metal acceleration."""
    logger.info(f"Loading {BASE_MODEL} with Metal optimization...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="eager",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA...")
    lora_config = LoraConfig(**ULTRA_LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


class UltraFastTrainer(Trainer):
    """Custom trainer with Metal optimizations."""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def train_ultra_fast():
    """Ultra-fast training pipeline."""
    start_time = time.time()
    logger.info("Starting Zen Coder 4B ultra-fast training")

    metal_available = setup_metal_optimization()
    if not metal_available:
        logger.warning("Continuing without Metal acceleration")

    model, tokenizer = setup_ultra_model_and_tokenizer()
    train_dataset, eval_dataset = load_zen_dataset_ultra_fast()

    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(
        format_ultra_fast,
        remove_columns=train_dataset.column_names,
        num_proc=8
    ).filter(lambda x: x["text"] is not None)

    eval_dataset = eval_dataset.map(
        format_ultra_fast,
        remove_columns=eval_dataset.column_names,
        num_proc=8
    ).filter(lambda x: x["text"] is not None)

    logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    def ultra_tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=ULTRA_TRAINING_CONFIG["max_seq_length"],
            padding="max_length",
            return_tensors="pt"
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    train_dataset = train_dataset.map(
        ultra_tokenize,
        batched=True,
        num_proc=8,
        remove_columns=["text", "input_length"]
    )
    eval_dataset = eval_dataset.map(
        ultra_tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text", "input_length"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **ULTRA_TRAINING_CONFIG,
    )

    trainer = UltraFastTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info(f"Starting training at {time.strftime('%H:%M:%S')}")
    train_result = trainer.train()

    end_time = time.time()
    total_time = end_time - start_time

    logger.info(f"Training completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    logger.info(f"Target achieved: {'Yes' if total_time < 3600 else 'No'}")

    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    stats = {
        "training_time_minutes": total_time / 60,
        "training_time_hours": total_time / 3600,
        "final_loss": train_result.training_loss,
        "epochs_completed": ULTRA_TRAINING_CONFIG["num_train_epochs"],
        "samples_trained": len(train_dataset),
        "target_achieved": total_time < 3600,
        "metal_acceleration": metal_available,
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
    }

    with open(f"{OUTPUT_DIR}/training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Training complete")
    return stats


if __name__ == "__main__":
    try:
        stats = train_ultra_fast()
        if stats["target_achieved"]:
            print("Training completed in under 1 hour")
        else:
            print(f"Completed in {stats['training_time_hours']:.2f} hours")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
