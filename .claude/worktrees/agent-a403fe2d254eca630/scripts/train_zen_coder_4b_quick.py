#!/usr/bin/env python3
"""
Quick start: Train Zen Coder 4B using PyTorch with LoRA.
Adaptation for zen-agentic-dataset.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
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
    TaskType
)
from datasets import load_dataset, Dataset
import huggingface_hub

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "zen-coder-4b"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_PATH = "/Users/z/work/zen/zen-agentic-dataset"
OUTPUT_DIR = "./models/zen-coder-4b"
HF_REPO = "zenai/zen-coder-4b"

TRAINING_CONFIG = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "max_seq_length": 4096,
    "fp16": False,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "save_steps": 250,
    "eval_steps": 250,
    "logging_steps": 10,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
}

LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}


def load_zen_agentic_dataset():
    """Load zen-agentic-dataset subset for training."""
    logger.info(f"Loading zen-agentic-dataset from {DATASET_PATH}")

    train_file = f"{DATASET_PATH}/training_data.jsonl"
    val_file = f"{DATASET_PATH}/validation_data.jsonl"

    if not os.path.exists(train_file):
        logger.error(f"Training data not found at {train_file}")
        logger.info("Falling back to huggingface dataset...")
        dataset = load_dataset("hanzoai/zen-agentic-dataset", split="train")
        train_dataset = dataset.select(range(min(5000, len(dataset))))
        val_dataset = dataset.select(range(5000, min(6000, len(dataset))))
        return train_dataset, val_dataset

    def load_jsonl(file_path, max_samples=None):
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return Dataset.from_list(samples)

    train_dataset = load_jsonl(train_file, max_samples=5000)
    val_dataset = load_jsonl(val_file, max_samples=500) if os.path.exists(val_file) else train_dataset.select(range(500))

    logger.info(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    return train_dataset, val_dataset


def format_coding_example(example):
    """Format zen-agentic examples for coding training."""
    system_prompt = """You are Zen, a highly capable AI coding assistant. You think step by step, write clean code, and provide helpful explanations. You excel at understanding requirements, debugging, and implementing solutions efficiently."""

    if "messages" in example:
        messages = example["messages"]
        if len(messages) >= 2:
            user_msg = messages[0]["content"] if messages[0]["role"] == "user" else ""
            assistant_msg = messages[1]["content"] if messages[1]["role"] == "assistant" else ""

            formatted_text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
{assistant_msg}<|im_end|>"""
            return {"text": formatted_text}

    if "input" in example and "output" in example:
        formatted_text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        return {"text": formatted_text}

    return None


def setup_model_and_tokenizer():
    """Setup base model and tokenizer."""
    logger.info(f"Loading base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def train_zen_coder():
    """Main training function."""
    logger.info("Starting Zen Coder 4B Training")
    logger.info("=" * 60)

    model, tokenizer = setup_model_and_tokenizer()
    train_dataset, eval_dataset = load_zen_agentic_dataset()

    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(format_coding_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_coding_example, remove_columns=eval_dataset.column_names)

    train_dataset = train_dataset.filter(lambda x: x["text"] is not None)
    eval_dataset = eval_dataset.filter(lambda x: x["text"] is not None)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=TRAINING_CONFIG["max_seq_length"],
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Training completed")
    logger.info(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train_zen_coder()
