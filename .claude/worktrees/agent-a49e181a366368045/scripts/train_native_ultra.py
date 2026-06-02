#!/usr/bin/env python3
"""
Native Rust training with Hanzo Metal kernels.
Target: < 30 minutes for Zen Coder 4B with native Metal acceleration.
"""

import subprocess
import json
import time
import sys
from pathlib import Path


def create_native_training_config():
    """Create optimized config for native Rust training."""
    config = {
        "model": {
            "name": "zen-coder-4b",
            "architecture": "qwen3",
            "checkpoint": None,
            "max_seq_length": 2048,
            "hidden_size": 3584,
            "num_layers": 32,
            "num_heads": 28,
        },
        "dataset": {
            "name": "zen-agentic-dataset",
            "path": "/Users/z/work/zen/zen-agentic-dataset",
            "format": "zen_agentic",
            "train_split": "train",
            "validation_split": "validation",
            "max_seq_length": 2048,
            "preprocessing": {
                "shuffle": True,
                "filter_min_length": 50,
                "filter_max_length": 2048
            }
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "epochs": 3,
            "warmup_steps": 100,
            "save_steps": 200,
            "eval_steps": 100,
            "gradient_accumulation_steps": 2,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "device": "metal"
        },
        "evaluation": {
            "benchmarks": ["perplexity"],
            "metrics": ["loss"],
            "eval_dataset": None,
            "output_dir": "./eval_results"
        },
        "logging": {
            "wandb": {"enabled": False},
            "tensorboard": True,
            "console_level": "info",
            "file_logging": {
                "enabled": True,
                "path": "./logs/training.log",
                "level": "debug"
            }
        }
    }

    config_path = "./ultra_training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def check_hanzo_training_status():
    """Check if hanzo-training compiles."""
    print("Checking hanzo-training compilation status...")

    try:
        result = subprocess.run(
            ["cargo", "check", "--package", "hanzo-training"],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("hanzo-training compiles successfully")
            return True
        else:
            print("hanzo-training has compilation errors:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("Compilation check timed out")
        return False
    except Exception as e:
        print(f"Error checking compilation: {e}")
        return False


def run_native_metal_training():
    """Run native Rust training with Metal acceleration."""
    print("STARTING NATIVE HANZO METAL TRAINING")
    print("=" * 60)

    config_path = create_native_training_config()
    print(f"Created config: {config_path}")

    if not check_hanzo_training_status():
        print("Cannot proceed - hanzo-training doesn't compile")
        print("Falling back to PyTorch Metal training...")
        return False

    cmd = [
        "cargo", "run", "--release", "--bin", "hanzo-train", "--",
        "--config", config_path,
        "--output", "./models/zen-coder-4b-native"
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Started at: {time.strftime('%H:%M:%S')}")

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())

        process.wait()

        end_time = time.time()
        total_time = end_time - start_time

        if process.returncode == 0:
            print("NATIVE TRAINING COMPLETED")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Target achieved: {'Yes' if total_time < 1800 else 'No'}")
            return True
        else:
            print(f"Training failed with return code: {process.returncode}")
            return False

    except Exception as e:
        print(f"Training failed: {e}")
        return False


def main():
    """Main function."""
    print("Hanzo Native Metal Ultra-Fast Training")
    print("Target: Train Zen Coder 4B in < 30 minutes with native Metal")
    print()

    if run_native_metal_training():
        print("Native Metal training completed")
        return

    print("Falling back to PyTorch Metal training...")
    try:
        subprocess.run([sys.executable, "train_zen_coder_4b_ultra.py"], check=True)
        print("PyTorch training completed")
    except Exception as e:
        print(f"PyTorch training also failed: {e}")


if __name__ == "__main__":
    main()
