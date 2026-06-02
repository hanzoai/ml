#!/usr/bin/env python3
"""
Hanzo ultra-fast training launcher.
Selects optimal training method (Native Rust vs PyTorch) and runs training.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def check_requirements():
    """Check system requirements for ultra-fast training."""
    print("Checking system requirements...")

    checks = {
        "macOS": sys.platform == "darwin",
        "Hanzo ML": Path("hanzo-ml").exists(),
        "Metal kernels": Path("hanzo-metal-kernels").exists(),
        "Training data": Path("/Users/z/work/zen/zen-agentic-dataset").exists(),
    }

    for check, status in checks.items():
        print(f"  {'OK' if status else 'MISSING'}: {check}")

    if not all(checks.values()):
        print("Missing requirements for ultra-fast training")
        return False

    print("All requirements met")
    return True


def choose_training_method():
    """Choose the fastest available training method."""
    print("\nSelecting optimal training method...")

    # Check if hanzo-training compiles
    print("Testing hanzo-training compilation...")
    try:
        result = subprocess.run(
            ["cargo", "check", "--package", "hanzo-training"],
            capture_output=True,
            text=True,
            timeout=60
        )
        rust_available = result.returncode == 0
    except Exception:
        rust_available = False

    # Check PyTorch + MPS
    try:
        result = subprocess.run([
            sys.executable, "-c",
            "import torch; exit(0 if torch.backends.mps.is_available() else 1)"
        ], capture_output=True)
        pytorch_mps = result.returncode == 0
    except Exception:
        pytorch_mps = False

    print(f"  Native Rust + Metal: {'OK' if rust_available else 'UNAVAILABLE'}")
    print(f"  PyTorch + MPS: {'OK' if pytorch_mps else 'UNAVAILABLE'}")

    if rust_available:
        print("Selected: Native Rust + Metal (fastest)")
        return "native"
    elif pytorch_mps:
        print("Selected: PyTorch + MPS")
        return "pytorch"
    else:
        print("Selected: CPU fallback (slow)")
        return "cpu"


def run_training(method):
    """Run training with selected method."""
    print(f"\nStarting training with {method}...")

    start_time = time.time()

    if method == "native":
        cmd = [sys.executable, "train_native_ultra.py"]
        target_time = 30 * 60
    elif method == "pytorch":
        cmd = [sys.executable, "train_zen_coder_4b_ultra.py"]
        target_time = 60 * 60
    else:
        print("CPU training not implemented (too slow)")
        return False

    print(f"Command: {' '.join(cmd)}")
    print(f"Target time: {target_time // 60} minutes")
    print(f"Started at: {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    try:
        process = subprocess.Popen(
            cmd,
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

        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Target met: {'Yes' if total_time < target_time else 'No'}")

        if process.returncode == 0:
            print("Training completed successfully")
            return True
        else:
            print(f"Training failed (exit code: {process.returncode})")
            return False

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"Training error: {e}")
        return False


def main():
    """Main launcher function."""
    print("Hanzo Ultra-Fast Zen Coder 4B Training")
    print("=" * 60)

    if not check_requirements():
        print("Cannot proceed without requirements")
        sys.exit(1)

    method = choose_training_method()

    print(f"\nReady to start ultra-fast training with {method}")
    response = input("Start training? [Y/n]: ").strip().lower()

    if response in ['', 'y', 'yes']:
        success = run_training(method)
        if success:
            print("\nTraining complete")
            print("Model saved to: ./models/zen-coder-4b-*")
        else:
            print("\nTraining failed - check logs above")
            sys.exit(1)
    else:
        print("Training cancelled")


if __name__ == "__main__":
    main()
