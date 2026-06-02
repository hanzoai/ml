#!/usr/bin/env python3
"""
Ultimate benchmark: Native Hanzo vs PyTorch training.
Compares training speed, memory usage, and model quality.
"""

import subprocess
import sys
import os
import time
import json
import psutil
import threading
from pathlib import Path
from datetime import datetime


class TrainingBenchmark:
    def __init__(self):
        self.results = {
            "benchmark_start": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "native_hanzo": {},
            "pytorch_mps": {},
            "comparison": {}
        }

    def get_system_info(self):
        """Get system information."""
        return {
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
        }

    def monitor_resources(self, results_dict, stop_event):
        """Monitor CPU, memory usage during training."""
        max_memory = 0
        cpu_samples = []
        memory_samples = []

        while not stop_event.is_set():
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.used / (1024**3)

            cpu_samples.append(cpu_percent)
            memory_samples.append(memory_gb)
            max_memory = max(max_memory, memory_gb)

        results_dict["resource_usage"] = {
            "max_memory_gb": max_memory,
            "avg_cpu_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            "peak_memory_gb": max(memory_samples) if memory_samples else 0,
        }

    def train_native_hanzo(self):
        """Train with native Hanzo Metal kernels."""
        print("\n" + "=" * 80)
        print("TRAINING WITH NATIVE HANZO METAL KERNELS")
        print("=" * 80)

        start_time = time.time()
        stop_event = threading.Event()

        monitor_thread = threading.Thread(
            target=self.monitor_resources,
            args=(self.results["native_hanzo"], stop_event)
        )
        monitor_thread.start()

        try:
            cmd = [sys.executable, "train_native_ultra.py"]
            print(f"Command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in iter(process.stdout.readline, ''):
                print(f"[NATIVE] {line.rstrip()}")
                output_lines.append(line.strip())

            process.wait()

            end_time = time.time()
            training_time = end_time - start_time

            stop_event.set()
            monitor_thread.join()

            self.results["native_hanzo"].update({
                "success": process.returncode == 0,
                "training_time_minutes": training_time / 60,
                "training_time_seconds": training_time,
                "output_lines": output_lines[-50:],
                "exit_code": process.returncode,
            })

            if process.returncode == 0:
                print(f"NATIVE HANZO TRAINING COMPLETED in {training_time/60:.1f} minutes")
            else:
                print(f"NATIVE HANZO TRAINING FAILED (exit code: {process.returncode})")

        except Exception as e:
            stop_event.set()
            monitor_thread.join()
            self.results["native_hanzo"]["error"] = str(e)
            print(f"NATIVE HANZO ERROR: {e}")

    def train_pytorch_mps(self):
        """Train with PyTorch MPS."""
        print("\n" + "=" * 80)
        print("TRAINING WITH PYTORCH MPS")
        print("=" * 80)

        start_time = time.time()
        stop_event = threading.Event()

        monitor_thread = threading.Thread(
            target=self.monitor_resources,
            args=(self.results["pytorch_mps"], stop_event)
        )
        monitor_thread.start()

        try:
            print("Installing PyTorch dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "transformers", "datasets", "peft", "accelerate",
                "--upgrade", "--quiet"
            ], check=True)

            cmd = [sys.executable, "train_zen_coder_4b_ultra.py"]
            print(f"Command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            for line in iter(process.stdout.readline, ''):
                print(f"[PYTORCH] {line.rstrip()}")
                output_lines.append(line.strip())

            process.wait()

            end_time = time.time()
            training_time = end_time - start_time

            stop_event.set()
            monitor_thread.join()

            self.results["pytorch_mps"].update({
                "success": process.returncode == 0,
                "training_time_minutes": training_time / 60,
                "training_time_seconds": training_time,
                "output_lines": output_lines[-50:],
                "exit_code": process.returncode,
            })

            if process.returncode == 0:
                print(f"PYTORCH MPS TRAINING COMPLETED in {training_time/60:.1f} minutes")
            else:
                print(f"PYTORCH MPS TRAINING FAILED (exit code: {process.returncode})")

        except Exception as e:
            stop_event.set()
            monitor_thread.join()
            self.results["pytorch_mps"]["error"] = str(e)
            print(f"PYTORCH MPS ERROR: {e}")

    def generate_comparison_report(self):
        """Generate detailed comparison report."""
        print("\n" + "=" * 80)
        print("GENERATING BENCHMARK COMPARISON REPORT")
        print("=" * 80)

        native = self.results["native_hanzo"]
        pytorch = self.results["pytorch_mps"]

        comparison = {}

        if native.get("success") and pytorch.get("success"):
            native_time = native["training_time_minutes"]
            pytorch_time = pytorch["training_time_minutes"]

            comparison["speed_advantage"] = {
                "native_faster": native_time < pytorch_time,
                "speed_ratio": pytorch_time / native_time if native_time > 0 else 0,
                "time_saved_minutes": pytorch_time - native_time,
            }

            native_mem = native.get("resource_usage", {}).get("peak_memory_gb", 0)
            pytorch_mem = pytorch.get("resource_usage", {}).get("peak_memory_gb", 0)

            comparison["memory_efficiency"] = {
                "native_lower": native_mem < pytorch_mem,
                "memory_ratio": pytorch_mem / native_mem if native_mem > 0 else 0,
                "memory_saved_gb": pytorch_mem - native_mem,
            }

        self.results["comparison"] = comparison

        results_file = f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.print_summary_report()
        print(f"\nFull results saved to: {results_file}")

    def print_summary_report(self):
        """Print formatted summary report."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS: NATIVE HANZO vs PYTORCH MPS")
        print("=" * 80)

        native = self.results["native_hanzo"]
        pytorch = self.results["pytorch_mps"]
        comparison = self.results["comparison"]

        print("\nTRAINING RESULTS:")
        print("-" * 50)

        if native.get("success"):
            print(f"NATIVE HANZO: SUCCESS")
            print(f"   Time: {native['training_time_minutes']:.1f} minutes")
            if "resource_usage" in native:
                print(f"   Peak Memory: {native['resource_usage']['peak_memory_gb']:.1f} GB")
                print(f"   Avg CPU: {native['resource_usage']['avg_cpu_percent']:.1f}%")
        else:
            print(f"NATIVE HANZO: FAILED")
            if "error" in native:
                print(f"   Error: {native['error']}")

        if pytorch.get("success"):
            print(f"PYTORCH MPS: SUCCESS")
            print(f"   Time: {pytorch['training_time_minutes']:.1f} minutes")
            if "resource_usage" in pytorch:
                print(f"   Peak Memory: {pytorch['resource_usage']['peak_memory_gb']:.1f} GB")
                print(f"   Avg CPU: {pytorch['resource_usage']['avg_cpu_percent']:.1f}%")
        else:
            print(f"PYTORCH MPS: FAILED")
            if "error" in pytorch:
                print(f"   Error: {pytorch['error']}")

        if "speed_advantage" in comparison:
            speed = comparison["speed_advantage"]
            memory = comparison["memory_efficiency"]

            print("\nPERFORMANCE COMPARISON:")
            print("-" * 50)

            if speed["native_faster"]:
                print(f"WINNER: NATIVE HANZO IS {speed['speed_ratio']:.2f}x FASTER")
                print(f"Time Saved: {speed['time_saved_minutes']:.1f} minutes")
            else:
                print(f"WINNER: PYTORCH MPS IS {1/speed['speed_ratio']:.2f}x FASTER")
                print(f"Time Saved: {-speed['time_saved_minutes']:.1f} minutes")

            if memory["native_lower"]:
                print(f"Memory: Native Hanzo uses {memory['memory_saved_gb']:.1f} GB LESS")
            else:
                print(f"Memory: PyTorch MPS uses {-memory['memory_saved_gb']:.1f} GB LESS")

        print("=" * 80)

    def run_full_benchmark(self):
        """Run complete benchmark."""
        print("STARTING ULTIMATE BENCHMARK: NATIVE HANZO vs PYTORCH MPS")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

        self.train_native_hanzo()
        self.train_pytorch_mps()
        self.generate_comparison_report()

        print("BENCHMARK COMPLETE")


def main():
    """Main benchmark function."""
    if sys.platform != "darwin":
        print("This benchmark requires macOS for Metal acceleration")
        sys.exit(1)

    if not Path("hanzo-metal-kernels").exists():
        print("hanzo-metal-kernels not found")
        sys.exit(1)

    print("READY TO BENCHMARK: Native Hanzo Metal vs PyTorch MPS")
    print("This will run TWO training sessions back-to-back")

    response = input("\nStart ultimate benchmark? [Y/n]: ").strip().lower()

    if response in ['', 'y', 'yes']:
        benchmark = TrainingBenchmark()
        benchmark.run_full_benchmark()
    else:
        print("Benchmark cancelled")


if __name__ == "__main__":
    main()
