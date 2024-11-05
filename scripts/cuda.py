import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List

import GPUtil
import torch


class TestPattern(Enum):
    COMPUTE_BOUND = auto()
    MEMORY_BOUND = auto()
    MIXED_PRECISION = auto()
    CYCLE_TEST = auto()


def get_available_gpus() -> List[str]:
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices if devices else ["cpu"]


class GPUStressTest:
    def __init__(
        self,
        device: str = "cuda:0",
        size: int = 32_000,
        num_threads: int = 1,
        pattern: TestPattern = TestPattern.COMPUTE_BOUND,
        log_dir: str = "gpu_stress_logs",
    ):
        self.device_str = device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system")
            device_idx = int(device.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {device_idx} not found. Available GPUs: 0-{torch.cuda.device_count()-1}"
                )
            torch.cuda.set_device(device_idx)
        self.size = size
        self.num_threads = num_threads
        self.pattern = pattern
        self.running = True
        self.total_iterations = 0
        self.start_time = None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.device = torch.device(device)
        device_name = device.replace(":", "_")
        self.csv_file = (
            self.log_dir
            / f"gpu_stats_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.json_file = (
            self.log_dir
            / f"test_summary_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.gpu_index = int(device.split(":")[1]) if device.startswith("cuda") else 0
        self.stats_history = []
        self.peak_temperature = 0
        self.peak_memory = 0
        self.peak_utilization = 0
        self.throttle_events = 0
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self.setup_logging()

    def _handle_interrupt(self, signum, frame):
        print("\nShutting down stress test...")
        self.running = False

    def setup_logging(self):
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "gpu_load",
                    "temperature",
                    "memory_used",
                    "memory_total",
                    "power_draw",
                    "power_limit",
                    "fan_speed",
                    "pcie_bandwidth",
                    "iterations",
                    "throttle_status",
                ]
            )

    def _get_nvidia_smi_data(self) -> Dict:
        try:
            cmd = [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=power.draw,power.limit,fan.speed,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd).decode()
            power_draw, power_limit, fan_speed, pcie_gen, pcie_width = map(
                float, output.strip().split(",")
            )
            return {
                "power_draw": power_draw,
                "power_limit": power_limit,
                "fan_speed": fan_speed,
                "pcie_bandwidth": pcie_gen * pcie_width * 0.985,
            }
        except:
            return {
                "power_draw": 0,
                "power_limit": 0,
                "fan_speed": 0,
                "pcie_bandwidth": 0,
            }

    def _check_throttling(self, gpu) -> str:
        throttle_reasons = []
        if gpu.temperature >= 80:
            throttle_reasons.append("THERMAL")
        if (
            self._get_nvidia_smi_data()["power_draw"]
            >= self._get_nvidia_smi_data()["power_limit"] * 0.95
        ):
            throttle_reasons.append("POWER")
        return "+".join(throttle_reasons) if throttle_reasons else "NONE"

    def _monitor_resources(self):
        while self.running:
            try:
                gpu = (
                    GPUtil.getGPUs()[self.gpu_index]
                    if self.device_str.startswith("cuda")
                    else None
                )
                if gpu:
                    nvidia_data = self._get_nvidia_smi_data()
                    self.peak_temperature = max(self.peak_temperature, gpu.temperature)
                    self.peak_memory = max(self.peak_memory, gpu.memoryUsed)
                    self.peak_utilization = max(self.peak_utilization, gpu.load * 100)
                    throttle_status = self._check_throttling(gpu)
                    if throttle_status != "NONE":
                        self.throttle_events += 1
                    stats = {
                        "timestamp": datetime.now().isoformat(),
                        "gpu_load": gpu.load * 100,
                        "temperature": gpu.temperature,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "power_draw": nvidia_data["power_draw"],
                        "power_limit": nvidia_data["power_limit"],
                        "fan_speed": nvidia_data["fan_speed"],
                        "pcie_bandwidth": nvidia_data["pcie_bandwidth"],
                        "iterations": self.total_iterations,
                        "throttle_status": throttle_status,
                    }
                    with open(self.csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(stats.values())
                    self._print_status(stats)
                    self.stats_history.append(stats)
                else:
                    print("\rCPU mode - monitoring disabled", end="")
            except Exception as e:
                print(f"\nMonitoring error: {e}")
            time.sleep(1)

    def _print_status(self, stats: Dict):
        status = (
            f"\rGPU Load: {stats['gpu_load']:>3.0f}% | "
            f"Temp: {stats['temperature']:>3.0f}°C | "
            f"Memory: {stats['memory_used']:>5.0f}/{stats['memory_total']:>5.0f} MB | "
            f"Power: {stats['power_draw']:>5.1f}/{stats['power_limit']:>5.1f}W | "
            f"Fan: {stats['fan_speed']:>3.0f}% | "
            f"PCIe: {stats['pcie_bandwidth']:>4.1f} GB/s | "
            f"Iters: {stats['iterations']:>6d} | "
            f"Runtime: {self._get_runtime():>8s}"
        )
        if stats["throttle_status"] != "NONE":
            status += f" | ⚠️ {stats['throttle_status']}"
        print(status, end="")

    def _get_runtime(self) -> str:
        if self.start_time is None:
            return "0:00:00"
        seconds = int((datetime.now() - self.start_time).total_seconds())
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def _compute_bound_test(self):
        X = torch.randn(self.size, self.size, device=self.device)
        return X @ X.T

    def _memory_bound_test(self):
        X = torch.randn(self.size * 2, self.size // 2, device=self.device)
        for _ in range(10):
            X = torch.roll(X, shifts=1, dims=0)
            X = torch.roll(X, shifts=1, dims=1)
        return X

    def _mixed_precision_test(self):
        sizes = {"fp16": self.size * 2, "fp32": self.size, "fp64": self.size // 2}
        X_half = torch.randn(
            sizes["fp16"], sizes["fp16"], device=self.device, dtype=torch.float16
        )
        result_half = X_half @ X_half.T
        X_float = torch.randn(
            sizes["fp32"], sizes["fp32"], device=self.device, dtype=torch.float32
        )
        result_float = X_float @ X_float.T
        X_double = torch.randn(
            sizes["fp64"], sizes["fp64"], device=self.device, dtype=torch.float64
        )
        result_double = X_double @ X_double.T
        return result_half, result_float, result_double

    def _cycle_test(self):
        cycle_time = time.time() % 10
        if cycle_time < 5:
            return self._compute_bound_test()
        else:
            time.sleep(0.1)
            return torch.randn(1000, 1000, device=self.device)

    def _stress_worker(self):
        try:
            while self.running:
                try:
                    if self.pattern == TestPattern.COMPUTE_BOUND:
                        result = self._compute_bound_test()
                    elif self.pattern == TestPattern.MEMORY_BOUND:
                        result = self._memory_bound_test()
                    elif self.pattern == TestPattern.MIXED_PRECISION:
                        result = self._mixed_precision_test()
                    else:
                        result = self._cycle_test()
                    torch.cuda.synchronize()
                    self.total_iterations += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.size = int(self.size * 0.9)
                        print(f"\nReducing matrix size to {self.size} due to OOM error")
                        torch.cuda.empty_cache()
                    else:
                        raise e
        except Exception as e:
            print(f"\nWorker error: {e}")
            self.running = False

    def _save_summary(self):
        summary = {
            "test_pattern": self.pattern.name,
            "duration": self._get_runtime(),
            "total_iterations": self.total_iterations,
            "peak_temperature": self.peak_temperature,
            "peak_memory": self.peak_memory,
            "peak_utilization": self.peak_utilization,
            "throttle_events": self.throttle_events,
            "matrix_size": self.size,
            "num_threads": self.num_threads,
        }
        with open(self.json_file, "w") as f:
            json.dump(summary, f, indent=4)

    def run(self):
        print(f"\nGPU Stress Test\n--------------")
        print(f"Device: {self.device_str}")
        if self.device_str.startswith("cuda"):
            print(f"GPU: {torch.cuda.get_device_name(self.gpu_index)}")
        print(f"Test Pattern: {self.pattern.name}")
        print(f"Initial matrix size: {self.size}x{self.size}")
        print(f"Number of threads: {self.num_threads}")
        print(f"Logging to: {self.log_dir}\n--------------\n")
        self.start_time = datetime.now()
        self.monitor_thread.start()
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            workers = [
                executor.submit(self._stress_worker) for _ in range(self.num_threads)
            ]
        for worker in workers:
            worker.result()
        self._save_summary()


def main():
    available_devices = get_available_gpus()
    parser = argparse.ArgumentParser(
        description="Advanced GPU Stress Test using PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=available_devices,
        default=available_devices[0],
        help="Device to run the test on",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available devices and exit"
    )
    parser.add_argument(
        "--size", type=int, default=32_000, help="Initial size of the square matrices"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of concurrent stress test threads",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=[p.name for p in TestPattern],
        default="COMPUTE_BOUND",
        help="Test pattern to use",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="gpu_stress_logs",
        help="Directory for logging data",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable devices:")
        for i, device in enumerate(available_devices):
            if device.startswith("cuda"):
                print(f"{device}: {torch.cuda.get_device_name(i)}")
            else:
                print(f"{device}: CPU")
        sys.exit(0)

    try:
        stress_test = GPUStressTest(
            device=args.device,
            size=args.size,
            num_threads=args.threads,
            pattern=TestPattern[args.pattern],
            log_dir=args.log_dir,
        )
        stress_test.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
