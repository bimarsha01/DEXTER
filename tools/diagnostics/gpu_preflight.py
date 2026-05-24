from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from utils.config import get_config


LOGGER_NAME = "gpu_preflight"


@dataclass
class GpuInfo:
    name: str
    driver_version: str
    vram_gb: float
    compute_cap: str


@dataclass
class TorchInfo:
    version: str
    is_cpu_only: bool



def _configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger



def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)



def _parse_gpu_output(output: str) -> GpuInfo:
    line = output.strip().splitlines()[0].strip()
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        raise ValueError(f"Unexpected nvidia-smi output: {output.strip()}")

    name = parts[0]
    driver_version = parts[1]
    compute_cap = parts[-1]
    vram_text = ",".join(parts[2:-1]).strip()

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", vram_text.replace(",", ""))
    if not match:
        raise ValueError(f"Could not parse VRAM from: {vram_text}")
    vram_value = float(match.group(1))
    if "gib" in vram_text.lower() or "gb" in vram_text.lower():
        vram_gb = vram_value
    else:
        vram_gb = vram_value / 1024.0

    return GpuInfo(name=name, driver_version=driver_version, vram_gb=vram_gb, compute_cap=compute_cap)



def _parse_driver_version(driver_version: str) -> Optional[float]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", driver_version)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None



def _get_gpu_info(logger: logging.Logger) -> GpuInfo:
    if shutil.which("nvidia-smi") is None:
        logger.error("nvidia-smi not found — CUDA is not available on this machine")
        raise SystemExit(1)

    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ]
    completed = _run_command(command)
    if completed.returncode != 0:
        logger.error("nvidia-smi failed: %s", (completed.stderr or completed.stdout or "unknown error").strip())
        raise SystemExit(1)

    logger.info("nvidia-smi output: %s", completed.stdout.strip())
    gpu_info = _parse_gpu_output(completed.stdout)

    logger.info("GPU name: %s", gpu_info.name)
    logger.info("Driver version: %s", gpu_info.driver_version)
    logger.info("Total VRAM: %.2f GB", gpu_info.vram_gb)
    logger.info("Compute capability: %s", gpu_info.compute_cap)
    return gpu_info



def _check_driver(logger: logging.Logger, driver_version: str) -> float:
    parsed = _parse_driver_version(driver_version)
    if parsed is None:
        logger.warning("Could not parse driver version '%s'", driver_version)
        return 0.0

    if parsed < 520.0:
        logger.warning("Driver < 520.0 — recommend updating to 535+ for CUDA 12.x compatibility")
    else:
        logger.info("Driver OK for CUDA 12.x")
    return parsed



def _check_torch(logger: logging.Logger) -> TorchInfo:
    completed = _run_command([sys.executable, "-m", "pip", "show", "torch"])
    if completed.returncode != 0:
        logger.info("PyTorch not installed")
        return TorchInfo(version="", is_cpu_only=False)

    version = ""
    for line in completed.stdout.splitlines():
        if line.lower().startswith("version:"):
            version = line.split(":", 1)[1].strip()
            break

    if not version:
        logger.info("PyTorch installed but version could not be parsed")
        return TorchInfo(version="", is_cpu_only=False)

    logger.info("PyTorch version: %s", version)
    is_cpu_only = "+cpu" in version.lower() or "+cu" not in version.lower()
    if is_cpu_only:
        logger.warning("CPU-only PyTorch detected — must uninstall before CUDA install")
        uninstall = _run_command([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
        if uninstall.returncode != 0:
            logger.error("Failed to uninstall CPU PyTorch: %s", (uninstall.stderr or uninstall.stdout or "unknown error").strip())
            raise SystemExit(1)
        logger.info("CPU PyTorch removed")
    return TorchInfo(version=version, is_cpu_only=is_cpu_only)



def _check_vram(logger: logging.Logger, vram_gb: float) -> None:
    if vram_gb < 4.0:
        logger.warning("Low VRAM — Whisper will use tiny model, RAG embeddings will use CPU")
    elif vram_gb <= 8.0:
        logger.info("Mid VRAM — Whisper will use base/small model")
    else:
        logger.info("High VRAM — full model stack available")



def _recommended_cuda_version(driver_version: float) -> str:
    return "12.1" if driver_version >= 520.0 else "11.8"


def gpu_diagnostic() -> dict[str, object]:
    logger = _configure_logging()
    cfg = get_config()
    hardware = getattr(cfg, "hardware", None)

    device = str(getattr(hardware, "device", "cpu") or "cpu")
    whisper_model = str(getattr(hardware, "whisper_model", "tiny") or "tiny")
    whisper_compute_type = str(getattr(hardware, "whisper_compute_type", "float32") or "float32")
    embedding_device = str(getattr(hardware, "embedding_device", "cpu") or "cpu")

    vram_used_gb = 0.0
    vram_total_gb = float(getattr(hardware, "vram_gb", 0.0) or 0.0)

    try:
        import torch

        if torch.cuda.is_available():
            try:
                vram_used_gb = float(torch.cuda.memory_allocated(0)) / (1024 ** 3)
            except Exception:
                vram_used_gb = 0.0
            try:
                vram_total_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
            except Exception:
                pass
    except Exception:
        pass

    message = (
        f"Hardware state: device={device}, VRAM={vram_used_gb:.2f}GB/{vram_total_gb:.2f}GB, "
        f"Whisper={whisper_model} ({whisper_compute_type}), Embeddings={embedding_device}"
    )
    print(message)
    logger.info(
        "gpu_diagnostic",
        device=device,
        vram_used_gb=round(vram_used_gb, 2),
        vram_total_gb=round(vram_total_gb, 2),
        whisper_model=whisper_model,
        whisper_compute_type=whisper_compute_type,
        embedding_device=embedding_device,
    )
    return {
        "device": device,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "whisper_model": whisper_model,
        "whisper_compute_type": whisper_compute_type,
        "embedding_device": embedding_device,
    }



def main() -> int:
    logger = _configure_logging()

    gpu_info = _get_gpu_info(logger)
    driver_numeric = _check_driver(logger, gpu_info.driver_version)
    _check_torch(logger)
    _check_vram(logger, gpu_info.vram_gb)

    cuda_version = _recommended_cuda_version(driver_numeric)
    summary = (
        f"Pre-flight complete: {gpu_info.name}, "
        f"{gpu_info.vram_gb:.0f}GB VRAM, driver {gpu_info.driver_version}, ready for CUDA {cuda_version}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
