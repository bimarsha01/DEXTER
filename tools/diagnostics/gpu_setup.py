from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


LOGGER_NAME = "gpu_setup"
PIP_TIMEOUT_SECONDS = 600


@dataclass
class GpuInfo:
    name: str
    driver_version: str
    vram_gb: float
    compute_cap: str


@dataclass
class SetupPlan:
    cuda_version: str
    driver_version: float
    gpu_name: str
    vram_gb: float
    compute_cap: str



def _configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger



def _run_command(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check)



def _run_streaming_command(command: list[str], logger: logging.Logger, timeout_seconds: int = PIP_TIMEOUT_SECONDS) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_lines: list[str] = []

    def _stream_output() -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            output_lines.append(line)
            print(line, end="")

    reader = threading.Thread(target=_stream_output, daemon=True)
    reader.start()

    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=10)
        except Exception:
            pass
        reader.join(timeout=5)
        logger.error("PyTorch install timed out — check network or try again")
        raise SystemExit(1)
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass
        reader.join(timeout=5)

    if return_code != 0:
        logger.error("Command failed with exit code %s", return_code)
        if output_lines:
            logger.error("Last output: %s", output_lines[-1].strip())
        raise SystemExit(return_code)

    return return_code



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



def _select_cuda_version(driver_version: float) -> str:
    if driver_version >= 525.0:
        return "12.1"
    if driver_version >= 450.0:
        return "11.8"
    logger.error("Driver too old for any CUDA PyTorch — minimum driver is 450.0")
    raise SystemExit(1)



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



def _check_torch(logger: logging.Logger) -> Optional[str]:
    completed = _run_command([sys.executable, "-m", "pip", "show", "torch"])
    if completed.returncode != 0:
        logger.info("PyTorch not installed")
        return None

    version = ""
    for line in completed.stdout.splitlines():
        if line.lower().startswith("version:"):
            version = line.split(":", 1)[1].strip()
            break

    if not version:
        logger.info("PyTorch installed but version could not be parsed")
        return None

    logger.info("PyTorch version: %s", version)
    is_cpu_only = "+cpu" in version.lower() or "+cu" not in version.lower()
    if is_cpu_only:
        logger.warning("CPU-only PyTorch detected — must uninstall before CUDA install")
        uninstall = _run_command([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
        if uninstall.returncode != 0:
            logger.error("Failed to uninstall CPU PyTorch: %s", (uninstall.stderr or uninstall.stdout or "unknown error").strip())
            raise SystemExit(1)
        logger.info("CPU PyTorch removed")
    return version



def _check_vram(logger: logging.Logger, vram_gb: float) -> None:
    if vram_gb < 4.0:
        logger.warning("Low VRAM — Whisper will use tiny model, RAG embeddings will use CPU")
    elif vram_gb <= 8.0:
        logger.info("Mid VRAM — Whisper will use base/small model")
    else:
        logger.info("High VRAM — full model stack available")



def _install_torch(cuda_version: str, logger: logging.Logger) -> None:
    logger.info("PyTorch install started — this may take several minutes (2–4 GB download)")
    if cuda_version == "12.1":
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]

    _run_streaming_command(command, logger, timeout_seconds=PIP_TIMEOUT_SECONDS)



def _verify_torch_cuda(logger: logging.Logger) -> None:
    verify_code = (
        "import torch\n"
        "print(torch.__version__)\n"
        "print(torch.cuda.is_available())\n"
        "print(torch.cuda.get_device_name(0))\n"
        "print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n"
    )
    completed = _run_command([sys.executable, "-c", verify_code])
    if completed.returncode != 0:
        logger.error("PyTorch verification failed: %s", (completed.stderr or completed.stdout or "unknown error").strip())
        raise SystemExit(1)

    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in output_lines:
        print(line)

    if len(output_lines) < 4:
        logger.error("PyTorch verification output was incomplete")
        raise SystemExit(1)

    version = output_lines[0]
    cuda_available = output_lines[1].lower() == "true"
    device_name = output_lines[2]
    vram_line = output_lines[3]

    if not cuda_available:
        logger.error("CUDA install succeeded but CUDA is not available — possible DLL conflict or wrong CUDA version")
        raise SystemExit(1)

    logger.info("PyTorch CUDA verified — %s, %s", device_name, vram_line.replace("VRAM:", "").strip())
    logger.info("Verified torch version: %s", version)



def _install_faster_whisper(logger: logging.Logger) -> None:
    _run_streaming_command([sys.executable, "-m", "pip", "install", "faster-whisper"], logger, timeout_seconds=PIP_TIMEOUT_SECONDS)
    _run_streaming_command(
        [sys.executable, "-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
        logger,
        timeout_seconds=PIP_TIMEOUT_SECONDS,
    )



def _verify_faster_whisper(logger: logging.Logger) -> None:
    verify_code = (
        "from faster_whisper import WhisperModel\n"
        "model = WhisperModel('tiny', device='cuda', compute_type='float16')\n"
        "print('faster-whisper CUDA OK')\n"
    )
    completed = _run_command([sys.executable, "-c", verify_code])
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "unknown error").strip()
        logger.error("faster-whisper verification failed: %s", message)
        raise SystemExit(1)

    output = (completed.stdout or "").strip()
    if output:
        print(output)



def _update_requirements_file(logger: logging.Logger) -> None:
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "requirements.txt")
    try:
        with open(requirements_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        header = [
            "# NOTE: PyTorch must be installed manually with CUDA support.\n",
            "# Run: python tools/diagnostics/gpu_setup.py\n",
            "# Do NOT run plain: pip install torch (installs CPU version)\n",
            "\n",
        ]

        updated_lines: list[str] = []
        torch_line_replaced = False
        for line in lines:
            stripped = line.strip()
            if stripped == "torch":
                updated_lines.append("torch  # installed via CUDA wheel — see gpu_setup.py for install command\n")
                torch_line_replaced = True
                continue
            updated_lines.append(line)

        if not torch_line_replaced:
            updated_lines.append("torch  # installed via CUDA wheel — see gpu_setup.py for install command\n")

        with open(requirements_path, "w", encoding="utf-8") as file:
            file.writelines(header + updated_lines)

        logger.info("requirements.txt updated")
    except Exception as exc:
        logger.error("Failed to update requirements.txt: %s", str(exc))
        raise SystemExit(1)



def main() -> int:
    logger = _configure_logging()
    gpu_info = _get_gpu_info(logger)
    driver_numeric = _check_driver(logger, gpu_info.driver_version)
    cuda_version = _select_cuda_version(driver_numeric)

    _check_torch(logger)
    _check_vram(logger, gpu_info.vram_gb)

    _install_torch(cuda_version, logger)
    _verify_torch_cuda(logger)
    _install_faster_whisper(logger)
    _verify_faster_whisper(logger)
    _update_requirements_file(logger)

    summary = (
        f"Pre-flight complete: {gpu_info.name}, "
        f"{gpu_info.vram_gb:.0f}GB VRAM, driver {gpu_info.driver_version}, ready for CUDA {cuda_version}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
