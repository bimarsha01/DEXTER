"""
Hardware detection and low-power mode profile selection.

Detects system resources (memory, CPU, GPU) and auto-selects
lightweight settings for weak hardware.
"""

import os
import psutil
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class HardwareProfile(TypedDict):
    """Profile of detected hardware resources."""
    available_memory_gb: float
    cpu_count: int
    has_gpu: bool
    is_weak_system: bool
    profile_name: str  # "powerful", "standard", or "lightweight"


class LowPowerSettings(TypedDict):
    """Configuration overrides for low-power mode."""
    whisper_model: str
    chunk_size: int
    batch_size: int
    max_embedding_threads: int
    disable_rag_warming: bool
    disable_proactive_mode: bool


# Hardware thresholds for "weak" systems
WEAK_SYSTEM_THRESHOLDS = {
    "min_memory_gb": 4.0,        # < 4 GB RAM = weak
    "min_cpu_count": 2,          # < 2 cores = weak
}

# Profile settings: lower numbers = less resource-intensive
PROFILES = {
    "lightweight": LowPowerSettings(
        whisper_model="tiny.en",
        chunk_size=300,
        batch_size=32,
        max_embedding_threads=1,
        disable_rag_warming=True,
        disable_proactive_mode=True,
    ),
    "standard": LowPowerSettings(
        whisper_model="base.en",
        chunk_size=500,
        batch_size=128,
        max_embedding_threads=2,
        disable_rag_warming=False,
        disable_proactive_mode=False,
    ),
    "powerful": LowPowerSettings(
        whisper_model="small.en",
        chunk_size=600,
        batch_size=256,
        max_embedding_threads=4,
        disable_rag_warming=False,
        disable_proactive_mode=False,
    ),
}


def detect_hardware() -> HardwareProfile:
    """
    Detect system hardware resources.

    Returns:
        HardwareProfile with detected resources and classification.
    """
    try:
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        # Get logical CPU count
        cpu_count = psutil.cpu_count(logical=True) or 1
        
        # Attempt to detect GPU
        # Check for NVIDIA GPU via CUDA_VISIBLE_DEVICES env var or nvidia_smi
        has_gpu = _detect_gpu()
        
        # Determine if system is "weak"
        is_weak = (
            available_memory_gb < WEAK_SYSTEM_THRESHOLDS["min_memory_gb"]
            or cpu_count < WEAK_SYSTEM_THRESHOLDS["min_cpu_count"]
        )
        
        # Select profile: lightweight if weak, powerful if GPU, standard otherwise
        if is_weak:
            profile_name = "lightweight"
        elif has_gpu:
            profile_name = "powerful"
        else:
            profile_name = "standard"
        
        profile = HardwareProfile(
            available_memory_gb=available_memory_gb,
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            is_weak_system=is_weak,
            profile_name=profile_name,
        )
        
        logger.info(
            f"Hardware detected: {available_memory_gb:.1f} GB RAM, "
            f"{cpu_count} CPU cores, GPU={has_gpu} → {profile_name} profile"
        )
        
        return profile
        
    except Exception as e:
        logger.warning(f"Failed to detect hardware: {e}; defaulting to standard profile")
        return HardwareProfile(
            available_memory_gb=8.0,
            cpu_count=4,
            has_gpu=False,
            is_weak_system=False,
            profile_name="standard",
        )


def _detect_gpu() -> bool:
    """
    Attempt to detect GPU presence.
    
    Returns:
        True if GPU is likely available.
    """
    # Check for CUDA via environment
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return True
    
    # Try importing torch and checking for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    
    return False


def get_low_power_settings(profile_name: str = None) -> LowPowerSettings:
    """
    Get configuration settings for a given profile.

    Args:
        profile_name: "lightweight", "standard", or "powerful".
                      If None, auto-detects hardware and selects profile.

    Returns:
        LowPowerSettings dict with overrides to apply.
    """
    if profile_name is None:
        profile = detect_hardware()
        profile_name = profile["profile_name"]
    
    if profile_name not in PROFILES:
        logger.warning(
            f"Unknown profile {profile_name}; using standard"
        )
        profile_name = "standard"
    
    return PROFILES[profile_name]


def apply_low_power_overrides(config_dict: dict) -> dict:
    """
    Apply low-power mode overrides to a config dict.

    Args:
        config_dict: Loaded config (e.g., from YAML or environment).

    Returns:
        Modified config_dict with low-power overrides applied.
    """
    # Detect hardware and get settings
    settings = get_low_power_settings()
    
    # Apply overrides to models section
    if "models" not in config_dict:
        config_dict["models"] = {}
    
    config_dict["models"]["whisper_model"] = settings["whisper_model"]
    
    # Apply overrides to RAG section
    if "rag" not in config_dict:
        config_dict["rag"] = {}
    
    config_dict["rag"]["chunk_size"] = settings["chunk_size"]
    config_dict["rag"]["batch_size"] = settings["batch_size"]
    config_dict["rag"]["max_embedding_threads"] = settings["max_embedding_threads"]
    
    # Store flags for runtime use (main.py can check these)
    if "runtime" not in config_dict:
        config_dict["runtime"] = {}
    
    config_dict["runtime"]["disable_rag_warming"] = settings["disable_rag_warming"]
    config_dict["runtime"]["disable_proactive_mode"] = settings["disable_proactive_mode"]
    
    return config_dict
