import os
import yaml
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from utils.logger import get_logger

logger = get_logger("config")

# Workspace root (DEXTER project directory)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ModelsConfig(BaseModel):
    """LLM and speech model identifiers from config.yaml `models`."""

    model_config = ConfigDict(extra="ignore")

    primary_llm: str = "gemini-2.0-flash"
    fallback_llm: str = "llama-3.3-70b-versatile"
    local_llm: str = "qwen3-coder:480b-cloud"
    whisper_model: str = "small.en"
    tts_voice: str = "en-GB-RyanNeural"


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: str = "Kathmandu"


class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    require_confirm_power_actions: bool = True
    allowed_apps: list[str] = Field(default_factory=list)
    allowed_file_roots: list[str] = Field(default_factory=list)
    tool_timeout_sec: float = 10.0


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent_catalog_path: str = "intent_catalog.md"
    intent_top_k: int = 3
    intent_min_score: float = 0.72


class WakeBehaviorConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_seconds: int = 30
    match_mode: str = "prefix"
    min_confidence: float = 0.86
    max_prefix_tokens: int = 4


class ActivationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "wake_word"
    clap_sensitivity: float = 3.0
    active_window_seconds: int = 30
    start_active: bool = False
    min_command_words: int = 2
    wake_words: list[str] = Field(default_factory=lambda: ["hey", "hey dexter"])


class AudioSettingsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_rate: int = 16000
    chunk_size: int = 512
    device_index: int | None = None


class SpeedConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    whisper_beam_size: int = 1


class SttConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    beam_size: int = 3
    best_of: int = 5
    temperature: float = 0.0
    patience: float = 1.0
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    initial_prompt: str = (
        "Common Windows assistant commands and app names: open, close, start, launch, "
        "take screenshot, screen capture, set volume, clipboard, Chrome, Google Chrome, "
        "Edge, Microsoft Edge, Firefox, Spotify, Discord, Notepad, Calculator, Settings, "
        "File Explorer, Visual Studio Code, VS Code."
    )


class HistoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_tokens: int = 1800


class McpConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False


class DexterConfig(BaseModel):
    """
    Runtime configuration loaded from config.yaml plus .env (API keys).
    Use get_config() — do not instantiate directly.
    """

    model_config = ConfigDict(extra="ignore")

    bot_name: str = "Dexter"
    wake_words: list[str] = Field(default_factory=lambda: ["hey"])
    wake_behavior: WakeBehaviorConfig = Field(default_factory=WakeBehaviorConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    audio_settings: AudioSettingsConfig = Field(default_factory=AudioSettingsConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)

    gemini_api_key: str = ""
    groq_api_key: str = ""

    @field_validator("wake_words", mode="before")
    @classmethod
    def _normalize_wake_words(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        return value or ["hey"]

    def validate_runtime(self) -> None:
        if not self.gemini_api_key and not self.groq_api_key:
            raise ValueError("At least one API key must be configured in the environment.")

        for root in self.security.allowed_file_roots:
            expanded = os.path.expandvars(os.path.expanduser(str(root)))
            if not os.path.exists(expanded):
                raise ValueError(f"Allowed root does not exist: {expanded}")

        if self.audio_settings.device_index is not None:
            try:
                import sounddevice as sd

                devices = sd.query_devices()
                idx = self.audio_settings.device_index
                if idx < 0 or idx >= len(devices):
                    raise ValueError(f"Audio device index out of range: {idx}")
            except Exception as e:
                raise ValueError(f"Audio device validation failed: {e}") from e


_CONFIG: DexterConfig | None = None


def _ensure_config_shape(config: dict) -> dict:
    config = config or {}
    config.setdefault("api_keys", {})
    config.setdefault("models", {})
    config.setdefault("security", {})
    config.setdefault("audio_settings", {})
    config.setdefault("speed", {})
    config.setdefault("stt", {})
    config.setdefault("activation", {})
    config.setdefault("mcp", {})
    return config


def _inject_api_keys(config: dict) -> None:
    api_keys = config.setdefault("api_keys", {})
    env_gemini = os.getenv("GEMINI_API_KEY")
    env_groq = os.getenv("GROQ_API_KEY")

    if env_gemini:
        api_keys["gemini"] = env_gemini
    else:
        current = (api_keys.get("gemini") or "").strip()
        if current.upper().startswith(("YOUR", "GEMINI")):
            api_keys["gemini"] = ""

    if env_groq:
        api_keys["groq"] = env_groq
    else:
        current = (api_keys.get("groq") or "").strip()
        if current.upper().startswith(("YOUR", "GROQ")):
            api_keys["groq"] = ""


def _load_raw_config() -> dict:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT_DIR, ".env"))

    config_path = os.path.join(ROOT_DIR, "config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except FileNotFoundError:
        logger.error("config_file_not_found", path=config_path)
        return {}

    config = _ensure_config_shape(config)
    _inject_api_keys(config)
    api = config.pop("api_keys", {})
    config["gemini_api_key"] = os.getenv("GEMINI_API_KEY") or (api.get("gemini") or "")
    config["groq_api_key"] = os.getenv("GROQ_API_KEY") or (api.get("groq") or "")
    return config


def config_validation_warnings(cfg: DexterConfig) -> list[str]:
    """Non-fatal configuration warnings (missing optional keys, etc.)."""
    warnings: list[str] = []
    if not cfg.gemini_api_key:
        warnings.append("Gemini API key is not configured. Set GEMINI_API_KEY in .env or environment.")
    if not cfg.groq_api_key:
        warnings.append("Groq API key is not configured. Set GROQ_API_KEY in .env or environment.")
    activation_mode = (cfg.activation.mode or "wake_word").strip().lower()
    if activation_mode == "wake_word" and not cfg.activation.wake_words:
        warnings.append("Wake words are missing for wake_word mode. Add activation.wake_words in config.yaml.")
    return warnings


def get_config() -> DexterConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    raw = _load_raw_config()
    try:
        _CONFIG = DexterConfig.model_validate(raw)
        _CONFIG.validate_runtime()
        return _CONFIG
    except ValidationError as e:
        raise RuntimeError(f"Dexter config validation failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Dexter config runtime validation failed: {e}") from e


def get_workspace_root() -> str:
    """Return the workspace root directory (DEXTER)."""
    return ROOT_DIR
