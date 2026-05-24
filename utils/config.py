import os
import yaml
from typing import Any

# Disable optional Pydantic plugin discovery to avoid intermittent startup failures
# caused by corrupted/invalid third-party entry point metadata in some environments.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

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
    whisper_model: str = "medium.en"
    tts_voice: str = "en-GB-RyanNeural"


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = "auto"
    whisper_model: str = "auto"
    whisper_compute_type: str = "auto"
    embedding_device: str = "auto"
    vram_gb: float = 0.0


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: str = ""


class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    require_confirm_power_actions: bool = True
    confirm_risky_tools: bool = True
    allowed_apps: list[str] = Field(default_factory=list)
    allowed_file_roots: list[str] = Field(default_factory=list)
    tool_timeout_sec: float = 10.0


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent_catalog_path: str = "intent_catalog.md"
    intent_top_k: int = 3
    intent_min_score: float = 0.72
    personal_roots: list[str] = Field(
        default_factory=lambda: [
            "%USERPROFILE%/Documents",
            "%USERPROFILE%/Desktop",
            "%USERPROFILE%/Projects",
        ]
    )
    chunk_size: int = 600
    chunk_overlap: int = 100
    refresh_seconds: int = 1800
    persist_directory: str = "./memory_db"
    multi_user_enabled: bool = True
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "auto"
    index_schema_version: int = 2
    max_context_chars: int = 3000
    batch_size: int = 256
    max_embedding_threads: int = 4
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            ".venv", "__pycache__", ".git", ".pytest_cache",
            "ipynb_checkpoints", ".DS_Store", ".mypy_cache",
            "node_modules", "site-packages", "memory_db", "AppData",
            "Downloads", "Pictures", "Music", "Videos", "Temp",
            "logs", ".egg-info", "dist", "build", ".tox", "*.pyc",
            "*.sqlite",
        ]
    )
    # New user-tunable RAG settings
    minimum_relevance_score: float = Field(
        default=45.0,
        description="Minimum final score required for a RAG result to be kept after filtering.",
    )
    max_results: int = 5
    excerpt_max_chars: int = 450
    boost_cap: float = Field(
        default=30.0,
        description="Maximum bonus score that filename and path boosting may add to a RAG hit.",
    )
    refresh_only_when_idle: bool = True
    refresh_idle_threshold_seconds: float = Field(
        default=30.0,
        description="Minimum idle window, in seconds, before background RAG refresh is allowed.",
    )
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_candidates: int = 12
    final_results: int = 5
    background_batch_size: int = 64
    background_batch_sleep_seconds: float = 2.0
    background_embedding_threads: int = 2
    cpu_throttle_threshold_percent: int = 65
    reranker_startup_wait_seconds: float = Field(
        default=30.0,
        description="Minimum runtime before the reranker is allowed to initialize.",
    )
    filename_partial_ratio_threshold: float = Field(
        default=70.0,
        description="Minimum fuzzy partial-ratio score required to award a filename match bonus.",
    )
    filename_fuzzy_match_threshold: float = Field(
        default=75.0,
        description="Minimum fuzzy token-set score required to accept a filename-only project match.",
    )
    filename_parent_match_threshold: float = Field(
        default=65.0,
        description="Minimum fuzzy score required to boost a parent-folder or path-component match.",
    )
    filename_score_spread_threshold: float = Field(
        default=5.0,
        description="Allowed score spread between top matching blocks before document excerpts are blended.",
    )
    result_score_vector_weight: float = Field(
        default=0.65,
        description="Weight applied to vector similarity when composing the final RAG score.",
    )
    result_score_text_weight: float = Field(
        default=0.30,
        description="Weight applied to text similarity when composing the final RAG score.",
    )
    result_score_importance_weight: float = Field(
        default=0.05,
        description="Weight applied to metadata importance when composing the final RAG score.",
    )
    source_penalty_factor: float = Field(
        default=0.55,
        description="Multiplier applied to deprioritized files such as tests and diagnostics.",
    )
    feedback_penalty: float = Field(
        default=0.15,
        description="Fraction of score removed when a source has repeated negative retrieval feedback.",
    )
    project_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum persisted project confidence required before using the project as a retrieval hint.",
    )
    document_confidence_low_threshold: float = Field(
        default=0.5,
        description="Below this confidence, document answers should ask for clarification.",
    )
    document_confidence_session_threshold: float = Field(
        default=0.65,
        description="At or above this confidence, a resolved document should be stored as the active project.",
    )
    document_confidence_summary_threshold: float = Field(
        default=0.95,
        description="Below this confidence, summary requests may blend multiple matching files.",
    )


class WakeBehaviorConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_seconds: int = 30
    match_mode: str = "prefix"
    min_confidence: float = 0.86
    max_prefix_tokens: int = 4


class ActivationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "smart"
    wake_word: str = "dexter"
    active_hours_start: str = "09:00"
    active_hours_end: str = "18:00"
    active_days: list[int] | None = None
    always_on_after_n_interactions: int = 3
    always_on_window_seconds: float = 120.0
    always_on_timeout_seconds: float = 300.0
    clap_sensitivity: float = 3.0
    active_window_seconds: int = 30
    start_active: bool = False
    min_command_words: int = 2
    fallback_to_always_on_after_failures: int = 3
    wake_words: list[str] = Field(default_factory=lambda: ["hey", "hey dexter", "dexter"])


class MediaConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_platform: str = "spotify"
    spotify_client_id: str = ""
    spotify_client_secret: str = ""


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gemini_daily_quota_cooldown_hours: float = 24.0
    groq_max_tools: int = 10
    ollama_timeout_seconds: float = 25.0
    overall_turn_timeout_seconds: float = 30.0
    # Maximum age, in seconds, after which a provider cooldown is considered stale and cleared on load
    max_cooldown_age_sec: int = 3600


class AudioSettingsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_rate: int = 16000
    chunk_size: int = 512
    whisper_batch_size: int = 16
    device_index: int | None = None
    vad_threshold: float = 0.25
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 900
    speech_pad_ms: int = 400
    max_speech_duration_s: int = 30


class SpeedConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    whisper_beam_size: int = 1


class SttConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    beam_size: int = 5
    best_of: int = 5
    temperature: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4])
    patience: float = 1.0
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    initial_prompt: str = (
        "Common Windows assistant commands and app names: open, close, start, launch, "
        "play, watch, search, find, what is, what's, weather, forecast, time, date, "
        "take screenshot, screen capture, read clipboard, copy to clipboard, Chrome, Google Chrome, "
        "Edge, Microsoft Edge, Firefox, Brave, Spotify, Discord, Notepad, Calculator, Settings, "
        "File Explorer, Visual Studio Code, VS Code, PowerShell, Command Prompt, Windows Terminal, Outlook, Word, Excel, PowerPoint."
    )


class HistoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_tokens: int = 1800


class McpConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    server_script: str = "mcp_server/dexter_mcp_server.py"
    timeout_seconds: float = 15.0
    start_delay_seconds: float = 5.0
    max_calls_per_minute: int = 30


class ProactiveConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    reminder_check_seconds: int = 60
    system_status_interval_seconds: int = 900


class HealthPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_rag_staleness_hours: float = 168.0
    min_vram_gb: float = 2.0
    max_provider_failure_rate: float = 0.5


class RuntimeConfig(BaseModel):
    """Runtime flags set by low-power mode detection."""

    model_config = ConfigDict(extra="ignore")

    disable_rag_warming: bool = False
    disable_proactive_mode: bool = False
    expect_cuda: bool = False


class PrivacyConfig(BaseModel):
    """Privacy-related runtime flags."""

    model_config = ConfigDict(extra="ignore")

    # If enabled, log full user transcripts to disk (logs/dexter.log).
    # This can contain sensitive personal voice content.
    debug_log_transcripts: bool = False


class VisionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capture_timeout: float = 5.0


class AutomationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    focus_wait_ms: int = 300
    post_action_verify: bool = True
    max_focus_retries: int = 3


class SpotifyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8888/callback"


class BriefingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    news_sources: list[str] = Field(default_factory=lambda: ["technology", "world"])
    include_weather: bool = True
    include_agenda: bool = True


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
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    audio_settings: AudioSettingsConfig = Field(default_factory=AudioSettingsConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    health_policy: HealthPolicy = Field(default_factory=HealthPolicy)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    spotify: SpotifyConfig = Field(default_factory=SpotifyConfig)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)

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
    config.setdefault("hardware", {})
    config.setdefault("security", {})
    config.setdefault("providers", {})
    config.setdefault("audio_settings", {})
    config.setdefault("speed", {})
    config.setdefault("stt", {})
    config.setdefault("activation", {})
    config.setdefault("rag", {})
    config.setdefault("mcp", {})
    config.setdefault("proactive", {})
    config.setdefault("runtime", {})
    config.setdefault("privacy", {})
    config.setdefault("vision", {})
    config.setdefault("health_policy", {})
    config.setdefault("automation", {})
    config.setdefault("media", {})
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
    
    # Apply low-power mode overrides if hardware is weak
    try:
        from utils.hardware_detect import apply_low_power_overrides
        config = apply_low_power_overrides(config)
    except Exception as e:
        logger.warning(f"Failed to apply hardware detection: {e}")
    
    api = config.pop("api_keys", {})
    config["gemini_api_key"] = os.getenv("GEMINI_API_KEY") or (api.get("gemini") or "")
    config["groq_api_key"] = os.getenv("GROQ_API_KEY") or (api.get("groq") or "")
    
    spotify = config.setdefault("spotify", {})
    spotify["client_id"] = os.getenv("SPOTIFY_CLIENT_ID") or spotify.get("client_id", "")
    spotify["client_secret"] = os.getenv("SPOTIFY_CLIENT_SECRET") or spotify.get("client_secret", "")
    spotify["redirect_uri"] = os.getenv("SPOTIFY_REDIRECT_URI") or spotify.get("redirect_uri", "http://localhost:8888/callback")
    
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
