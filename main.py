"""
╔══════════════════════════════════════════════════════════════╗
║  DEXTER — AI Voice Assistant for Windows                     ║
║  A Jarvis-style local assistant powered by:                  ║
║    • Gemini (Primary) → Groq (Fallback) → Ollama (Local)    ║
║    • Whisper (Speech-to-Text on GPU)                         ║
║    • Edge-TTS (Text-to-Speech)                               ║
║    • Silero VAD (Voice Activity Detection)                   ║
║    • ChromaDB (Long-Term Vector Memory)                      ║
╚══════════════════════════════════════════════════════════════╝
"""
import asyncio
import atexit
import getpass
import os
import signal
import site
import ctypes
import time
from contextlib import suppress
from datetime import datetime
from utils.logger import get_logger
from utils.config import get_config, config_validation_warnings
from core.health import GPUStatus, RAGStatus, HealthMonitor, set_global_health_monitor
from core.event_bus import EventBus
from core.hardware_watchdog import HardwareWatchdog
from utils.lazy_loader import LazyLoader
from utils.user_profile import UserProfile
from utils.asr_corrections import ASRCorrectionEngine
from utils.vocabulary_builder import VocabularyBuilder
from tools.input_tools import can_automate
import threading
from core.brain.session_state import ContextStore, SessionContext
from core.feedback import FeedbackStore

logger = get_logger("main")

_WATCHDOG_STOP_EVENT = threading.Event()
_ACTIVE_WATCHDOG: HardwareWatchdog | None = None
_ACTIVE_PIPELINE = None
_ACTIVE_DASHBOARD = None
_CLEANUP_DONE = False


def _cleanup_runtime_sync() -> None:
    global _ACTIVE_WATCHDOG, _ACTIVE_PIPELINE, _ACTIVE_DASHBOARD, _CLEANUP_DONE

    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True

    watchdog = _ACTIVE_WATCHDOG
    pipeline = _ACTIVE_PIPELINE

    try:
        if watchdog is not None:
            watchdog.stop()
    except Exception as exc:
        logger.debug("watchdog_stop_failed", error=str(exc), exc_info=True)

    try:
        if pipeline is not None and hasattr(pipeline, "stop"):
            pipeline.stop()
    except Exception as exc:
        logger.debug("pipeline_stop_failed", error=str(exc), exc_info=True)

    try:
        dashboard = _ACTIVE_DASHBOARD
        if dashboard is not None and hasattr(dashboard, "stop"):
            dashboard.stop()
    except Exception as exc:
        logger.debug("dashboard_stop_failed", error=str(exc), exc_info=True)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Clean shutdown complete")


atexit.register(_cleanup_runtime_sync)


def _handle_sigint(signum, frame):
    _cleanup_runtime_sync()


signal.signal(signal.SIGINT, _handle_sigint)

# ─── Add Nvidia DLLs for faster-whisper CUDA support on Windows ──────────────
def _setup_cuda_dlls():
    """Setup CUDA DLL paths for faster-whisper GPU support."""
    cuda_paths = []
    
    # Add DLL directories for os.add_dll_directory (Windows 10.0.14286+)
    try:
        for sp in site.getsitepackages():
            for lib in ["cublas", "cudnn", "cublaslt", "nccl", "cusparse"]:
                bin_path = os.path.join(sp, "nvidia", lib, "bin")
                if os.path.exists(bin_path):
                    try:
                        os.add_dll_directory(bin_path)
                        logger.debug("cuda_dll_directory_added", lib=lib, path=bin_path)
                        cuda_paths.append(bin_path)
                    except Exception as e:
                        logger.debug("cuda_dll_directory_failed", lib=lib, error=str(e))
    except Exception as e:
        logger.debug("cuda_dll_setup_exception", error=str(e))
    
    # Also add to PATH for broader compatibility
    if cuda_paths:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ";".join(cuda_paths) + ";" + current_path
        logger.debug("cuda_paths_added_to_env", count=len(cuda_paths))

_setup_cuda_dlls()

class _DisabledVADListener:
    def listen(self, *args, **kwargs):
        logger.warning(
            "vad_disabled",
            reason="VAD import failed at startup; microphone listening is unavailable in this session",
        )
        return None


def _build_greeting(profile: UserProfile = None) -> str:
    """Build a contextual greeting based on time of day and username."""
    hour = datetime.now().hour
    if hour < 12:
        time_greeting = "Good morning"
    elif hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    name = profile.name if profile else ""
    if name:
        return f"{time_greeting}, {name}. Dexter is online and ready."
    return f"{time_greeting}. Dexter is online and ready."


def _format_gb(value: float) -> str:
    return f"{value:.0f}GB" if value >= 10 else f"{value:.1f}GB"


def _format_age(seconds: float) -> str:
    if seconds <= 0:
        return "just now"
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    if days >= 1:
        return f"{int(days)}d ago"
    if hours >= 1:
        return f"{int(hours)}h ago"
    if minutes >= 1:
        return f"{int(minutes)}m ago"
    return f"{int(seconds)}s ago"


def _provider_marks(provider_status: dict[str, str]) -> str:
    parts: list[str] = []
    for name, status in provider_status.items():
        normalized = str(status or "").strip().lower()
        mark = "✓" if normalized in {"ready", "available", "healthy", "ok", "true", "success"} else "✗"
        parts.append(f"{name.lower()}{mark}")
    return " ".join(parts)


def resolve_hardware_config(runtime_config) -> None:
    hardware = runtime_config.hardware

    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        total_vram_gb = 0.0
        if gpu_available:
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = float(props.total_memory) / (1024 ** 3)
    except Exception as exc:
        gpu_available = False
        total_vram_gb = 0.0
        logger.warning("hardware_probe_failed", error=str(exc), exc_info=True)

    requested_device = str(getattr(hardware, "device", "auto") or "auto").strip().lower()
    if requested_device == "auto":
        resolved_device = "cuda" if gpu_available else "cpu"
    elif requested_device in {"cuda", "cpu"}:
        resolved_device = requested_device if (requested_device == "cpu" or gpu_available) else "cpu"
    else:
        resolved_device = "cuda" if gpu_available else "cpu"

    requested_whisper_model = str(getattr(hardware, "whisper_model", "auto") or "auto").strip().lower()
    if requested_whisper_model == "auto":
        if total_vram_gb >= 8.0:
            resolved_whisper_model = "medium"
        elif total_vram_gb >= 4.0:
            resolved_whisper_model = "small"
        else:
            resolved_whisper_model = "tiny"
    else:
        resolved_whisper_model = requested_whisper_model

    requested_compute_type = str(getattr(hardware, "whisper_compute_type", "auto") or "auto").strip().lower()
    if requested_compute_type == "auto":
        if resolved_device == "cuda" and total_vram_gb >= 4.0:
            resolved_compute_type = "float16"
        elif resolved_device == "cuda":
            resolved_compute_type = "int8"
        else:
            resolved_compute_type = "float32"
    else:
        resolved_compute_type = requested_compute_type

    requested_embedding_device = str(getattr(hardware, "embedding_device", "auto") or "auto").strip().lower()
    if requested_embedding_device == "auto":
        resolved_embedding_device = "cuda" if gpu_available else "cpu"
    elif requested_embedding_device in {"cuda", "cpu"}:
        resolved_embedding_device = requested_embedding_device if (requested_embedding_device == "cpu" or gpu_available) else "cpu"
    else:
        resolved_embedding_device = "cuda" if gpu_available else "cpu"

    hardware.device = resolved_device
    hardware.whisper_model = resolved_whisper_model
    hardware.whisper_compute_type = resolved_compute_type
    hardware.embedding_device = resolved_embedding_device
    hardware.vram_gb = total_vram_gb if gpu_available else 0.0

    if hasattr(runtime_config, "rag"):
        runtime_config.rag.embedding_device = resolved_embedding_device

    logger.info(
        f"Hardware: {resolved_device}, Whisper: {resolved_whisper_model} ({resolved_compute_type}), Embeddings: {resolved_embedding_device}"
    )


def check_gpu_readiness(runtime_config, health_monitor: HealthMonitor | None = None) -> GPUStatus:
    expected_cuda = bool(getattr(getattr(runtime_config, "runtime", None), "expect_cuda", False))
    status = GPUStatus(expected_cuda=expected_cuda)
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        status.cuda_available = cuda_available
        if cuda_available:
            device_index = int(torch.cuda.current_device())
            device_name = str(torch.cuda.get_device_name(device_index))
            major, minor = torch.cuda.get_device_capability(device_index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                total_vram_gb = float(total_bytes) / (1024 ** 3)
                free_vram_gb = float(free_bytes) / (1024 ** 3)
            except Exception:
                props = torch.cuda.get_device_properties(device_index)
                total_vram_gb = float(props.total_memory) / (1024 ** 3)
                free_vram_gb = total_vram_gb

            status.status = "ready"
            status.device_name = device_name
            status.compute_capability = f"{major}.{minor}"
            status.total_vram_gb = total_vram_gb
            status.free_vram_gb = free_vram_gb
            status.details = "CUDA available"

            logger.info(
                "gpu_readiness_detected",
                device=device_name,
                compute_capability=status.compute_capability,
                total_vram_gb=round(total_vram_gb, 2),
                free_vram_gb=round(free_vram_gb, 2),
            )
            if total_vram_gb < 4.0:
                logger.warning(
                    "gpu_low_vram_warning",
                    total_vram_gb=round(total_vram_gb, 2),
                    detail="VRAM below 4GB may affect Whisper batch size",
                )
        else:
            status.status = "unavailable"
            status.details = "CUDA unavailable"
            if expected_cuda:
                logger.warning("gpu_unavailable", expected_cuda=expected_cuda)
                logger.error("cuda_expected_but_unavailable", expected_cuda=expected_cuda)
            else:
                logger.info("gpu_unavailable", expected_cuda=expected_cuda)
    except Exception as exc:
        status.status = "unavailable"
        status.details = str(exc)
        logger.error("gpu_readiness_check_failed", error=str(exc), exc_info=True)
        if expected_cuda:
            logger.error("cuda_expected_but_unavailable", error=str(exc))

    if health_monitor is not None:
        health_monitor.set_gpu_status(status)
    return status


def check_rag_readiness(memory_vault, runtime_config, health_monitor: HealthMonitor | None = None) -> RAGStatus:
    now = time.time()
    status = RAGStatus()
    try:
        rag_proxy = getattr(memory_vault, "personal_rag", None)
        status.index_exists = rag_proxy is not None

        if rag_proxy is None:
            status.status = "empty"
            disable_warming = bool(getattr(getattr(runtime_config, "runtime", None), "disable_rag_warming", False))
            status.details = "RAG warming is disabled at startup" if disable_warming else "RAG index is unavailable at startup"
            if health_monitor is not None:
                health_monitor.set_rag_status(status)
            return status

        if hasattr(rag_proxy, "is_ready") and not bool(getattr(rag_proxy, "is_ready")):
            status.status = "warming"
            status.details = "RAG index is still warming up"
            if health_monitor is not None:
                health_monitor.set_rag_status(status)
            return status

        rag_index = getattr(rag_proxy, "_index", rag_proxy)
        collection = getattr(rag_index, "_collection", None)
        doc_count = 0
        if collection is not None:
            try:
                doc_count = int(collection.count() or 0)
            except Exception:
                doc_count = 0
        elif hasattr(rag_index, "get_all_indexed_filenames"):
            try:
                doc_count = len(rag_index.get_all_indexed_filenames())
            except Exception:
                doc_count = 0

        status.doc_count = doc_count
        embedding_provider = str(getattr(getattr(rag_index, "_embedding_profile", None), "provider", "") or "")
        status.embedding_model_loaded = bool(getattr(rag_index, "_ef", None) is not None or embedding_provider == "chromadb-default")
        status.last_updated_ts = float(getattr(rag_index, "_last_refresh", 0.0) or 0.0)

        if doc_count <= 0:
            status.status = "empty"
            status.details = "RAG index is present but empty"
        elif not status.embedding_model_loaded:
            status.status = "warming"
            status.details = "RAG embedding model is still loading"
        else:
            age_seconds = now - status.last_updated_ts if status.last_updated_ts > 0 else 0.0
            if age_seconds > 7 * 24 * 3600:
                status.status = "stale"
                status.details = f"RAG index last updated {_format_age(age_seconds)}"
                logger.warning(
                    "rag_index_stale",
                    doc_count=doc_count,
                    last_updated_ts=status.last_updated_ts,
                    age_seconds=round(age_seconds, 2),
                )
            else:
                status.status = "ready"
                status.details = f"RAG index refreshed {_format_age(age_seconds)}"

        logger.info(
            "rag_readiness_checked",
            status=status.status,
            doc_count=doc_count,
            last_updated_ts=status.last_updated_ts,
            embedding_model_loaded=status.embedding_model_loaded,
        )
    except Exception as exc:
        status.status = "empty"
        status.details = str(exc)
        logger.error("rag_readiness_check_failed", error=str(exc), exc_info=True)

    if health_monitor is not None:
        health_monitor.set_rag_status(status)
    return status


async def main():
    logger.info("boot_banner_top", char="=", repeat=60)
    logger.info("boot_title", title="DEXTER AI ASSISTANT — Booting Up")
    logger.info("boot_banner_bottom", char="=", repeat=60)

    watchdog = None
    pipeline = None
    proactive_task = None
    dashboard_task = None

    # 1. Load Configuration & Profile (Fast)
    start_time = time.perf_counter()
    runtime_config = get_config()
    user_profile = UserProfile(runtime_config)
    logger.info("configuration_loaded", bot_name=runtime_config.bot_name)
    for warning in config_validation_warnings(runtime_config):
        logger.warning("configuration_warning", detail=warning)
    logger.info("configuration_validated")
    resolve_hardware_config(runtime_config)

    try:
        # Safe mode: disable audio input/output for diagnostics or CI
        safe_mode = os.environ.get("DEXTER_SAFE_MODE", "0").strip() == "1"
        if safe_mode:
            logger.info("safe_mode_enabled", reason="DEXTER_SAFE_MODE=1")

        _automation_available = can_automate()
        if _automation_available:
            logger.info("Automation subsystem: available")
        else:
            logger.warning("Automation subsystem: unavailable — input_tools and vision_tools will be disabled")

        health_monitor = HealthMonitor(service_name="Dexter", automation_available=_automation_available)
        set_global_health_monitor(health_monitor)
        health_monitor.healthy("startup", "configuration loaded")
        logger.info("automation_capability_checked", supported=can_automate())

        event_bus = EventBus()
        watchdog = HardwareWatchdog(runtime_config.hardware.watchdog, event_bus, _WATCHDOG_STOP_EVENT)
        watchdog.start()

        global _ACTIVE_WATCHDOG
        _ACTIVE_WATCHDOG = watchdog

        # 2. Kick off slow background initializations via LazyLoaders
        logger.info("initializing_stage", stage="background_loaders")
        
        def _load_transcriber():
            from core.audio.transcriber import DexterTranscriber

            stt_model = runtime_config.stt.model or runtime_config.hardware.whisper_model
            t = DexterTranscriber(
                model_size=stt_model,
                beam_size=runtime_config.stt.beam_size,
                best_of=runtime_config.stt.best_of,
                temperature=runtime_config.stt.temperature,
                patience=runtime_config.stt.patience,
                log_prob_threshold=runtime_config.stt.log_prob_threshold,
                no_speech_threshold=runtime_config.stt.no_speech_threshold,
                condition_on_previous_text=runtime_config.stt.condition_on_previous_text,
                initial_prompt=runtime_config.stt.initial_prompt,
            )
            if not safe_mode:
                try:
                    t.warm_up()
                except Exception as e:
                    logger.warning("transcriber_warmup_failed", error=str(e))
            return t

        transcriber_loader = LazyLoader("Transcriber", _load_transcriber)

        def _load_vad():
            if safe_mode: return _DisabledVADListener()
            try:
                from core.audio.vad import VADListener
                return VADListener(
                    sample_rate=runtime_config.audio_settings.sample_rate,
                    chunk_size=runtime_config.audio_settings.chunk_size,
                )
            except Exception as e:
                logger.warning("vad_import_failed", error=str(e), exc_info=True)
                return _DisabledVADListener()

        vad_loader = LazyLoader("VAD", _load_vad)

        def _load_memory():
            from core.brain.memory import DexterMemory

            return DexterMemory(disable_rag_warming=runtime_config.runtime.disable_rag_warming, event_bus=event_bus)

        memory_loader = LazyLoader("Memory", _load_memory)

        # 3. Initialize fast synchronous components
        logger.info("initializing_stage", stage="fast_components")

        context_store = ContextStore()
        session_context = context_store.load()
        feedback_store = FeedbackStore()
        logger.info(
            "session_context_loaded",
            source="boot",
            has_project=bool(session_context.project),
            turn_summaries=len(session_context.recent_turn_summaries),
        )
        
        # TTS manager with cancellation support.
        if not safe_mode:
            from core.audio.speaker import TTSManager

            tts_manager = TTSManager(voice=runtime_config.models.tts_voice)
            health_monitor.healthy("tts", "speaker ready")
        else:
            class _DummyTTS:
                async def speak(self, text: str, interrupt: bool = True):
                    logger.info("tts_speak_skipped_safe_mode", text_preview=(text or "")[:80])
                async def play_chime(self):
                    logger.info("tts_chime_skipped_safe_mode")
                def stop(self):
                    pass
            tts_manager = _DummyTTS()
            health_monitor.degraded("tts", "safe_mode: audio disabled")

        # Initialize ASR Correction and kick off background vocabulary sync
        asr_engine = ASRCorrectionEngine()
        def _sync_vocab():
            try:
                logger.debug("vocabulary_sync_started")
                builder = VocabularyBuilder(asr_engine=asr_engine, config=runtime_config)
                builder.build_all()
                logger.debug("vocabulary_sync_completed")
            except Exception as e:
                logger.warning("vocabulary_sync_failed", error=str(e))
        threading.Thread(target=_sync_vocab, daemon=True, name="VocabSync").start()

        # Connect to LLM backends (Gemini → Groq → Ollama)
        from core.brain.llm_router import Brain
        from tools import document_tools

        document_tools.set_event_bus(event_bus)
        brain = Brain(event_bus=event_bus, asr_engine=asr_engine, session_context=session_context)
        health_monitor.healthy("brain", "llm router ready")

        provider_status, primary_provider = await brain.check_provider_status()
        logger.info(
            "startup_provider_status",
            Gemini=provider_status.get("Gemini", "UNKNOWN"),
            Groq=provider_status.get("Groq", "UNKNOWN"),
            Ollama=provider_status.get("Ollama", "UNKNOWN"),
        )
        logger.info("startup_primary_provider", provider=primary_provider)

        health_monitor.evaluate()
        health_monitor.start_evaluation_loop(interval_seconds=300.0)

        proactive = None
        if runtime_config.proactive.enabled and not runtime_config.runtime.disable_proactive_mode:
            from core.proactive import ProactiveAssistant

            proactive = ProactiveAssistant(
                event_bus=event_bus,
                check_interval_seconds=runtime_config.proactive.reminder_check_seconds,
                system_status_interval_seconds=runtime_config.proactive.system_status_interval_seconds,
            )
            health_monitor.healthy("proactive", "background assistant ready")
        elif runtime_config.runtime.disable_proactive_mode:
            logger.info("proactive_mode_disabled", reason="low_power_mode")

        # 4. Greet the user IMMEDIATELY while loaders finish in background
        greeting_time = time.perf_counter() - start_time
        logger.info("startup_profiler", stage="greeting", elapsed_sec=round(greeting_time, 2))
        
        if not safe_mode:
            greeting = _build_greeting(user_profile)
            await tts_manager.speak(greeting)

        # 5. Wait for heavy background loaders to complete before starting pipeline
        logger.info("initializing_stage", stage="waiting_for_loaders")
        
        transcriber = await asyncio.to_thread(transcriber_loader.get)
        health_monitor.healthy("stt", "transcriber ready")
        
        ear = await asyncio.to_thread(vad_loader.get)
        if isinstance(ear, _DisabledVADListener):
            health_monitor.degraded("vad", "disabled")
        else:
            health_monitor.healthy("vad", "listening component ready")
            
        memory_vault = await asyncio.to_thread(memory_loader.get)
        health_monitor.healthy("memory", "long-term memory ready")

        health_monitor.attach_runtime_context(runtime_config, memory_vault, event_bus)

        gpu_status = check_gpu_readiness(runtime_config, health_monitor)
        rag_status = check_rag_readiness(memory_vault, runtime_config, health_monitor)

        ready_time = time.perf_counter() - start_time
        logger.info("startup_profiler", stage="fully_ready", elapsed_sec=round(ready_time, 2))
        
        logger.info("boot_spacer")
        logger.info("boot_banner_top", char="═", repeat=60)
        activation_mode = (runtime_config.activation.mode or "wake_word").strip().lower()
        wake_words = list(runtime_config.activation.wake_words or runtime_config.wake_words)
        logger.info("assistant_ready", activation_mode=activation_mode, wake_words=wake_words)
        logger.info("boot_banner_bottom", char="═", repeat=60)

        rag_age_label = "warming"
        if rag_status.status in {"ready", "stale"} and rag_status.last_updated_ts > 0:
            rag_age_label = _format_age(time.time() - rag_status.last_updated_ts)
        elif rag_status.status == "empty":
            rag_age_label = "empty"

        gpu_label = "no GPU"
        if gpu_status.status == "ready" and gpu_status.device_name:
            vram_label = _format_gb(gpu_status.total_vram_gb)
            gpu_label = f"{gpu_status.device_name} ({vram_label})"
        elif gpu_status.expected_cuda and not gpu_status.cuda_available:
            gpu_label = "expected but unavailable"

        provider_summary = _provider_marks(provider_status)
        logger.info(
            "dexter_ready_summary",
            summary=(
                f"DEXTER ready — GPU: {gpu_label}, "
                f"RAG: {rag_status.doc_count:,} docs ({rag_age_label}), "
                f"Providers: {provider_summary}"
            ),
        )

        from core.pipeline import AsyncPipeline
        from server.ws_bridge import DashboardServer

        pipeline = AsyncPipeline(
            config=runtime_config,
            transcriber=transcriber,
            vad_listener=ear,
            tts_manager=tts_manager,
            memory_vault=memory_vault,
            brain=brain,
            event_bus=event_bus,
            health_monitor=health_monitor,
            asr_engine=asr_engine,
            context_store=context_store,
            session_context=session_context,
            feedback_store=feedback_store,
            watchdog_stop_event=_WATCHDOG_STOP_EVENT,
        )
        global _ACTIVE_PIPELINE
        _ACTIVE_PIPELINE = pipeline

        dashboard = DashboardServer(
            event_bus=event_bus,
            health_monitor=health_monitor,
            websocket_port=int(getattr(runtime_config.server, "websocket_port", 8765)),
            static_dir=os.path.join(os.path.dirname(__file__), getattr(runtime_config.server, "static_dir", "static/")),
            current_project_getter=lambda: pipeline.session_context.project,
            current_provider_getter=lambda: getattr(brain, "last_provider", None),
            current_state_getter=lambda: pipeline.state.name,
        )
        global _ACTIVE_DASHBOARD
        _ACTIVE_DASHBOARD = dashboard
        logger.info(f"Dashboard available at {dashboard.url}")
        dashboard_task = asyncio.create_task(dashboard.serve(), name="dexter-dashboard-server")

        proactive_task = None
        if proactive is not None:
            proactive_task = asyncio.create_task(proactive.run())

        from tools import registry as tool_registry

        async def _start_mcp_background(config, registry_module):
            """Start MCP server after a short delay so voice pipeline is responsive first."""
            delay = float(getattr(config.mcp, "start_delay_seconds", 5.0))
            await asyncio.sleep(delay)
            success = await registry_module.initialize_mcp(config)
            if success:
                logger.info("mcp_ready_for_use")
            else:
                logger.warning(
                    "mcp_unavailable",
                    message="File and document tools will not be available this session.",
                )

        mcp_task = asyncio.create_task(
            _start_mcp_background(runtime_config, tool_registry)
        )

        try:
            await pipeline.run()
        finally:
            _cleanup_runtime_sync()
            mcp_task.cancel()
            if dashboard_task is not None:
                dashboard_task.cancel()
                with suppress(asyncio.CancelledError):
                    await dashboard_task
            try:
                await tool_registry.shutdown_mcp()
            except Exception as e:
                logger.warning("mcp_shutdown_error", error=str(e))
            if proactive_task is not None:
                proactive_task.cancel()
            try:
                health_monitor.stop_evaluation_loop()
            except Exception:
                pass

    except Exception as e:
        logger.error("critical_system_error", error=str(e), exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown_requested", reason="keyboard_interrupt")
