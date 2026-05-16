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
import getpass
import os
import site
import ctypes
import time
from datetime import datetime
from utils.logger import get_logger
from utils.config import get_config, config_validation_warnings
from core.health import HealthMonitor, set_global_health_monitor
from utils.lazy_loader import LazyLoader
from utils.user_profile import UserProfile
from utils.asr_corrections import ASRCorrectionEngine
from utils.vocabulary_builder import VocabularyBuilder
import threading

logger = get_logger("main")

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

# ─── Module Imports ──────────────────────────────────────────────────────────
from core.audio.transcriber import DexterTranscriber
from core.audio.speaker import TTSManager
from core.brain.llm_router import Brain
from core.brain.memory import DexterMemory
from core.event_bus import EventBus
from core.pipeline import AsyncPipeline
from core.proactive import ProactiveAssistant


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


async def main():
    logger.info("boot_banner_top", char="=", repeat=60)
    logger.info("boot_title", title="DEXTER AI ASSISTANT — Booting Up")
    logger.info("boot_banner_bottom", char="=", repeat=60)

    # 1. Load Configuration & Profile (Fast)
    start_time = time.perf_counter()
    runtime_config = get_config()
    user_profile = UserProfile(runtime_config)
    logger.info("configuration_loaded", bot_name=runtime_config.bot_name)
    for warning in config_validation_warnings(runtime_config):
        logger.warning("configuration_warning", detail=warning)
    logger.info("configuration_validated")

    try:
        # Safe mode: disable audio input/output for diagnostics or CI
        safe_mode = os.environ.get("DEXTER_SAFE_MODE", "0").strip() == "1"
        if safe_mode:
            logger.info("safe_mode_enabled", reason="DEXTER_SAFE_MODE=1")

        health_monitor = HealthMonitor(service_name="Dexter")
        set_global_health_monitor(health_monitor)
        health_monitor.healthy("startup", "configuration loaded")

        # 2. Kick off slow background initializations via LazyLoaders
        logger.info("initializing_stage", stage="background_loaders")
        
        def _load_transcriber():
            stt_model = runtime_config.stt.model or runtime_config.models.whisper_model
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
            return DexterMemory(disable_rag_warming=runtime_config.runtime.disable_rag_warming)

        memory_loader = LazyLoader("Memory", _load_memory)

        # 3. Initialize fast synchronous components
        logger.info("initializing_stage", stage="fast_components")
        
        # TTS manager with cancellation support.
        if not safe_mode:
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
                builder = VocabularyBuilder()
                builder.sync_all()
                asr_engine.reload()
                logger.debug("vocabulary_sync_completed")
            except Exception as e:
                logger.warning("vocabulary_sync_failed", error=str(e))
        threading.Thread(target=_sync_vocab, daemon=True, name="VocabSync").start()

        # Connect to LLM backends (Gemini → Groq → Ollama)
        event_bus = EventBus()
        brain = Brain(event_bus=event_bus, asr_engine=asr_engine)
        health_monitor.healthy("brain", "llm router ready")

        provider_status, primary_provider = await brain.check_provider_status()
        logger.info(
            "startup_provider_status",
            Gemini=provider_status.get("Gemini", "UNKNOWN"),
            Groq=provider_status.get("Groq", "UNKNOWN"),
            Ollama=provider_status.get("Ollama", "UNKNOWN"),
        )
        logger.info("startup_primary_provider", provider=primary_provider)

        proactive = None
        if runtime_config.proactive.enabled and not runtime_config.runtime.disable_proactive_mode:
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

        ready_time = time.perf_counter() - start_time
        logger.info("startup_profiler", stage="fully_ready", elapsed_sec=round(ready_time, 2))
        
        logger.info("boot_spacer")
        logger.info("boot_banner_top", char="═", repeat=60)
        activation_mode = (runtime_config.activation.mode or "wake_word").strip().lower()
        wake_words = list(runtime_config.activation.wake_words or runtime_config.wake_words)
        logger.info("assistant_ready", activation_mode=activation_mode, wake_words=wake_words)
        logger.info("boot_banner_bottom", char="═", repeat=60)

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
        )
        proactive_task = None
        if proactive is not None:
            proactive_task = asyncio.create_task(proactive.run())

        try:
            await pipeline.run()
        finally:
            if proactive_task is not None:
                proactive_task.cancel()

    except Exception as e:
        logger.error("critical_system_error", error=str(e), exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown_requested", reason="keyboard_interrupt")
