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
import os
import site
import ctypes
from utils.logger import get_logger
from utils.config import get_config, config_validation_warnings
from core.health import HealthMonitor, set_global_health_monitor

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


async def main():
    logger.info("boot_banner_top", char="=", repeat=60)
    logger.info("boot_title", title="DEXTER AI ASSISTANT — Booting Up")
    logger.info("boot_banner_bottom", char="=", repeat=60)

    # 1. Load Configuration
    runtime_config = get_config()
    logger.info("configuration_loaded", bot_name=runtime_config.bot_name)
    for warning in config_validation_warnings(runtime_config):
        logger.warning("configuration_warning", detail=warning)
    logger.info("configuration_validated")

    try:
        # 2. Boot up all components
        # Safe mode: disable audio input/output for diagnostics or CI
        safe_mode = os.environ.get("DEXTER_SAFE_MODE", "0").strip() == "1"
        if safe_mode:
            logger.info("safe_mode_enabled", reason="DEXTER_SAFE_MODE=1")
        logger.info("initializing_stage", stage="audio_pipeline")
        health_monitor = HealthMonitor(service_name="Dexter")
        set_global_health_monitor(health_monitor)
        health_monitor.healthy("startup", "configuration loaded")

        # Load Whisper on GPU for speech-to-text
        transcriber = DexterTranscriber(
            model_size=runtime_config.models.whisper_model,
            beam_size=runtime_config.stt.beam_size,
            best_of=runtime_config.stt.best_of,
            temperature=runtime_config.stt.temperature,
            patience=runtime_config.stt.patience,
            log_prob_threshold=runtime_config.stt.log_prob_threshold,
            no_speech_threshold=runtime_config.stt.no_speech_threshold,
            condition_on_previous_text=runtime_config.stt.condition_on_previous_text,
            initial_prompt=runtime_config.stt.initial_prompt,
        )

        # Load Silero VAD for voice activity detection.
        # If torch/VAD is unavailable, keep the assistant alive with listening disabled.
        try:
            if not safe_mode:
                from core.audio.vad import VADListener

                ear = VADListener(
                    sample_rate=runtime_config.audio_settings.sample_rate,
                    chunk_size=runtime_config.audio_settings.chunk_size,
                )
            else:
                ear = _DisabledVADListener()
            health_monitor.healthy("vad", "listening component ready")
        except Exception as e:
            logger.warning("vad_import_failed", error=str(e), exc_info=True)
            ear = _DisabledVADListener()
            health_monitor.degraded("vad", f"disabled: {e}")

        # TTS manager with cancellation support. In safe mode, use a dummy TTS
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

        logger.info("initializing_stage", stage="memory_system")
        # Load ChromaDB long-term memory
        memory_vault = DexterMemory(
            disable_rag_warming=runtime_config.runtime.disable_rag_warming
        )
        health_monitor.healthy("memory", "long-term memory ready")

        logger.info("initializing_stage", stage="neural_network")
        # Connect to LLM backends (Gemini → Groq → Ollama)
        event_bus = EventBus()
        brain = Brain(event_bus=event_bus)
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

        # 3. Greet the user (skip in safe mode to avoid audio)
        if not safe_mode:
            await tts_manager.speak(
                "All systems online sir. Dexter is ready for your command."
            )
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
