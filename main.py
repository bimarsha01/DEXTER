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

logger = get_logger("main")

# ─── Add Nvidia DLLs for faster-whisper CUDA support on Windows ──────────────
try:
    for sp in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            bin_path = os.path.join(sp, "nvidia", lib, "bin")
            if os.path.exists(bin_path):
                os.add_dll_directory(bin_path)
                # Force-load critical DLLs
                try:
                    if "cublas" in lib:
                        ctypes.CDLL(os.path.join(bin_path, "cublas64_12.dll"))
                    elif "cudnn" in lib:
                        ctypes.CDLL(os.path.join(bin_path, "cudnn64_9.dll"))
                except OSError:
                    logger.debug("nvidia_dll_preload_skipped", lib=lib)
except Exception as e:
    logger.warning("nvidia_dll_setup_failed", error=str(e))

# ─── Module Imports ──────────────────────────────────────────────────────────
from core.audio.vad import VADListener
from core.audio.transcriber import DexterTranscriber
from core.audio.speaker import TTSManager
from core.brain.llm_router import Brain
from core.brain.memory import DexterMemory
from core.event_bus import EventBus
from core.pipeline import AsyncPipeline


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
        logger.info("initializing_stage", stage="audio_pipeline")

        # Load Whisper on GPU for speech-to-text
        transcriber = DexterTranscriber(
            model_size=runtime_config.models.whisper_model,
            beam_size=runtime_config.speed.whisper_beam_size,
        )

        # Load Silero VAD for voice activity detection
        ear = VADListener()

        # TTS manager with cancellation support
        tts_manager = TTSManager(voice=runtime_config.models.tts_voice)

        logger.info("initializing_stage", stage="memory_system")
        # Load ChromaDB long-term memory
        memory_vault = DexterMemory()

        logger.info("initializing_stage", stage="neural_network")
        # Connect to LLM backends (Gemini → Groq → Ollama)
        event_bus = EventBus()
        brain = Brain(event_bus=event_bus)

        # 3. Greet the user
        await tts_manager.speak(
            "All systems online, sir. Dexter is ready for your command."
        )
        logger.info("boot_spacer")
        logger.info("boot_banner_top", char="═", repeat=60)
        logger.info("assistant_ready", wake_words=list(runtime_config.wake_words))
        logger.info("boot_banner_bottom", char="═", repeat=60)

        pipeline = AsyncPipeline(
            config=runtime_config,
            transcriber=transcriber,
            vad_listener=ear,
            tts_manager=tts_manager,
            memory_vault=memory_vault,
            brain=brain,
            event_bus=event_bus,
        )
        await pipeline.run()

    except Exception as e:
        logger.error("critical_system_error", error=str(e), exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutdown_requested", reason="keyboard_interrupt")
