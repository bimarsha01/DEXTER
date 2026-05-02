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
from utils.logger import logger
from utils.config import get_config, config_validation_warnings

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
                    pass
except Exception as e:
    logger.warning(f"Could not load Nvidia DLLs: {e}")

# ─── Module Imports ──────────────────────────────────────────────────────────
from core.audio.vad import VADListener
from core.audio.transcriber import DexterTranscriber
from core.audio.speaker import TTSManager
from core.brain.llm_router import Brain
from core.brain.memory import DexterMemory
from core.event_bus import EventBus
from core.pipeline import AsyncPipeline


async def main():
    logger.info("=" * 60)
    logger.info("  DEXTER AI ASSISTANT — Booting Up...")
    logger.info("=" * 60)

    # 1. Load Configuration
    runtime_config = get_config()
    logger.info(f"Configuration loaded for: {runtime_config.bot_name}")
    for warning in config_validation_warnings(runtime_config):
        logger.warning(warning)
    logger.info("Typed runtime configuration validated successfully.")

    try:
        # 2. Boot up all components
        logger.info("─── Initializing Audio Pipeline ───")

        # Load Whisper on GPU for speech-to-text
        transcriber = DexterTranscriber(
            model_size=runtime_config.models.whisper_model,
            beam_size=runtime_config.speed.whisper_beam_size,
        )

        # Load Silero VAD for voice activity detection
        ear = VADListener()

        # TTS manager with cancellation support
        tts_manager = TTSManager(voice=runtime_config.models.tts_voice)

        logger.info("─── Initializing Memory System ───")
        # Load ChromaDB long-term memory
        memory_vault = DexterMemory()

        logger.info("─── Initializing Neural Network ───")
        # Connect to LLM backends (Gemini → Groq → Ollama)
        event_bus = EventBus()
        brain = Brain(event_bus=event_bus)

        # 3. Greet the user
        await tts_manager.speak(
            "All systems online, sir. Dexter is ready for your command."
        )
        logger.info("")
        logger.info("═" * 60)
        logger.info("  DEXTER IS READY — Listening for wake word...")
        logger.info(f"  Wake words: {runtime_config.wake_words}")
        logger.info("═" * 60)

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
        logger.error(f"Critical System Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dexter shutting down. Goodbye, sir.")
