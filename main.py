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
import re
import site
import time
import ctypes
from utils.logger import logger
from utils.config import load_config
from utils.metrics import metrics

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


def strip_wake_words(text: str, wake_words: list) -> str:
    """
    Remove wake words from the transcribed text so the LLM gets a clean command.
    e.g., 'Hey Dexter, what time is it?' → 'what time is it?'
    """
    clean = text
    for ww in sorted(wake_words, key=len, reverse=True):  # longest first
        clean = re.sub(re.escape(ww), "", clean, flags=re.IGNORECASE)
    # Remove leftover leading punctuation/whitespace
    clean = re.sub(r"^[\s,.\-!?]+", "", clean).strip()
    return clean if clean else text  # fallback to original if nothing left


def validate_config(config: dict):
    """Print startup diagnostics about API key configuration."""
    gemini_key = config.get("api_keys", {}).get("gemini", "")
    groq_key = config.get("api_keys", {}).get("groq", "")

    if not gemini_key or "YOUR" in gemini_key.upper():
        logger.warning("⚠ Gemini API key is not configured in config.yaml")
    else:
        logger.info(f"Gemini API key: ...{gemini_key[-8:]}")

    if not groq_key or "YOUR" in groq_key.upper():
        logger.warning("⚠ Groq API key is not configured in config.yaml")
    else:
        logger.info(f"Groq API key: ...{groq_key[-8:]}")


async def main():
    logger.info("=" * 60)
    logger.info("  DEXTER AI ASSISTANT — Booting Up...")
    logger.info("=" * 60)

    # 1. Load Configuration
    config = load_config()
    logger.info(f"Configuration loaded for: {config['bot_name']}")
    validate_config(config)

    try:
        # 2. Boot up all components
        logger.info("─── Initializing Audio Pipeline ───")

        # Load Whisper on GPU for speech-to-text
        transcriber = DexterTranscriber(
            model_size=config["models"]["whisper_model"],
            beam_size=config.get("speed", {}).get("whisper_beam_size", 1),
        )

        # Load Silero VAD for voice activity detection
        ear = VADListener()

        # TTS manager with cancellation support
        tts_manager = TTSManager(voice=config["models"]["tts_voice"])

        logger.info("─── Initializing Memory System ───")
        # Load ChromaDB long-term memory
        memory_vault = DexterMemory()

        logger.info("─── Initializing Neural Network ───")
        # Connect to LLM backends (Gemini → Groq → Ollama)
        brain = Brain()

        # 3. Greet the user
        await tts_manager.speak(
            "All systems online, sir. Dexter is ready for your command."
        )
        logger.info("")
        logger.info("═" * 60)
        logger.info("  DEXTER IS READY — Listening for wake word...")
        logger.info(f"  Wake words: {config['wake_words']}")
        logger.info("═" * 60)

        # ─── Main Loop ───────────────────────────────────────────────────
        while True:
            # Listen for voice (runs in thread to not block async loop)
            vad_start = time.perf_counter()
            audio_path = await asyncio.to_thread(ear.listen, on_speech_start=tts_manager.stop)
            metrics.record_latency("vad_ms", (time.perf_counter() - vad_start) * 1000)

            if audio_path:
                # 4. Transcribe audio → text using Whisper
                stt_start = time.perf_counter()

                def _on_partial(text: str) -> None:
                    if text:
                        logger.debug(f"Partial STT: '{text[:80]}'")

                identified_text = transcriber.transcribe(audio_path, on_partial=_on_partial)
                metrics.record_latency("stt_ms", (time.perf_counter() - stt_start) * 1000)

                if not identified_text:
                    continue  # Discard ambient noise

                logger.debug(f"Whisper heard: '{identified_text}'")

                # 5. Check for wake word
                text_lower = identified_text.lower()
                wake_word_triggered = any(
                    ww in text_lower for ww in config["wake_words"]
                )

                if wake_word_triggered:
                    # Strip wake word to get clean command
                    clean_command = strip_wake_words(
                        identified_text, config["wake_words"]
                    )
                    logger.info(f"Command: {clean_command}")

                    # 6. Recall relevant memories (costs 0 API tokens)
                    related_memory = memory_vault.recall_context(clean_command)

                    # 7. Route to LLM (Gemini → Groq → Ollama)
                    response_text = await brain.process_command(
                        clean_command, long_term_memory=related_memory
                    )

                    # 8. Speak the response
                    logger.info(f"Dexter: {response_text}")
                    tts_task = asyncio.create_task(tts_manager.speak(response_text))

                    def _handle_tts_result(task: asyncio.Task) -> None:
                        try:
                            task.result()
                        except Exception as e:
                            logger.error(f"TTS task failed: {e}")

                    tts_task.add_done_callback(_handle_tts_result)

                    # 9. Save to long-term memory
                    memory_vault.remember(
                        f"User: {clean_command} | Dexter: {response_text}"
                    )

            # Brief sleep to keep CPU usage clean
            await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"Critical System Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dexter shutting down. Goodbye, sir.")
