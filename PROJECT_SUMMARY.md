# Dexter Project Summary

## Overview
Dexter is a modular, voice-first personal AI assistant for Windows. It listens for speech, transcribes locally, routes commands through a multi-LLM fallback chain, executes OS-level tools, and replies with neural TTS. The system is designed to run mostly locally, with cloud LLMs as primary/fallback and an optional local offline model.

## Current Architecture
### Audio Ingest (Listening and Speech-to-Text)
- Voice Activity Detection: Silero VAD via PyTorch Hub in `core/audio/vad.py`.
- Audio capture: `sounddevice` with a queue-driven stream, saving to `temp_mic.wav`.
- Speech-to-text: `faster-whisper` in `core/audio/transcriber.py`.
  - Prefers GPU CUDA (float16); falls back to CPU int8 on failure.
  - `beam_size` configurable via `config.yaml` for speed.

### Wake Word Handling
- Wake words are defined in `config.yaml` under `wake_words` (default: "hey dexter", "dexter").
- `main.py` strips wake words from the recognized transcript before sending to LLMs.

### LLM Router (Brain)
- Multi-backend LLM router in `core/brain/llm_router.py`.
- Fallback chain: Gemini (primary) -> Groq (fallback) -> Ollama (offline).
- Uses the new `google-genai` SDK for Gemini.
- Tool schema generation for Groq is auto-built using `inspect.signature`.
- Shared conversation history is capped at 20 messages.
- Basic provider health tracking and cooldown for rate limit errors.

### Tooling (Hands)
- Central registry in `tools/registry.py` with a wrapped tool list and metrics.
- Current tool modules:
  - `tools/pc_controls.py`: app launching, closing, lock, volume control via pycaw.
  - `tools/system_tools.py`: time, weather, system info, clipboard, screenshots, power actions, health report.
  - `tools/file_tools.py`: notes create/read/list in a dedicated notes folder.
  - `tools/web_browser.py`: Google/YouTube search, open URL.
  - `tools/input_tools.py`: typing, shortcuts, enter key, minimize windows.
  - `tools/vision_tools.py`: referenced in registry (file capture and screen capture), exists in registry but not shown in this summary.

### Memory
- Long-term memory via ChromaDB in `core/brain/memory.py`.
- Stores and retrieves recent context for prompt augmentation.

### Text-to-Speech (Speaking)
- TTS via `edge-tts` in `core/audio/speaker.py`.
- Audio playback uses PowerShell MediaPlayer, with `ffplay` and `start` fallbacks.
- Supports canceling ongoing speech.

### Configuration and Logging
- `config.yaml` drives model selection, wake words, audio settings, speed, and security.
- Logger in `utils/logger.py` uses UTF-8 output and debug-level logs.
- Metrics collection is referenced via `utils/metrics` (not detailed in this summary).

## Key Files
- `main.py`: bootstraps components, wake word handling, main loop.
- `config.yaml`: core configuration.
- `core/brain/llm_router.py`: LLM routing, tool calling, fallback logic.
- `core/audio/transcriber.py`: faster-whisper setup.
- `core/audio/vad.py`: Silero VAD listener.
- `core/audio/speaker.py`: TTS playback and cancellation.
- `tools/registry.py`: tool registration and execution.

## Dependencies (from requirements.txt)
- LLM backends: `google-genai`, `groq`, `ollama` (optional offline).
- Audio: `sounddevice`, `soundfile`, `torch`, `torchaudio`, `numpy`.
- STT: `faster-whisper`.
- TTS: `edge-tts`.
- Memory: `chromadb`, `sentence-transformers`.
- Automation: `pyautogui`, `pillow`.
- Volume control: `pycaw`, `comtypes`.

## Operational Flow
1. Initialize config, GPU DLL hints, and components.
2. Listen for speech, capture audio on VAD.
3. Transcribe with faster-whisper.
4. Check wake word and strip it.
5. Retrieve memory context.
6. Route command to LLM with fallback.
7. Execute tool if requested.
8. Speak response and store memory.

## Known Configuration Details
- Primary LLM: `gemini-2.0-flash`.
- Fallback LLM: `llama-3.3-70b-versatile` (Groq).
- Local LLM: `qwen3-coder:480b-cloud` (Ollama, optional).
- Whisper model: `small.en`.
- TTS voice: `en-GB-RyanNeural`.

## Current Notes
- `clap_wake` settings exist in `config.yaml`, but no clap detection is implemented in `core/audio/vad.py` yet.
- `vision_tools` are referenced in the registry and should be verified if present and functional.
- The LLM router references `core/brain/intent_router.py` and `utils/metrics.py`, which should exist and be verified for completeness.
