import threading
import time as real_time
from unittest import mock

import pytest


def test_vad_suppress_for_prevents_speech_callback_within_window(monkeypatch):
    # Import inside the test so module-level imports can be patched if needed.
    import core.audio.vad as vad

    class FakeProb:
        def __init__(self, v: float):
            self._v = v

        def item(self) -> float:
            return self._v

    class FakeTensor:
        def float(self):
            return self

        def flatten(self):
            return self

        def __len__(self):
            return 512

    # Speech probability: first processed chunk => speech, second processed chunk => silence.
    call_state = {"i": 0}

    def fake_model(_tensor_chunk, _sample_rate):
        i = call_state["i"]
        call_state["i"] += 1
        return FakeProb(0.9 if i == 0 else 0.0)

    fake_utils = (
        lambda *_a, **_kw: [],
        lambda *_a, **_kw: None,
        lambda *_a, **_kw: None,
        object,  # VADIterator placeholder
        lambda *_a, **_kw: None,
    )

    def fake_hub_load(*_a, **_kw):
        return fake_model, fake_utils

    def fake_from_numpy(_arr):
        return FakeTensor()

    class DummyInputStream:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(vad.torch.hub, "load", fake_hub_load)
    monkeypatch.setattr(vad.torch, "from_numpy", fake_from_numpy)
    monkeypatch.setattr(vad.sd, "InputStream", DummyInputStream)
    monkeypatch.setattr(vad.sf, "write", lambda *_a, **_kw: None)

    listener = vad.VADListener(sample_rate=16000, chunk_size=512)

    on_speech_start = mock.Mock()

    listener.suppress_for(2.0)
    suppress_start = real_time.time()

    results = {"out": None}

    def run_listen():
        results["out"] = listener.listen(
            output_file=None,
            silence_threshold=0.01,  # short; will be overridden to 0.8 after first speech
            on_speech_start=on_speech_start,
            on_clap=None,
        )

    thread = threading.Thread(target=run_listen, daemon=True)
    thread.start()

    # Wait for the listen loop to start and block on q.get().
    real_time.sleep(0.02)

    # Chunk 1: within suppression window => should not trigger callback.
    listener.q.put([0.0] * 512)
    real_time.sleep(0.02)
    assert on_speech_start.call_count == 0

    # Chunk 2: enqueue only after suppression window has elapsed.
    while real_time.time() - suppress_start < 2.05:
        real_time.sleep(0.01)
    listener.q.put([0.0] * 512)  # speech

    # Chunk 3: initial silence (sets silence_start_time).
    listener.q.put([0.0] * 512)  # silence

    # Chunk 4: later silence to exceed effective_threshold (0.8s).
    real_time.sleep(0.9)
    listener.q.put([0.0] * 512)  # silence

    thread.join(timeout=4.0)
    assert not thread.is_alive(), "listen() should exit after silence threshold"

    assert on_speech_start.call_count == 1

