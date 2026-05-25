import sys
import threading
from types import SimpleNamespace
import unittest.mock as mock

import pytest

import psutil
import chromadb

from core.hardware_watchdog import HardwareWatchdog
import core.hardware_watchdog as hwmod

# Replace watchdog logger with a simple mock implementing expected methods
hwmod.logger = mock.Mock(critical=mock.Mock(), info=mock.Mock(), warning=mock.Mock(), error=mock.Mock(), debug=mock.Mock())


class DummyConfig(SimpleNamespace):
    pass


def make_watchdog(config=None, event_bus=None):
    cfg = config or DummyConfig()
    stop_event = threading.Event()
    return HardwareWatchdog(cfg, event_bus, stop_event), stop_event


@pytest.fixture
def watchdog(request):
    wd, stop_event = make_watchdog()

    def _cleanup() -> None:
        try:
            wd.stop()
        finally:
            wd.join(timeout=3.0)

    request.addfinalizer(_cleanup)
    return wd, stop_event


def test_normal_readings_no_stop(monkeypatch, watchdog):
    wd, stop_event = watchdog
    # Mock sensor readers to safe values
    wd._read_cpu_temp_c = lambda: 45.0
    wd._read_gpu_temp_c = lambda: 40.0
    wd._read_vram_info = lambda: (10.0, 1_000_000)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=30.0))
    monkeypatch.setattr(psutil, "disk_usage", lambda root: SimpleNamespace(free=500, total=1000))

    for _ in range(5):
        status = wd._check_all()
        assert status == "ok"
        assert not stop_event.is_set()


def test_cpu_warn_logs_but_no_stop(caplog, monkeypatch, watchdog):
    wd, stop_event = watchdog
    wd._read_cpu_temp_c = lambda: 87.0
    wd._read_gpu_temp_c = lambda: None
    wd._read_vram_info = lambda: None
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=30.0))
    monkeypatch.setattr(psutil, "disk_usage", lambda root: SimpleNamespace(free=500, total=1000))

    caplog.clear()
    status = wd._check_all()
    assert status in {"warn", "ok"}
    assert not stop_event.is_set()
    # Should have a warning about CPU temp (logger is mocked in tests)
    assert hwmod.logger.warning.called
    assert any("CPU temp high" in str(args[0]) for args, _ in getattr(hwmod.logger.warning, 'call_args_list', []))


def test_cpu_critical_sustained_triggers_stop(monkeypatch, watchdog):
    wd, stop_event = watchdog
    # Force CPU temp to critical level
    wd._read_cpu_temp_c = lambda: 97.0
    wd._read_gpu_temp_c = lambda: None
    wd._read_vram_info = lambda: None
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=30.0))
    monkeypatch.setattr(psutil, "disk_usage", lambda root: SimpleNamespace(free=500, total=1000))

    # Control time progression inside the watchdog module
    current = {"t": 1_000_000.0}
    def fake_time():
        return current["t"]

    monkeypatch.setattr("core.hardware_watchdog.time.time", fake_time)

    # First call: starts timer, should be warn (not critical yet)
    first = wd._check_cpu_temp()
    assert first == "warn"

    # Advance 25s -> still warn
    current["t"] += 25.0
    second = wd._check_cpu_temp()
    assert second == "warn"

    # Advance beyond critical duration (31s)
    current["t"] += 6.0
    # Spy on _trigger_emergency_stop so we can assert it was invoked
    spy = mock.Mock(wraps=wd._trigger_emergency_stop)
    monkeypatch.setattr(wd, "_trigger_emergency_stop", spy)

    # Now call _check_all, which will call _trigger_emergency_stop when critical
    overall = wd._check_all()
    assert overall == "critical"
    assert spy.called
    # Reason may be generic; accept either contains 'CPU' or 'critical'
    args = spy.call_args[0] if spy.call_args else ("",)
    reason = args[0] if args else ""
    assert ("CPU" in reason) or ("critical" in reason)
    assert stop_event.is_set()


def test_gpu_critical_sustained_triggers_stop(monkeypatch, watchdog):
    wd, stop_event = watchdog
    wd._read_cpu_temp_c = lambda: None
    wd._read_gpu_temp_c = lambda: 94.0
    wd._read_vram_info = lambda: None
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=30.0))
    monkeypatch.setattr(psutil, "disk_usage", lambda root: SimpleNamespace(free=500, total=1000))

    current = {"t": 2_000_000.0}
    monkeypatch.setattr("core.hardware_watchdog.time.time", lambda: current["t"])

    first = wd._check_gpu_temp()
    assert first == "warn"
    current["t"] += 31.0
    spy = mock.Mock(wraps=wd._trigger_emergency_stop)
    monkeypatch.setattr(wd, "_trigger_emergency_stop", spy)
    overall = wd._check_all()
    assert overall == "critical"
    assert spy.called
    assert stop_event.is_set()


def test_ram_critical_immediate_sets_stop(monkeypatch, watchdog):
    wd, stop_event = watchdog
    wd._read_cpu_temp_c = lambda: None
    wd._read_gpu_temp_c = lambda: None
    wd._read_vram_info = lambda: None
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=97.0))
    monkeypatch.setattr(psutil, "disk_usage", lambda root: SimpleNamespace(free=500, total=1000))

    status = wd._check_ram()
    assert status == "critical"
    wd._check_all()
    assert stop_event.is_set()


def test_vram_oom_recovery_cpu_retry_no_emergency(monkeypatch):
    # Create a PersonalRAGIndex-like object with minimal init by mocking chromadb
    mock_client = SimpleNamespace(get_or_create_collection=lambda name, metadata: SimpleNamespace(), get_collection=lambda name: SimpleNamespace())
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: mock_client)

    # Import here to avoid module-level side effects earlier
    from core.brain.rag import PersonalRAGIndex

    # Create fake torch module with cuda.empty_cache spy
    fake_cuda = SimpleNamespace(empty_cache=mock.Mock())
    fake_torch = SimpleNamespace(cuda=fake_cuda, cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Event bus spy
    eb = SimpleNamespace(emit=mock.Mock())

    # Instantiate with chromadb-default embedding to avoid model loading
    pri = PersonalRAGIndex(persist_directory="/tmp", embedding_model="chromadb-default", event_bus=eb)

    # GPU operation raises OOM-like exception, CPU operation returns expected value
    def gpu_op():
        raise Exception("CUDA out of memory")

    def cpu_op():
        return [ [0.1, 0.2] ]

    result = pri._retry_cpu_after_cuda_oom("test_reason", gpu_op, cpu_op)
    # Ensure empty_cache was called and CPU result returned
    assert fake_cuda.empty_cache.called
    assert result == cpu_op()
    # No emergency stop event emitted
    assert not eb.emit.called


def test_sensors_unavailable_graceful(monkeypatch, watchdog):
    wd, stop_event = watchdog
    # sensors_temperatures raises
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda fahrenheit=False: (_ for _ in ()).throw(Exception("no sensor")), raising=False)
    res = wd._check_cpu_temp()
    assert res == "ok"
    assert not stop_event.is_set()


def test_resume_after_stop(caplog, monkeypatch, watchdog):
    wd, stop_event = watchdog
    # Trigger stop via RAM
    wd._read_cpu_temp_c = lambda: None
    wd._read_gpu_temp_c = lambda: None
    wd._read_vram_info = lambda: None
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=97.0))
    monkeypatch.setattr(psutil, "disk_usage", lambda root: SimpleNamespace(free=500, total=1000))

    wd._check_all()
    assert stop_event.is_set()

    # Now simulate safe readings and advance time so resume happens
    wd._check_all = lambda: "ok"
    # control time so safe_since triggers 60s
    current = {"t": 1_234_000.0}
    monkeypatch.setattr("core.hardware_watchdog.time.time", lambda: current["t"])
    # Call wait_for_safe_resume which should clear event when time advances
    stop_event.set()
    # Advance time beyond 60s
    current["t"] += 61.0
    # Patch shutdown wait to no-op to avoid delays
    wd._shutdown_event.wait = lambda timeout=None: None
    caplog.clear()
    wd._wait_for_safe_resume()
    assert not stop_event.is_set()
    assert any("resuming" in r.getMessage().lower() for r in caplog.records)


def test_watchdog_self_stop_exits_loop_quickly(monkeypatch, watchdog):
    wd, stop_event = watchdog
    # make poll interval small by patching method (avoid min clamp in implementation)
    monkeypatch.setattr(wd, "_poll_interval", lambda: 0.05)
    wd._check_all = lambda: "ok"
    wd.start()
    # Stop the watchdog thread and ensure it exits promptly
    wd.stop()
    wd.join(timeout=1.0)
    # Thread should have exited
    assert wd._thread is None or not wd._thread.is_alive()
    # Stopping the watchdog itself should not set the pipeline stop_event
    assert not stop_event.is_set()
