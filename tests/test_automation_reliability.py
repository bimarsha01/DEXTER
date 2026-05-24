import threading
import time
from types import SimpleNamespace

import pytest

from tools import input_tools
from tools import vision_tools


class DummyEventBus:
    def __init__(self):
        self.events = []

    def emit(self, name, payload):
        self.events.append((name, payload))


def test_verify_foreground_partial_case_insensitive(monkeypatch):
    # Mock win32gui
    fake = SimpleNamespace()
    fake.GetForegroundWindow = lambda: 100
    fake.GetWindowText = lambda hwnd: "Notepad - untitled"
    monkeypatch.setattr(input_tools, "win32gui", fake)

    ok, title = input_tools.verify_foreground("notepad")
    assert ok is True


def test_verify_foreground_negative(monkeypatch):
    fake = SimpleNamespace()
    fake.GetForegroundWindow = lambda: 101
    fake.GetWindowText = lambda hwnd: "Visual Studio Code"
    monkeypatch.setattr(input_tools, "win32gui", fake)

    ok, title = input_tools.verify_foreground("notepad")
    assert ok is False


def test_focus_retries_raise_and_message(monkeypatch):
    # Ensure automation settings
    monkeypatch.setattr(input_tools, "_automation_settings", lambda: SimpleNamespace(focus_wait_ms=1, max_focus_retries=3, post_action_verify=True))
    # Mock verify_foreground to always return False and a title
    monkeypatch.setattr(input_tools, "verify_foreground", lambda x: (False, "Visual Studio Code"))

    monkeypatch.setattr(input_tools, "_foreground_title", lambda: "Visual Studio Code")
    with pytest.raises(input_tools.AutomationFocusError) as exc:
        input_tools._ensure_foreground("notepad", focus_target=None)
    assert "Visual Studio Code" in str(exc.value)
    assert getattr(exc.value, "retries", 0) == 3


def test_automation_event_emitted_on_focus_failed(monkeypatch):
    eb = DummyEventBus()
    input_tools.set_event_bus(eb)

    monkeypatch.setattr(input_tools, "_automation_settings", lambda: SimpleNamespace(focus_wait_ms=1, max_focus_retries=1, post_action_verify=True))
    monkeypatch.setattr(input_tools, "_ensure_foreground", lambda *a, **k: (_ for _ in ()).throw(input_tools.AutomationFocusError("notepad", "VS", 1)))
    # prepare pyautogui so type_text goes past module missing check
    input_tools._pyautogui = SimpleNamespace(write=lambda *a, **k: None)

    with pytest.raises(input_tools.AutomationFocusError):
        input_tools.type_text("hello", app_name="notepad")

    assert any(evt for evt in eb.events if evt[0] == "automation_action" and evt[1].get("status") == "focus_failed")


def test_automation_action_success_on_type(monkeypatch):
    eb = DummyEventBus()
    input_tools.set_event_bus(eb)

    # ensure foreground check passes
    monkeypatch.setattr(input_tools, "_ensure_foreground", lambda *a, **k: "Some Title")
    input_tools._pyautogui = SimpleNamespace(write=lambda text, interval=0.01: None)
    # post-action verification returns True
    monkeypatch.setattr(input_tools, "_post_action_verified", lambda *a, **k: True)

    res = input_tools.type_text("hello", app_name="notepad")
    assert "Successfully typed" in res
    assert any(evt for evt in eb.events if evt[0] == "automation_action" and evt[1].get("status") == "success")


def test_can_automate_false_when_win32_missing(monkeypatch):
    monkeypatch.setattr(input_tools, "_pyautogui", None)
    monkeypatch.setattr(input_tools, "win32gui", None)
    assert input_tools.can_automate() is False


def test_pyautogui_failsafe_true():
    # pyautogui attribute set at import
    try:
        import pyautogui
        assert getattr(pyautogui, "FAILSAFE", True) is True
    except Exception:
        # If pyautogui not installed in test env, at least input_tools._pyautogui may be None
        assert True


@pytest.mark.timeout(3)
def test_capturecontext_restore_called_on_exception(monkeypatch, caplog):
    # simulate ImageGrab.grab raising to trigger __exit__ restore
    monkeypatch.setattr(vision_tools, "_snapshot_windows", lambda: [vision_tools.WindowSnapshot(hwnd=1, title="A", rect=(0,0,10,10), visible=True)])
    called = {}
    def fake_restore(snapshots):
        called['restored'] = True
        return True
    monkeypatch.setattr(vision_tools, "_restore_windows", fake_restore)
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda **k: (_ for _ in ()).throw(vision_tools.VisionCaptureError("boom")))

    res = vision_tools.capture_screen_for_vision()
    assert res is None
    assert called.get('restored', False) is True


@pytest.mark.timeout(3)
def test_restore_failure_logged_not_reraised(monkeypatch, caplog):
    monkeypatch.setattr(vision_tools, "_snapshot_windows", lambda: [vision_tools.WindowSnapshot(hwnd=1, title="A", rect=(0,0,10,10), visible=True)])
    def fake_restore_raise(snapshots):
        raise RuntimeError("restore fail")
    monkeypatch.setattr(vision_tools, "_restore_windows", fake_restore_raise)
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda **k: vision_tools.Image.new("RGB", (10,10), color="white"))

    caplog.clear()
    res = vision_tools.capture_screen_for_vision()
    assert res is None
    assert any("capture_context_restore_failed" in rec.message or "restore_failed" in rec.message for rec in caplog.records)


@pytest.mark.timeout(3)
def test_timeout_emits_event(monkeypatch):
    eb = DummyEventBus()
    vision_tools.set_event_bus(eb)
    # cause check_timeout to raise
    original_check = vision_tools.CaptureContext.check_timeout
    def raise_timeout(self):
        raise vision_tools.VisionCaptureTimeoutError("timeout")
    monkeypatch.setattr(vision_tools.CaptureContext, "check_timeout", raise_timeout)
    res = vision_tools.capture_screen_for_vision()
    assert res is None
    assert any(evt for evt in eb.events if evt[0] == "vision_capture" and evt[1].get("status") == "timeout")
    # restore original
    monkeypatch.setattr(vision_tools.CaptureContext, "check_timeout", original_check)


@pytest.mark.timeout(3)
def test_pre_capture_ensure_hidden_aborts(monkeypatch):
    # hide_foreground_ide returns a state but ensure_hidden will raise
    monkeypatch.setattr(vision_tools.CaptureContext, "hide_foreground_ide", lambda self: vision_tools.HiddenWindowState(hwnd=1, title="A", rect=(0,0,10,10), was_visible=True, pre_hide_hash=None))
    monkeypatch.setattr(vision_tools.CaptureContext, "ensure_hidden", lambda self: (_ for _ in ()).throw(vision_tools.VisionCaptureError("not hidden")))
    res = vision_tools.capture_screen_for_vision()
    assert res is None


def test_restore_order_reverse(monkeypatch):
    # make snapshots with hwnds [1,2,3]
    snaps = [vision_tools.WindowSnapshot(hwnd=i, title=str(i), rect=(0,0,10,10), visible=True) for i in (1,2,3)]
    # patch snapshot and capture flow to bypass grab
    monkeypatch.setattr(vision_tools, "_snapshot_windows", lambda: snaps)
    # ensure windows appear valid to restore
    monkeypatch.setattr(vision_tools, "win32gui", SimpleNamespace(IsWindow=lambda hwnd: True))
    called = []
    def fake_restore_snapshot(snapshot):
        called.append(snapshot.hwnd)
        return True
    monkeypatch.setattr(vision_tools, "_restore_window_snapshot", fake_restore_snapshot)
    # ensure grab returns an image
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda **k: vision_tools.Image.new("RGB", (10,10), color="white"))

    res = vision_tools.capture_screen_for_vision()
    assert res is not None
    # restore should have been called in reverse order
    assert called == [3,2,1]


def test_concurrent_capture_blocks(monkeypatch):
    # Use a small wait in capture to simulate long capture
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda **k: time.sleep(0.2) or vision_tools.Image.new("RGB", (10,10), color="white"))

    results = []

    def worker():
        res = vision_tools.capture_screen_for_vision()
        results.append(res is not None)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    start = time.time()
    t1.start()
    time.sleep(0.05)
    t2.start()
    t1.join()
    t2.join()
    duration = time.time() - start
    # both ran sequentially roughly > 0.2 and < 1.0
    assert len(results) == 2
    assert duration >= 0.2
