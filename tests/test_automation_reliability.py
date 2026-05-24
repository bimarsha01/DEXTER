from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from core.event_bus import EventBus
from tools import input_tools, vision_tools


@dataclass
class _EventCollector:
    bus: EventBus
    queue: object

    def drain(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                events.append(self.queue.get_nowait())
            except Exception:
                break
        return events


@pytest.fixture
def event_collector():
    bus = EventBus(maxsize=0)
    queue = bus.subscribe(maxsize=0)
    collector = _EventCollector(bus=bus, queue=queue)
    input_tools.set_event_bus(bus)
    vision_tools.set_event_bus(bus)
    try:
        yield collector
    finally:
        input_tools.set_event_bus(None)
        vision_tools.set_event_bus(None)


@pytest.fixture
def fake_win32gui():
    class FakeWin32Gui:
        def __init__(self, title: str):
            self.title = title
            self.focused = []

        def GetForegroundWindow(self):
            return 101

        def GetWindowText(self, hwnd):
            return self.title if hwnd == 101 else ""

        def EnumWindows(self, callback, results):
            callback(101, results)

        def ShowWindow(self, hwnd, _cmd):
            self.focused.append(("show", hwnd))
            return True

        def SetForegroundWindow(self, hwnd):
            self.focused.append(("fg", hwnd))
            return True

    return FakeWin32Gui


@pytest.fixture
def fake_pyautogui():
    return SimpleNamespace(
        write=Mock(return_value=None),
        hotkey=Mock(return_value=None),
        press=Mock(return_value=None),
        position=Mock(return_value=SimpleNamespace(x=11, y=22)),
    )


def test_verify_foreground_returns_true_when_expected_window_title_matches(monkeypatch, fake_win32gui):
    monkeypatch.setattr(input_tools, "win32gui", fake_win32gui("Dexter - Notes"))

    matched, title = input_tools.verify_foreground("Dexter")

    assert matched is True
    assert title == "Dexter - Notes"


def test_ensure_foreground_retries_max_focus_retries_before_raising(monkeypatch, fake_win32gui):
    monkeypatch.setattr(input_tools, "win32gui", fake_win32gui("Other Window"))
    monkeypatch.setattr(input_tools, "_automation_settings", lambda: SimpleNamespace(focus_wait_ms=0, post_action_verify=False, max_focus_retries=3))

    verify_calls: list[str | None] = []

    def fake_verify(expected_title_fragment: str | None):
        verify_calls.append(expected_title_fragment)
        return False, "Other Window"

    monkeypatch.setattr(input_tools, "verify_foreground", fake_verify)
    monkeypatch.setattr(input_tools, "_foreground_title", lambda: "Other Window")

    with pytest.raises(input_tools.AutomationFocusError):
        input_tools._ensure_foreground("Dexter")

    assert len(verify_calls) == 4


def test_automation_focus_error_message_includes_actual_foreground_title(monkeypatch, fake_win32gui):
    monkeypatch.setattr(input_tools, "win32gui", fake_win32gui("Actual Editor Title"))
    monkeypatch.setattr(input_tools, "_automation_settings", lambda: SimpleNamespace(focus_wait_ms=0, post_action_verify=False, max_focus_retries=1))
    monkeypatch.setattr(input_tools, "verify_foreground", lambda _expected: (False, "Actual Editor Title"))
    monkeypatch.setattr(input_tools, "_foreground_title", lambda: "Actual Editor Title")

    with pytest.raises(input_tools.AutomationFocusError) as excinfo:
        input_tools._ensure_foreground("Dexter")

    assert "Actual Editor Title" in str(excinfo.value)


def test_capture_context_restores_window_state_on_success(monkeypatch):
    restore_calls: list[list[object]] = []
    snapshots = [SimpleNamespace(hwnd=11, title="Editor", rect=(0, 0, 10, 10), visible=True)]
    dummy_image = Image.new("RGB", (32, 32), color="white")

    monkeypatch.setattr(vision_tools, "_snapshot_windows", lambda: snapshots)
    monkeypatch.setattr(vision_tools, "_restore_windows", lambda items: restore_calls.append(list(items)) or True)
    monkeypatch.setattr(vision_tools.CaptureContext, "check_timeout", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "hide_foreground_ide", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "ensure_hidden", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "verify_post_capture", lambda self, _image: None)
    monkeypatch.setattr(vision_tools, "_get_foreground_window_bbox", lambda: ("Editor - VS Code", (0, 0, 10, 10)))
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda all_screens=True: dummy_image)
    monkeypatch.setattr(vision_tools, "_resize_image", lambda image, max_dimension: image)

    result = vision_tools.capture_screen_for_vision(max_dimension=128)

    assert result.capture_mode == "full_screen"
    assert result.image_bytes
    assert restore_calls == [snapshots]


def test_capture_context_restores_window_state_when_capture_raises_midway(monkeypatch):
    restore_calls: list[list[object]] = []
    snapshots = [SimpleNamespace(hwnd=12, title="Editor", rect=(0, 0, 10, 10), visible=True)]

    monkeypatch.setattr(vision_tools, "_snapshot_windows", lambda: snapshots)
    monkeypatch.setattr(vision_tools, "_restore_windows", lambda items: restore_calls.append(list(items)) or True)
    monkeypatch.setattr(vision_tools.CaptureContext, "check_timeout", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "hide_foreground_ide", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "ensure_hidden", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "verify_post_capture", lambda self, _image: None)
    monkeypatch.setattr(vision_tools, "_get_foreground_window_bbox", lambda: ("Editor - VS Code", (0, 0, 10, 10)))
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda all_screens=True: (_ for _ in ()).throw(RuntimeError("capture failed")))

    with pytest.raises(RuntimeError, match="capture failed"):
        vision_tools.capture_screen_for_vision(max_dimension=128)

    assert restore_calls == [snapshots]


def test_capture_timeout_aborts_and_restores_state_without_partial_result(monkeypatch):
    restore_calls: list[list[object]] = []
    snapshots = [SimpleNamespace(hwnd=13, title="Editor", rect=(0, 0, 10, 10), visible=True)]
    dummy_image = Image.new("RGB", (32, 32), color="white")
    timeout_calls = {"count": 0}

    def fake_check_timeout(self):
        timeout_calls["count"] += 1
        if timeout_calls["count"] >= 4:
            raise vision_tools.VisionCaptureTimeoutError("Vision capture exceeded timeout of 0.01s")

    monkeypatch.setattr(vision_tools, "_snapshot_windows", lambda: snapshots)
    monkeypatch.setattr(vision_tools, "_restore_windows", lambda items: restore_calls.append(list(items)) or True)
    monkeypatch.setattr(vision_tools.CaptureContext, "check_timeout", fake_check_timeout)
    monkeypatch.setattr(vision_tools.CaptureContext, "hide_foreground_ide", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "ensure_hidden", lambda self: None)
    monkeypatch.setattr(vision_tools.CaptureContext, "verify_post_capture", lambda self, _image: None)
    monkeypatch.setattr(vision_tools, "_get_foreground_window_bbox", lambda: ("Editor - VS Code", (0, 0, 10, 10)))
    monkeypatch.setattr(vision_tools.ImageGrab, "grab", lambda all_screens=True: dummy_image)
    monkeypatch.setattr(vision_tools, "_resize_image", lambda image, max_dimension: image)

    with pytest.raises(vision_tools.VisionCaptureTimeoutError):
        vision_tools.capture_screen_for_vision(max_dimension=128)

    assert restore_calls == [snapshots]
    assert timeout_calls["count"] >= 4


def test_automation_action_event_emitted_for_success_and_focus_failed(monkeypatch, event_collector, fake_win32gui, fake_pyautogui):
    monkeypatch.setattr(input_tools, "win32gui", fake_win32gui("Dexter - Notes"))
    monkeypatch.setattr(input_tools, "_pyautogui", fake_pyautogui)
    monkeypatch.setattr(input_tools, "_mouse_position", lambda: (11, 22))
    monkeypatch.setattr(input_tools, "_read_clipboard_text", lambda: "clipboard")
    monkeypatch.setattr(input_tools, "_automation_settings", lambda: SimpleNamespace(focus_wait_ms=0, post_action_verify=False, max_focus_retries=0))

    result = input_tools.type_text("hello", app_name="Dexter")
    assert "Successfully typed" in result

    monkeypatch.setattr(input_tools, "win32gui", fake_win32gui("Other Window"))
    monkeypatch.setattr(input_tools, "verify_foreground", lambda _expected: (False, "Other Window"))
    monkeypatch.setattr(input_tools, "_foreground_title", lambda: "Other Window")

    with pytest.raises(input_tools.AutomationFocusError):
        input_tools.type_text("hello", app_name="Dexter")

    events = event_collector.drain()
    automation_events = [event for event in events if event.get("type") == "automation_action"]
    statuses = [event["payload"]["status"] for event in automation_events]

    assert "success" in statuses
    assert "focus_failed" in statuses
    assert any(event["payload"]["action"] == "type" for event in automation_events)
