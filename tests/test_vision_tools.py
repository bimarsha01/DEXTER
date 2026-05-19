from unittest import mock

from PIL import Image

from tools.vision_tools import ScreenCaptureResult, capture_screen_for_vision, describe_screen_without_vision


@mock.patch("tools.vision_tools._get_foreground_window_bbox", return_value=("Visual Studio Code", (0, 0, 1920, 1080)))
@mock.patch("tools.vision_tools.ImageGrab.grab")
def test_capture_screen_for_vision_uses_full_screen(grab_mock, foreground_mock):
    grab_mock.return_value = Image.new("RGB", (1600, 900), color="white")

    result = capture_screen_for_vision(max_dimension=1280)

    assert isinstance(result, ScreenCaptureResult)
    assert result.capture_mode == "full_screen"
    assert result.foreground_window == "Visual Studio Code"
    grab_mock.assert_called_once_with(all_screens=True)
    assert result.image_bytes


@mock.patch("tools.vision_tools._caption_screen_locally", return_value="a web browser showing Facebook")
@mock.patch("tools.vision_tools.capture_screen_for_vision")
def test_describe_screen_without_vision_uses_local_caption(capture_mock, caption_mock):
    capture_mock.return_value = ScreenCaptureResult(
        image_bytes=b"fake-bytes",
        foreground_window="Facebook - Brave",
        capture_mode="full_screen",
    )

    result = describe_screen_without_vision()

    assert "a web browser showing Facebook" in result
    assert "Brave" in result
