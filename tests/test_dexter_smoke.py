import os
import unittest
from unittest import mock

import utils.config as dexter_config
from core.brain.intent_router import IntentRouter
from core.pipeline import AsyncPipeline
from core.wake_word.detector import WakeWordDetector
from tools.audit_tool_schemas import build_report
from tools.pc_controls import open_application
from tools.system_tools import get_current_time
from tools.web_browser import open_url_in_browser, search_content_platform
from utils.config import DexterConfig, DefaultsConfig
from utils.config import get_config, config_validation_warnings
from utils.transcript_correction import TranscriptCorrector


class DexterSmokeTests(unittest.TestCase):
    def test_schema_audit_is_clean(self):
        report = build_report()
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["extra"], [])

    def test_actionable_utterance_detection(self):
        self.assertTrue(AsyncPipeline._looks_actionable_utterance("Open Chrome."))
        self.assertTrue(AsyncPipeline._looks_actionable_utterance("What is the current temperature in Mumbai?"))
        self.assertFalse(AsyncPipeline._looks_actionable_utterance("hmm okay"))

    def test_wake_word_detector_prefix_match(self):
        detector = WakeWordDetector(["hey"], match_mode="prefix")
        detection = detector.detect("Hey Dexter, what time is it?")
        self.assertTrue(detection.triggered)
        self.assertEqual(detection.phrase, "hey")
        self.assertIn("Dexter", detection.cleaned_text)

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "smoke-test-placeholder"}, clear=False)
    def test_config_loads_and_validates(self):
        dexter_config._CONFIG = None
        cfg = get_config()
        warnings = config_validation_warnings(cfg)
        self.assertIsInstance(cfg.wake_words, list)
        self.assertTrue(cfg.wake_words)
        self.assertEqual(cfg.defaults.city, "Kathmandu")
        self.assertIsInstance(warnings, list)

    def test_weather_city_is_extracted_and_defaults(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))
        decision = router.detect_intent("what is the weather in Kathmandu?")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool_name, "get_weather")
        self.assertEqual(decision.args["city"], "Kathmandu")

        temperature_decision = router.detect_intent("what is the current temperature of Mumbai?")
        self.assertEqual(temperature_decision.action, "tool")
        self.assertEqual(temperature_decision.tool_name, "get_weather")
        self.assertEqual(temperature_decision.args["city"], "Mumbai")

        default_decision = router.detect_intent("what is the weather")
        self.assertEqual(default_decision.args["city"], "Kathmandu")

    def test_time_in_city_routes_to_time_tool(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))
        decision = router.detect_intent("what is the current time in Mumbai?")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool_name, "get_current_time")
        self.assertEqual(decision.args["city"], "Mumbai")

    def test_current_time_tool_supports_city_timezones(self):
        result = get_current_time("Mumbai")
        self.assertIn("Mumbai", result)
        self.assertIn("The current time in Mumbai is", result)

    def test_media_intents_route_to_content_platform_search(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))

        decision = router.detect_intent("play Blinding Lights by The Weeknd on Spotify")
        self.assertEqual(decision.tool_name, "search_content_platform")
        self.assertEqual(decision.args["platform"], "spotify")
        self.assertEqual(decision.args["content_type"], "music")

        youtube_music_decision = router.detect_intent("play lo-fi music on YouTube Music")
        self.assertEqual(youtube_music_decision.tool_name, "search_content_platform")
        self.assertEqual(youtube_music_decision.args["platform"], "youtube music")
        self.assertEqual(youtube_music_decision.args["content_type"], "music")

        netflix_decision = router.detect_intent("watch Friends on Netflix")
        self.assertEqual(netflix_decision.tool_name, "search_content_platform")
        self.assertEqual(netflix_decision.args["platform"], "netflix")
        self.assertEqual(netflix_decision.args["content_type"], "movie")

        video_decision = router.detect_intent("find the latest F1 highlights on ESPN")
        self.assertEqual(video_decision.tool_name, "search_content_platform")
        self.assertEqual(video_decision.args["platform"], "espn")
        self.assertEqual(video_decision.args["content_type"], "sports")

        search_decision = router.detect_intent("search for Inception on Netflix")
        self.assertEqual(search_decision.tool_name, "search_content_platform")
        self.assertEqual(search_decision.args["platform"], "netflix")
        self.assertEqual(search_decision.args["content_type"], "movie")

        soundcloud_decision = router.detect_intent("play this song on SoundCloud")
        self.assertEqual(soundcloud_decision.tool_name, "search_content_platform")
        self.assertEqual(soundcloud_decision.args["platform"], "soundcloud")
        self.assertEqual(soundcloud_decision.args["content_type"], "music")

        default_decision = router.detect_intent("play something")
        self.assertEqual(default_decision.tool_name, "search_content_platform")
        self.assertEqual(default_decision.args["platform"], "youtube music")
        self.assertEqual(default_decision.args["content_type"], "music")

    def test_open_chrome_routes_directly_to_open_application(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))
        decision = router.detect_intent("open chrome")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool_name, "open_application")
        self.assertEqual(decision.args["app_name"], "chrome")

    def test_stripped_browser_command_routes_to_browser_tool(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))
        decision = router.detect_intent("youtube in chrome")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool_name, "open_url_in_browser")
        self.assertEqual(decision.args["browser"], "chrome")
        self.assertEqual(decision.args["url"], "https://www.youtube.com")

    def test_logged_browser_command_routes_to_browser_tool(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))
        decision = router.detect_intent("open YouTube in Google Chrome")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool_name, "open_url_in_browser")
        self.assertEqual(decision.args["browser"], "google chrome")
        self.assertEqual(decision.args["url"], "https://www.youtube.com")

    def test_stripped_media_command_routes_to_platform_search(self):
        router = IntentRouter(DexterConfig(defaults=DefaultsConfig(city="Kathmandu")))
        decision = router.detect_intent("Justin Bieber's Baby from Spotify")
        self.assertEqual(decision.action, "tool")
        self.assertEqual(decision.tool_name, "search_content_platform")
        self.assertEqual(decision.args["platform"], "spotify")
        self.assertEqual(decision.args["content_type"], "music")

    def test_transcript_corrector_does_not_mangle_browser_commands(self):
        corrector = TranscriptCorrector()
        result = corrector.correct("Open YouTube in Google Chrome.")
        self.assertEqual(result.corrected, "Open YouTube in Google Chrome.")

    @mock.patch("tools.pc_controls._resolve_command", return_value=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
    @mock.patch("tools.pc_controls.subprocess.Popen")
    @mock.patch("tools.pc_controls.os.path.exists", return_value=True)
    def test_open_application_resolves_executable_path(self, exists_mock, popen_mock, resolve_mock):
        message = open_application("chrome")
        self.assertIn("Successfully opened chrome", message)
        popen_mock.assert_called_once()
        args, kwargs = popen_mock.call_args
        self.assertEqual(os.path.normpath(args[0][0]), os.path.normpath(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))

    @mock.patch("tools.web_browser._resolve_browser_executable", return_value=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
    @mock.patch("tools.web_browser.subprocess.Popen")
    def test_open_url_in_browser_uses_specific_executable(self, popen_mock, resolve_mock):
        message = open_url_in_browser("youtube.com", "chrome")
        self.assertIn("Successfully opened https://youtube.com in chrome", message)
        popen_mock.assert_called_once()
        args, kwargs = popen_mock.call_args
        self.assertEqual(os.path.normpath(args[0][0]), os.path.normpath(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
        self.assertEqual(args[0][1], "https://youtube.com")

    @mock.patch("tools.web_browser.webbrowser.open")
    def test_search_content_platform_builds_known_platform_url(self, open_mock):
        message = search_content_platform("lo-fi music", platform="YouTube Music")
        self.assertIn("youtube music", message.lower())
        open_mock.assert_called_once()
        self.assertIn("music.youtube.com/search?q=lo-fi+music", open_mock.call_args[0][0])

    @mock.patch("tools.web_browser.webbrowser.open")
    def test_search_content_platform_falls_back_for_unknown_platform(self, open_mock):
        message = search_content_platform("Inception", platform="SomeNewPlatform")
        self.assertIn("Successfully opened", message)
        open_mock.assert_called_once()
        self.assertIn("google.com/search", open_mock.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
