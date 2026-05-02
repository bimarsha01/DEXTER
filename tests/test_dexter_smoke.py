import os
import unittest
from unittest import mock

import utils.config as dexter_config
from core.wake_word.detector import WakeWordDetector
from tools.audit_tool_schemas import build_report
from utils.config import get_config, config_validation_warnings


class DexterSmokeTests(unittest.TestCase):
    def test_schema_audit_is_clean(self):
        report = build_report()
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["extra"], [])

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
        self.assertIsInstance(warnings, list)


if __name__ == "__main__":
    unittest.main()
