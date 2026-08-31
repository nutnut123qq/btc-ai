import unittest
import sys
from pathlib import Path

# Add btc-ai to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph import _strip_json_fences, _extract_first_json_object, _parse_json_strict


class TestGraphParsers(unittest.TestCase):
    """
    Hermetic unit tests for LLM output parsing, markdown fence stripping,
    and graceful JSON extraction in graph.py.
    """

    def test_strip_json_fences(self):
        raw = "```json\n{\"forecast\": \"BULLISH\", \"confidence\": 80}\n```"
        cleaned = _strip_json_fences(raw)
        self.assertEqual(cleaned, "{\"forecast\": \"BULLISH\", \"confidence\": 80}")

    def test_extract_first_json_object(self):
        text = "Here is the result:\n{\"status\": \"ok\", \"val\": 123}\nHope this helps!"
        extracted = _extract_first_json_object(text)
        self.assertEqual(extracted, "{\"status\": \"ok\", \"val\": 123}")

    def test_parse_json_strict_valid(self):
        raw = "```json\n{\"forecast\": \"BEARISH\", \"confidence\": 75}\n```"
        parsed = _parse_json_strict(raw)
        self.assertEqual(parsed["forecast"], "BEARISH")
        self.assertEqual(parsed["confidence"], 75)

    def test_parse_json_strict_surrounding_text(self):
        raw = "Analysis complete.\n```json\n{\"forecast\": \"BULLISH\", \"confidence\": 90}\n```\nEnd of response."
        parsed = _parse_json_strict(raw)
        self.assertEqual(parsed["forecast"], "BULLISH")
        self.assertEqual(parsed["confidence"], 90)

if __name__ == "__main__":
    unittest.main()
