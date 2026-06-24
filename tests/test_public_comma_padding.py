import json
import tempfile
import unittest
from pathlib import Path

from src.demo.comma_padding import build_comma_prompt, make_prompt_variant, slugify_prompt, write_prompt_metadata


class CommaPaddingTests(unittest.TestCase):
    def test_build_comma_prompt_default_separator(self):
        self.assertEqual(build_comma_prompt("a red cube", repeat=3), "a red cube ,,,")

    def test_build_comma_prompt_no_separator(self):
        self.assertEqual(build_comma_prompt("a red cube", repeat=3, separator=""), "a red cube,,,")

    def test_build_comma_prompt_zero_repeat(self):
        self.assertEqual(build_comma_prompt("a red cube", repeat=0), "a red cube")

    def test_build_comma_prompt_rejects_invalid_repeat(self):
        with self.assertRaises(ValueError):
            build_comma_prompt("a red cube", repeat=-1)

    def test_slugify_prompt(self):
        self.assertEqual(slugify_prompt("A puppy: on a screen!"), "a_puppy_on_a_screen")

    def test_write_prompt_metadata(self):
        variant = make_prompt_variant("a red cube", repeat=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            write_prompt_metadata(path, variant, seed=42)
            data = json.loads(path.read_text())
        self.assertEqual(data["original_prompt"], "a red cube")
        self.assertEqual(data["modified_prompt"], "a red cube ,,")
        self.assertEqual(data["seed"], 42)


if __name__ == "__main__":
    unittest.main()
