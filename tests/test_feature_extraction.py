import tempfile
import unittest
from pathlib import Path

from src.feature_extraction import (
    bio_to_features,
    read_bio,
    sentence_to_features,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_sentence_features_include_context_and_word_features(self):
        features = sentence_to_features("北京大学")
        self.assertEqual(len(features), 4)
        self.assertEqual(features[0]["char"], "北")
        self.assertEqual(features[0]["prev_char"], "BOS")
        self.assertEqual(features[0]["next_char"], "京")
        self.assertTrue(features[0]["is_first"])
        self.assertEqual(features[-1]["next_char"], "EOS")
        self.assertTrue(features[-1]["is_last"])
        self.assertIn("word", features[0])
        self.assertIn("part_of_speech", features[0])

    def test_empty_text_returns_empty_features(self):
        self.assertEqual(sentence_to_features(""), [])

    def test_bio_file_is_read_and_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bio"
            path.write_text("北\tB-LOC\n京\tI-LOC\n\n", encoding="utf-8")
            sentences = read_bio(path)
            features, labels = bio_to_features(sentences)
            self.assertEqual(len(features), 1)
            self.assertEqual(len(features[0]), 2)
            self.assertEqual(labels, [["B-LOC", "I-LOC"]])


if __name__ == "__main__":
    unittest.main()
