import tempfile
import unittest
from pathlib import Path

from src.crf_model import CRFNER


class CRFModelTests(unittest.TestCase):
    def test_crf_can_train_predict_and_reload(self):
        features = [
            [
                {"char": "张", "prev_char": "BOS", "next_char": "三"},
                {"char": "三", "prev_char": "张", "next_char": "EOS"},
            ],
            [
                {"char": "李", "prev_char": "BOS", "next_char": "四"},
                {"char": "四", "prev_char": "李", "next_char": "EOS"},
            ],
        ]
        labels = [["B-PER", "I-PER"], ["B-PER", "I-PER"]]
        model = CRFNER(max_iterations=20).fit(features, labels)
        predictions = model.predict(features)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0]), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crf.pkl"
            model.save(path)
            loaded = CRFNER.load(path)
            self.assertEqual(
                loaded.predict(features).tolist(),
                predictions.tolist() if hasattr(predictions, "tolist") else predictions,
            )


if __name__ == "__main__":
    unittest.main()
