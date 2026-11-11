import tempfile
import unittest
from pathlib import Path

from src.hmm_model import HMMNER


class HMMModelTests(unittest.TestCase):
    def test_hmm_can_train_predict_and_reload(self):
        features = [
            [
                {"char": "张", "char_type": "CHINESE", "prev_char": "BOS", "next_char": "三"},
                {"char": "三", "char_type": "CHINESE", "prev_char": "张", "next_char": "EOS"},
            ],
            [
                {"char": "李", "char_type": "CHINESE", "prev_char": "BOS", "next_char": "四"},
                {"char": "四", "char_type": "CHINESE", "prev_char": "李", "next_char": "EOS"},
            ],
        ]
        labels = [["B-PER", "I-PER"], ["B-PER", "I-PER"]]
        model = HMMNER(alpha=1.0).fit(features, labels)
        predictions = model.predict(features)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0]), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hmm.pkl"
            model.save(path)
            loaded = HMMNER.load(path)
            self.assertEqual(loaded.predict(features), predictions)

    def test_unknown_observation_is_decodable(self):
        features = [[{"char": "张", "char_type": "CHINESE"}]]
        labels = [["B-PER"]]
        model = HMMNER().fit(features, labels)
        prediction = model.predict([[{"char": "新", "char_type": "CHINESE"}]])
        self.assertEqual(len(prediction[0]), 1)


if __name__ == "__main__":
    unittest.main()
