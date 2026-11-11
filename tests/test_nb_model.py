import tempfile
import unittest
from pathlib import Path

from src.nb_model import NaiveBayesNER


class NaiveBayesModelTests(unittest.TestCase):
    def test_nb_can_train_predict_and_reload(self):
        features = [
            [
                {"char": "张", "char_type": "CHINESE", "prev_char": "BOS"},
                {"char": "三", "char_type": "CHINESE", "prev_char": "张"},
            ],
            [
                {"char": "李", "char_type": "CHINESE", "prev_char": "BOS"},
                {"char": "四", "char_type": "CHINESE", "prev_char": "李"},
            ],
        ]
        labels = [["B-PER", "I-PER"], ["B-PER", "I-PER"]]
        model = NaiveBayesNER().fit(features, labels)
        predictions = model.predict(features)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0]), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nb.pkl"
            model.save(path)
            loaded = NaiveBayesNER.load(path)
            self.assertEqual(loaded.predict(features), predictions)

    def test_prediction_repairs_invalid_i_tag(self):
        features = [[{"char": "张", "char_type": "CHINESE"}]]
        labels = [["I-PER"]]
        model = NaiveBayesNER().fit(features, labels)
        prediction = model.predict(features)
        self.assertEqual(prediction, [["B-PER"]])


if __name__ == "__main__":
    unittest.main()
