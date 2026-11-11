import tempfile
import unittest
from pathlib import Path

from src.model_common import (
    calculate_metrics,
    calculate_entity_metrics,
    flatten_entity_metrics,
    load_dataset,
    repair_bio_sequence,
    save_metrics,
    save_predictions,
)


class ModelCommonTests(unittest.TestCase):
    def test_load_dataset_supports_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bio"
            path.write_text("张\tB-PER\n三\tI-PER\n\n", encoding="utf-8")
            features, labels, texts = load_dataset(path, limit=1)
            self.assertEqual(len(features), 1)
            self.assertEqual(labels, [["B-PER", "I-PER"]])
            self.assertEqual(texts, ["张三"])

    def test_repair_bio_sequence(self):
        self.assertEqual(
            repair_bio_sequence(["I-PER", "I-PER", "O", "I-LOC"]),
            ["B-PER", "I-PER", "O", "B-LOC"],
        )

    def test_metrics_and_result_files(self):
        true_labels = [["B-PER", "I-PER", "O"]]
        predictions = [["B-PER", "I-PER", "O"]]
        self.assertEqual(calculate_metrics(true_labels, predictions)["f1"], 1.0)
        self.assertEqual(calculate_entity_metrics(true_labels, predictions)["PER"]["f1"], 1.0)
        self.assertEqual(flatten_entity_metrics({"PER": {"precision": 1.0, "recall": 1.0, "f1": 1.0}})["per_precision"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.csv"
            predictions_path = Path(directory) / "predictions.jsonl"
            save_metrics(metrics_path, [{"model": "test", "f1": 1.0}])
            save_predictions(predictions_path, ["张三"], true_labels, {"test": predictions})
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(predictions_path.is_file())


if __name__ == "__main__":
    unittest.main()
