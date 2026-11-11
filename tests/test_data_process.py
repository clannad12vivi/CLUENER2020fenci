import unittest
import tempfile
from pathlib import Path

from src.data_process import read_cluener, record_to_bio


class DataProcessTests(unittest.TestCase):
    def test_target_entities_are_converted_to_bio(self):
        record = {
            "text": "北京大学校长张三",
            "label": {
                "organization": {"北京大学": [[0, 3]]},
                "name": {"张三": [[6, 7]]},
            },
        }
        self.assertEqual(
            record_to_bio(record),
            [
                ("北", "B-ORG"),
                ("京", "I-ORG"),
                ("大", "I-ORG"),
                ("学", "I-ORG"),
                ("校", "O"),
                ("长", "O"),
                ("张", "B-PER"),
                ("三", "I-PER"),
            ],
        )

    def test_all_cluener_entities_are_converted(self):
        record = {
            "text": "喜欢电影",
            "label": {"movie": {"电影": [[2, 3]]}},
        }
        self.assertEqual(
            record_to_bio(record),
            [
                ("喜", "O"),
                ("欢", "O"),
                ("电", "B-MOV"),
                ("影", "I-MOV"),
            ],
        )

    def test_partially_overlapping_entity_is_rejected(self):
        record = {
            "text": "北京大学城",
            "label": {
                "book": {"北京大学": [[0, 3]]},
                "organization": {"学城": [[3, 4]]},
            },
        }
        with self.assertRaises(ValueError):
            record_to_bio(record)

    def test_invalid_span_is_rejected(self):
        record = {
            "text": "北京",
            "label": {"address": {"上海": [[0, 1]]}},
        }
        with self.assertRaises(ValueError):
            record_to_bio(record)

    def test_bom_encoded_jsonl_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(
                '{"text":"张三","label":{"name":{"张三":[[0,1]]}}}\n',
                encoding="utf-8-sig",
            )
            self.assertEqual(len(read_cluener(path)), 1)


if __name__ == "__main__":
    unittest.main()
