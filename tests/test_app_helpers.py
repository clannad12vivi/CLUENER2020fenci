import unittest

from app import decode_entities, render_entities


class AppHelperTests(unittest.TestCase):
    def test_decode_entities(self):
        entities = decode_entities("北京大学张三", ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "B-PER", "I-PER"])
        self.assertEqual(
            entities,
            [
                {"text": "北京大学", "type": "ORG", "start": 0, "end": 4},
                {"text": "张三", "type": "PER", "start": 4, "end": 6},
            ],
        )

    def test_render_entities_contains_entity_and_type(self):
        html = render_entities(
            "张三",
            [{"text": "张三", "type": "PER", "start": 0, "end": 2}],
        )
        self.assertIn("张三", html)
        self.assertIn("人名", html)


if __name__ == "__main__":
    unittest.main()
