from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

LABEL_MAPPING = {
    "name": "PER",
    "address": "LOC",
    "book": "BOOK",
    "company": "COMP",
    "game": "GAME",
    "government": "GOV",
    "movie": "MOV",
    "organization": "ORG",
    "position": "POS",
    "scene": "SCENE",
}
TARGET_LABELS = frozenset(LABEL_MAPPING.values())


def read_cluener(path: str | Path) -> list[dict[str, Any]]:
    """读取 CLUENER 的 JSONL 文件并返回记录列表。"""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{input_path}")

    records: list[dict[str, Any]] = []
    # utf-8-sig 同时兼容无 BOM 和 Windows 工具生成的 UTF-8 BOM 文件。
    with input_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path} 第 {line_number} 行不是有效 JSON") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{input_path} 第 {line_number} 行缺少有效 text 字段")
            records.append(record)
    return records


def _entity_spans(record: dict[str, Any]) -> Iterator[tuple[int, int, str]]:
    """产生实体的半开区间 [start, end) 和目标标签。"""
    text = record["text"]
    labels = record.get("label", {})
    if not isinstance(labels, dict):
        raise ValueError("label 字段必须是对象")

    for source_label, entities in labels.items():
        target_label = LABEL_MAPPING.get(source_label)
        if target_label is None:
            continue
        if not isinstance(entities, dict):
            raise ValueError(f"实体标签 {source_label} 的值必须是对象")
        for entity_text, spans in entities.items():
            if not isinstance(entity_text, str) or not isinstance(spans, list):
                raise ValueError(f"实体 {source_label} 的格式不正确")
            for span in spans:
                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or not all(isinstance(index, int) for index in span)
                ):
                    raise ValueError(f"实体 {entity_text} 的位置格式不正确")
                start, inclusive_end = span
                end = inclusive_end + 1
                if start < 0 or end > len(text) or start >= end:
                    raise ValueError(
                        f"实体 {entity_text} 的位置越界：[{start}, {inclusive_end}]"
                    )
                if text[start:end] != entity_text:
                    raise ValueError(
                        f"实体文本与位置不一致：期望 {entity_text!r}，"
                        f"实际为 {text[start:end]!r}"
                    )
                yield start, end, target_label


def record_to_bio(record: dict[str, Any]) -> list[tuple[str, str]]:
    """将一条 CLUENER 记录转换为字符和 BIO 标签。"""
    text = record["text"]
    labels = ["O"] * len(text)
    spans = list(_entity_spans(record))
    spans.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[1]))
    occupied: list[tuple[int, int, str]] = []

    for start, end, entity_label in spans:
        overlap = next(
            (
                (old_start, old_end, old_label)
                for old_start, old_end, old_label in occupied
                if start < old_end and end > old_start
            ),
            None,
        )
        if overlap:
            old_start, old_end, old_label = overlap
            if old_start <= start and end <= old_end:
                continue
            raise ValueError(
                f"实体区间存在部分重叠：[{start}, {end})/{entity_label} "
                f"与 [{old_start}, {old_end})/{old_label}"
            )
        occupied.append((start, end, entity_label))

    for start, end, entity_label in sorted(occupied):
        labels[start] = f"B-{entity_label}"
        for index in range(start + 1, end):
            labels[index] = f"I-{entity_label}"

    return list(zip(text, labels))


def write_bio(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """将记录写为常见的字符 TAB 标签格式，每条文本空行分隔。"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            for character, label in record_to_bio(record):
                file.write(f"{character}\t{label}\n")
            file.write("\n")


def convert_file(input_path: str | Path, output_path: str | Path) -> int:
    """转换一个 JSONL 文件并返回样本数量。"""
    records = read_cluener(input_path)
    write_bio(records, output_path)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="将 CLUENER JSONL 转换为字符级 BIO")
    parser.add_argument("input", type=Path, help="输入文件，例如 data/train.json")
    parser.add_argument("output", type=Path, help="输出文件，例如 data/train.bio")
    args = parser.parse_args()
    count = convert_file(args.input, args.output)
    print(f"转换完成：{count} 条样本 -> {args.output}")


if __name__ == "__main__":
    main()
