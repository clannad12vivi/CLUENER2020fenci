"""审计CLUENER2020数据规模、类别分布和BIO无法表示的嵌套标注。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data_process import read_cluener


def audit_file(path: Path) -> dict[str, Any]:
    records = read_cluener(path)
    entity_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    nested_count = 0
    partial_overlap_count = 0

    for record in records:
        spans: list[tuple[int, int, str]] = []
        for label, entities in (record.get("label") or {}).items():
            for spans_for_entity in entities.values():
                entity_counts[label] += len(spans_for_entity)
                sample_counts[label] += 1
                spans.extend(
                    (start, end + 1, label)
                    for start, end in spans_for_entity
                )
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if left[1] > right[0]:
                if (
                    left[0] <= right[0]
                    and right[1] <= left[1]
                ) or (
                    right[0] <= left[0]
                    and left[1] <= right[1]
                ):
                    nested_count += 1
                else:
                    partial_overlap_count += 1

    return {
        "samples": len(records),
        "entity_counts": dict(sorted(entity_counts.items())),
        "samples_with_entity": dict(sorted(sample_counts.items())),
        "nested_overlap_pairs": nested_count,
        "partial_overlap_pairs": partial_overlap_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计CLUENER2020数据集")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_summary.json"),
    )
    args = parser.parse_args()

    summary = {
        split: audit_file(args.data_dir / f"{split}.json")
        for split in ("train", "dev", "test")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
