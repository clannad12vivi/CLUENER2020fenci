"""三个基线模型共用的数据、标签和评估工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .feature_extraction import bio_to_features, read_bio


SequenceLabels = list[list[str]]


def load_dataset(
    path: str | Path,
    limit: int | None = None,
) -> tuple[list[list[dict[str, str | bool]]], SequenceLabels, list[str]]:
    """读取 BIO 数据并生成 CRF 特征、标签和原始文本。"""
    sentences = read_bio(path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        sentences = sentences[:limit]
    features, labels = bio_to_features(sentences)
    texts = ["".join(characters) for characters, _ in sentences]
    return features, labels, texts


def repair_bio_sequence(labels: Sequence[str]) -> list[str]:
    """修正独立分类器可能产生的非法 BIO 序列。"""
    repaired: list[str] = []
    previous_type: str | None = None
    for label in labels:
        if label == "O":
            repaired.append("O")
            previous_type = None
            continue
        if "-" not in label:
            repaired.append("O")
            previous_type = None
            continue
        prefix, entity_type = label.split("-", 1)
        if not entity_type:
            repaired.append("O")
            previous_type = None
        elif prefix == "I" and previous_type == entity_type:
            repaired.append(label)
        else:
            repaired.append(f"B-{entity_type}")
        previous_type = entity_type if repaired[-1] != "O" else None
    return repaired


def calculate_metrics(
    true_labels: Sequence[Sequence[str]],
    predicted_labels: Sequence[Sequence[str]],
) -> dict[str, float]:
    """计算实体级 Precision、Recall 和 F1。"""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("真实标签和预测标签的样本数不一致")
    return {
        "precision": float(precision_score(true_labels, predicted_labels)),
        "recall": float(recall_score(true_labels, predicted_labels)),
        "f1": float(f1_score(true_labels, predicted_labels)),
    }


def calculate_entity_metrics(
    true_labels: Sequence[Sequence[str]],
    predicted_labels: Sequence[Sequence[str]],
) -> dict[str, dict[str, float]]:
    """计算 PER、LOC、ORG 三类实体的实体级指标。"""
    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, dict[str, float]] = {}
    entity_types = sorted(
        key for key in report if key not in {"micro avg", "macro avg", "weighted avg"}
    )
    for entity_type in entity_types:
        values = report.get(entity_type, {})
        result[entity_type] = {
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1": float(values.get("f1-score", 0.0)),
        }
    return result


def flatten_entity_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """将分类别指标展开为适合 CSV 保存的字段。"""
    flattened: dict[str, float] = {}
    for entity_type, values in metrics.items():
        prefix = entity_type.lower()
        flattened[f"{prefix}_precision"] = values["precision"]
        flattened[f"{prefix}_recall"] = values["recall"]
        flattened[f"{prefix}_f1"] = values["f1"]
    return flattened


def save_metrics(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """将多个模型的指标和训练耗时保存为 CSV。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("不能保存空的指标结果")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(
    path: str | Path,
    texts: Sequence[str],
    true_labels: Sequence[Sequence[str]],
    predictions: dict[str, Sequence[Sequence[str]]],
) -> None:
    """将每条文本的真实标签和各模型预测结果保存为 JSONL。"""
    if not (len(texts) == len(true_labels) == len(next(iter(predictions.values()), []))):
        raise ValueError("文本、真实标签和预测结果数量不一致")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for index, (text, labels) in enumerate(zip(texts, true_labels)):
            record: dict[str, Any] = {
                "text": text,
                "true_labels": list(labels),
                "index": index,
            }
            for model_name, model_predictions in predictions.items():
                record[model_name] = list(model_predictions[index])
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
