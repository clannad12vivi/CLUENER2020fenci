"""监督式 HMM 命名实体识别模型。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence, Union

import joblib


Feature = dict[str, Union[str, bool]]
FeatureSequence = list[list[Feature]]
LabelSequence = list[list[str]]


def _observation(feature: Feature, mode: str = "char") -> str:
    """将特征字典压缩为 HMM 的低稀疏离散观测符号。"""
    if mode == "char":
        return str(feature.get("char", ""))
    if mode == "char_type":
        return "|".join(
            f"{field}={feature.get(field, '')}"
            for field in ("char", "char_type")
        )
    raise ValueError(f"不支持的 HMM 观测模式：{mode}")


class HMMNER:
    """使用监督计数和 Viterbi 解码的字符级 HMM。"""

    def __init__(self, alpha: float = 1.0, observation_mode: str = "char") -> None:
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        self.alpha = alpha
        if observation_mode not in {"char", "char_type"}:
            raise ValueError("observation_mode 必须是 char 或 char_type")
        self.observation_mode = observation_mode
        self.labels: list[str] = []
        self.vocabulary: set[str] = set()
        self.start_counts: Counter[str] = Counter()
        self.end_counts: Counter[str] = Counter()
        self.transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.emission_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, features: FeatureSequence, labels: LabelSequence) -> "HMMNER":
        if len(features) != len(labels):
            raise ValueError("特征序列和标签序列数量不一致")
        if not features:
            raise ValueError("训练数据不能为空")

        label_set: set[str] = set()
        for feature_sequence, label_sequence in zip(features, labels):
            if not feature_sequence or len(feature_sequence) != len(label_sequence):
                raise ValueError("单条特征序列和标签序列长度不一致")
            observations = [
                _observation(feature, self.observation_mode)
                for feature in feature_sequence
            ]
            self.vocabulary.update(observations)
            label_set.update(label_sequence)
            self.start_counts[label_sequence[0]] += 1
            self.end_counts[label_sequence[-1]] += 1
            for index, (observation, label) in enumerate(zip(observations, label_sequence)):
                self.emission_counts[label][observation] += 1
                if index:
                    self.transition_counts[label_sequence[index - 1]][label] += 1
        self.labels = sorted(label_set)
        if not self.labels:
            raise ValueError("训练数据中没有标签")
        return self

    def _log_start(self, label: str) -> float:
        denominator = sum(self.start_counts.values()) + self.alpha * len(self.labels)
        return math.log((self.start_counts[label] + self.alpha) / denominator)

    def _log_end(self, label: str) -> float:
        denominator = sum(self.end_counts.values()) + self.alpha * len(self.labels)
        return math.log((self.end_counts[label] + self.alpha) / denominator)

    def _log_transition(self, previous: str, current: str) -> float:
        denominator = sum(self.transition_counts[previous].values()) + (
            self.alpha * len(self.labels)
        )
        numerator = self.transition_counts[previous][current] + self.alpha
        return math.log(numerator / denominator)

    def _log_emission(self, label: str, observation: str) -> float:
        counts = self.emission_counts[label]
        denominator = sum(counts.values()) + self.alpha * (len(self.vocabulary) + 1)
        numerator = counts[observation] + self.alpha
        return math.log(numerator / denominator)

    def predict(self, features: FeatureSequence) -> LabelSequence:
        if not self.labels:
            raise RuntimeError("模型尚未训练")
        predictions: LabelSequence = []
        for feature_sequence in features:
            if not feature_sequence:
                predictions.append([])
                continue
            observations = [
                _observation(feature, self.observation_mode)
                for feature in feature_sequence
            ]
            scores: dict[str, float] = {
                label: self._log_start(label) + self._log_emission(label, observations[0])
                for label in self.labels
            }
            paths: dict[str, list[str]] = {label: [label] for label in self.labels}
            for observation in observations[1:]:
                next_scores: dict[str, float] = {}
                next_paths: dict[str, list[str]] = {}
                for current in self.labels:
                    candidates = [
                        (
                            scores[previous]
                            + self._log_transition(previous, current)
                            + self._log_emission(current, observation),
                            previous,
                        )
                        for previous in self.labels
                    ]
                    best_score, best_previous = max(candidates, key=lambda item: item[0])
                    next_scores[current] = best_score
                    next_paths[current] = paths[best_previous] + [current]
                scores, paths = next_scores, next_paths
            best_label = max(
                self.labels,
                key=lambda label: scores[label] + self._log_end(label),
            )
            predictions.append(paths[best_label])
        return predictions

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "HMMNER":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"模型文件不是 {cls.__name__} 类型")
        return model
