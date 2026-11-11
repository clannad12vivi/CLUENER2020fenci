"""CRF 命名实体识别模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import joblib
import sklearn_crfsuite

from .feature_extraction import sentence_to_features


FeatureSequence = list[list[dict[str, Union[str, bool]]]]
LabelSequence = list[list[str]]


class CRFNER:
    """基于 sklearn-crfsuite 的字符级 CRF 模型。"""

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须大于 0")
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def fit(self, features: FeatureSequence, labels: LabelSequence) -> "CRFNER":
        if len(features) != len(labels):
            raise ValueError("特征序列和标签序列数量不一致")
        if not features:
            raise ValueError("训练数据不能为空")
        self.model.fit(features, labels)
        return self

    def predict(self, features: FeatureSequence) -> LabelSequence:
        if not features:
            return []
        return self.model.predict(features)

    def predict_texts(self, texts: Sequence[str]) -> LabelSequence:
        features = [sentence_to_features(text) for text in texts]
        return self.predict(features)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "CRFNER":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"模型文件不是 {cls.__name__} 类型")
        return model
