"""基于 MultinomialNB 的逐字符命名实体识别基线模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

from .model_common import repair_bio_sequence


Feature = dict[str, Union[str, bool]]
FeatureSequence = list[list[Feature]]
LabelSequence = list[list[str]]


class NaiveBayesNER:
    """将每个字符独立分类为一个 BIO 标签的朴素贝叶斯模型。"""

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        self.vectorizer = DictVectorizer(sparse=True)
        self.fit_prior = fit_prior
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    def fit(self, features: FeatureSequence, labels: LabelSequence) -> "NaiveBayesNER":
        if len(features) != len(labels):
            raise ValueError("特征序列和标签序列数量不一致")
        flattened_features = [item for sequence in features for item in sequence]
        flattened_labels = [label for sequence in labels for label in sequence]
        if not flattened_features:
            raise ValueError("训练数据不能为空")
        if len(flattened_features) != len(flattened_labels):
            raise ValueError("特征数量和标签数量不一致")
        matrix = self.vectorizer.fit_transform(flattened_features)
        self.model.fit(matrix, flattened_labels)
        return self

    def predict(
        self,
        features: FeatureSequence,
        repair_bio: bool = True,
    ) -> LabelSequence:
        flattened_features = [item for sequence in features for item in sequence]
        if not flattened_features:
            return [[] for _ in features]
        flat_predictions = self.model.predict(self.vectorizer.transform(flattened_features))
        predictions: LabelSequence = []
        offset = 0
        for sequence in features:
            sequence_prediction = flat_predictions[offset : offset + len(sequence)].tolist()
            predictions.append(
                repair_bio_sequence(sequence_prediction)
                if repair_bio
                else sequence_prediction
            )
            offset += len(sequence)
        return predictions

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesNER":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"模型文件不是 {cls.__name__} 类型")
        return model
