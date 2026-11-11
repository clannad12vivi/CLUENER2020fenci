"""训练并比较 CRF、HMM 和朴素贝叶斯三个 NER 模型。"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from .crf_model import CRFNER
from .hmm_model import HMMNER
from .model_common import (
    calculate_metrics,
    calculate_entity_metrics,
    flatten_entity_metrics,
    load_dataset,
    save_metrics,
    save_predictions,
)
from .nb_model import NaiveBayesNER


def _as_lists(predictions: Any) -> list[list[str]]:
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    return [list(sequence) for sequence in predictions]


def _limit_value(value: int) -> int | None:
    if value < 0:
        raise ValueError("样本数量不能为负数")
    return None if value == 0 else value


def _split_dataset(
    features: list[list[dict[str, str | bool]]],
    labels: list[list[str]],
    texts: list[str],
    test_ratio: float,
    seed: int,
) -> tuple[
    tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
    tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
]:
    if not 0 < test_ratio < 1:
        raise ValueError("internal_test_ratio 必须在 0 和 1 之间")
    if not (len(features) == len(labels) == len(texts)):
        raise ValueError("训练特征、标签和文本数量不一致")
    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    test_size = max(1, int(len(indices) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    def select(
        selected_indices: list[int],
    ) -> tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]]:
        return (
            [features[index] for index in selected_indices],
            [labels[index] for index in selected_indices],
            [texts[index] for index in selected_indices],
        )

    return select(train_indices), select(test_indices)


def _split_three_way(
    features: list[list[dict[str, str | bool]]],
    labels: list[list[str]],
    texts: list[str],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[
    tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
    tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
    tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
]:
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio 和 test_ratio 必须为正且总和小于 1")
    if not (len(features) == len(labels) == len(texts)):
        raise ValueError("训练特征、标签和文本数量不一致")
    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    test_size = max(1, int(len(indices) * test_ratio))
    validation_size = max(1, int(len(indices) * validation_ratio))
    test_indices = indices[:test_size]
    validation_indices = indices[test_size : test_size + validation_size]
    train_indices = indices[test_size + validation_size :]

    def select(
        selected_indices: list[int],
    ) -> tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]]:
        return (
            [features[index] for index in selected_indices],
            [labels[index] for index in selected_indices],
            [texts[index] for index in selected_indices],
        )

    return select(train_indices), select(validation_indices), select(test_indices)


def _combine_datasets(
    first: tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
    second: tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]],
) -> tuple[list[list[dict[str, str | bool]]], list[list[str]], list[str]]:
    return (
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )


def run_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    train_limit = _limit_value(args.train_limit)
    dev_limit = _limit_value(args.dev_limit)
    parameters: dict[str, dict[str, Any]] = {}
    if args.params_file:
        with Path(args.params_file).open("r", encoding="utf-8-sig") as file:
            parameters = json.load(file)

    all_train_features, all_train_labels, all_train_texts = load_dataset(
        data_dir / "train.bio", train_limit
    )
    if args.use_internal_test:
        (
            train_part,
            validation_part,
            internal_test_part,
        ) = _split_three_way(
            all_train_features,
            all_train_labels,
            all_train_texts,
            args.internal_validation_ratio,
            args.internal_test_ratio,
            args.seed,
        )
        train_features, train_labels, _ = _combine_datasets(
            train_part, validation_part
        )
        (
            internal_test_features,
            internal_test_labels,
            internal_test_texts,
        ) = internal_test_part
        split_manifest = {
            "source": "data/train.json",
            "seed": args.seed,
            "validation_ratio": args.internal_validation_ratio,
            "internal_test_ratio": args.internal_test_ratio,
            "official_train_samples": len(all_train_features),
            "internal_train_samples": len(train_part[0]),
            "internal_validation_samples": len(validation_part[0]),
            "internal_test_samples": len(internal_test_features),
            "final_fit_samples": len(train_features),
            "official_dev_samples": None,
        }
    else:
        train_features, train_labels, _ = (
            all_train_features,
            all_train_labels,
            all_train_texts,
        )
        internal_test_features, internal_test_labels, internal_test_texts = [], [], []
    dev_features, dev_labels, dev_texts = load_dataset(data_dir / "dev.bio", dev_limit)
    if args.use_internal_test:
        split_manifest["official_dev_samples"] = len(dev_features)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "data_split_manifest.json").write_text(
            json.dumps(split_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    model_builders = {
        "CRF": lambda: CRFNER(
            c1=parameters.get("CRF", {}).get("c1", 0.1),
            c2=parameters.get("CRF", {}).get("c2", 0.1),
            max_iterations=parameters.get(
                "CRF", {}
            ).get("max_iterations", args.crf_max_iterations),
        ),
        "HMM": lambda: HMMNER(
            alpha=parameters.get("HMM", {}).get("alpha", 1.0),
            observation_mode=parameters.get("HMM", {}).get(
                "observation_mode", "char"
            ),
        ),
        "NaiveBayes": lambda: NaiveBayesNER(
            alpha=parameters.get("NaiveBayes", {}).get("alpha", 1.0),
            fit_prior=parameters.get("NaiveBayes", {}).get("fit_prior", True),
        ),
    }
    predictions: dict[str, dict[str, list[list[str]]]] = {
        "dev": {},
        "internal_test": {},
    }
    metric_rows: list[dict[str, Any]] = []

    for model_name, builder in model_builders.items():
        print(f"\n[{model_name}] 开始训练，样本数：{len(train_features)}")
        model = builder()
        start = time.perf_counter()
        model.fit(train_features, train_labels)
        train_seconds = time.perf_counter() - start

        model_path = model_dir / f"{model_name.lower()}_model.pkl"
        model.save(model_path)
        print(f"[{model_name}] 模型已保存：{model_path}")

        dev_prediction = _as_lists(model.predict(dev_features))
        internal_test_prediction = (
            _as_lists(model.predict(internal_test_features))
            if internal_test_features
            else []
        )
        predictions["dev"][model_name] = dev_prediction
        predictions["internal_test"][model_name] = internal_test_prediction

        splits = [("dev", dev_labels, dev_prediction)]
        if internal_test_features:
            splits.append(("internal_test", internal_test_labels, internal_test_prediction))
        for split_name, true_labels, predicted in splits:
            metrics = calculate_metrics(true_labels, predicted)
            entity_metrics = calculate_entity_metrics(true_labels, predicted)
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "train_samples": len(train_features),
                    "precision": round(metrics["precision"], 6),
                    "recall": round(metrics["recall"], 6),
                    "f1": round(metrics["f1"], 6),
                    "train_seconds": round(train_seconds, 3),
                    **{
                        key: round(value, 6)
                        for key, value in flatten_entity_metrics(entity_metrics).items()
                    },
                }
            )
            print(
                f"[{model_name}] {split_name}: "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )

    save_metrics(results_dir / "baseline_metrics.csv", metric_rows)
    save_predictions(
        results_dir / "dev_predictions.jsonl",
        dev_texts,
        dev_labels,
        predictions["dev"],
    )
    if internal_test_features:
        save_predictions(
            results_dir / "internal_test_predictions.jsonl",
            internal_test_texts,
            internal_test_labels,
            predictions["internal_test"],
        )
    print(f"\n指标已保存：{results_dir / 'baseline_metrics.csv'}")
    print(f"预测结果已保存：{results_dir}")
    return metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="比较三个中文 NER 模型")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="最佳参数 JSON 文件，可由 src.tune_models 生成",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=2000,
        help="训练样本数，默认 2000；设置为 0 使用全部训练集",
    )
    parser.add_argument(
        "--dev-limit",
        type=int,
        default=0,
        help="验证样本数，默认使用全部验证集；设置为正数可加快调试",
    )
    parser.add_argument(
        "--internal-test-ratio",
        type=float,
        default=0.1,
        help="从官方 train 划出的内部测试比例，默认 0.1",
    )
    parser.add_argument(
        "--internal-validation-ratio",
        type=float,
        default=0.1,
        help="从官方 train 划出的调参验证比例，默认 0.1",
    )
    parser.add_argument("--seed", type=int, default=42, help="数据划分随机种子")
    parser.add_argument(
        "--use-internal-test",
        action="store_true",
        help="启用训练集 80/20 内部测试；默认使用全部训练集训练",
    )
    parser.add_argument(
        "--crf-max-iterations",
        type=int,
        default=50,
        help="CRF 最大迭代次数",
    )
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
