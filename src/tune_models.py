"""使用内部测试集为三个 NER 模型选择参数。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .crf_model import CRFNER
from .hmm_model import HMMNER
from .model_common import calculate_metrics, load_dataset, save_metrics
from .nb_model import NaiveBayesNER
from .train_compare import _limit_value, _split_three_way


def _grid_results(
    model_name: str,
    candidates: list[dict[str, Any]],
    train_features: list[list[dict[str, str | bool]]],
    train_labels: list[list[str]],
    eval_features: list[list[dict[str, str | bool]]],
    eval_labels: list[list[str]],
    crf_max_iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_f1 = -1.0
    for config in candidates:
        if model_name == "CRF":
            model = CRFNER(
                c1=config["c1"],
                c2=config["c2"],
                max_iterations=config.get("max_iterations", crf_max_iterations),
            )
        elif model_name == "HMM":
            model = HMMNER(
                alpha=config["alpha"],
                observation_mode=config.get("observation_mode", "char"),
            )
        else:
            model = NaiveBayesNER(alpha=config["alpha"])
        start = time.perf_counter()
        model.fit(train_features, train_labels)
        predictions = model.predict(eval_features)
        metrics = calculate_metrics(eval_labels, predictions)
        row = {
            "model": model_name,
            **config,
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "seconds": round(time.perf_counter() - start, 3),
        }
        rows.append(row)
        print(
            f"[{model_name}] {config} -> "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = config
    if best_config is None:
        raise RuntimeError(f"{model_name} 没有得到可用参数")
    return best_config, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="为三个 NER 模型选择最佳参数")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--params-file", type=Path, default=Path("results/best_params.json"))
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--internal-test-ratio", type=float, default=0.1)
    parser.add_argument("--internal-validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crf-max-iterations", type=int, default=200)
    args = parser.parse_args()

    features, labels, texts = load_dataset(
        args.data_dir / "train.bio", _limit_value(args.train_limit)
    )
    (train_features, train_labels, _), (eval_features, eval_labels, _), _ = _split_three_way(
        features,
        labels,
        texts,
        args.internal_validation_ratio,
        args.internal_test_ratio,
        args.seed,
    )
    grids = {
        "CRF": [
            {"c1": c1, "c2": c2, "max_iterations": max_iterations}
            for c1 in (0.01, 0.1)
            for c2 in (0.01, 0.1)
            for max_iterations in (50, 100, 200)
        ],
        "HMM": [
            {"alpha": alpha, "observation_mode": "char"}
            for alpha in (0.01, 0.1, 1.0)
        ],
        "NaiveBayes": [
            {"alpha": alpha, "fit_prior": fit_prior}
            for alpha in (0.1, 1.0, 10.0)
            for fit_prior in (True, False)
        ],
    }
    best_params: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for model_name, candidates in grids.items():
        best, rows = _grid_results(
            model_name,
            candidates,
            train_features,
            train_labels,
            eval_features,
            eval_labels,
            args.crf_max_iterations,
        )
        best_params[model_name] = dict(best)
        all_rows.extend(rows)
        print(f"[{model_name}] 最佳参数：{best_params[model_name]}")

    args.params_file.parent.mkdir(parents=True, exist_ok=True)
    args.params_file.write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_metrics(args.results_dir / "tuning_metrics.csv", all_rows)
    print(f"\n最佳参数已保存：{args.params_file}")
    print(f"调参结果已保存：{args.results_dir / 'tuning_metrics.csv'}")


if __name__ == "__main__":
    main()
