"""中文命名实体识别 Streamlit 交互界面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.crf_model import CRFNER
from src.feature_extraction import sentence_to_features
from src.hmm_model import HMMNER
from src.nb_model import NaiveBayesNER


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "CRF": ROOT / "models" / "crf_model.pkl",
    "HMM": ROOT / "models" / "hmm_model.pkl",
    "NaiveBayes": ROOT / "models" / "naivebayes_model.pkl",
}
LABEL_NAMES = {
    "PER": "人名",
    "LOC": "地址",
    "COMP": "公司",
    "GOV": "政府",
    "ORG": "组织机构",
    "BOOK": "书名",
    "GAME": "游戏",
    "MOV": "电影",
    "POS": "职位",
    "SCENE": "景点",
}
LABEL_COLORS = {
    "PER": "#ffd6d6",
    "LOC": "#d6e8ff",
    "COMP": "#d9f5d6",
    "GOV": "#fff0c2",
    "ORG": "#e6d6ff",
    "BOOK": "#f8d6ff",
    "GAME": "#d6fff5",
    "MOV": "#ffe0cc",
    "POS": "#e0e0e0",
    "SCENE": "#d6f0ff",
}


@st.cache_resource
def load_model(model_name: str) -> Any:
    path = MODEL_PATHS[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{path}")
    model_types = {"CRF": CRFNER, "HMM": HMMNER, "NaiveBayes": NaiveBayesNER}
    return model_types[model_name].load(path)


def decode_entities(text: str, labels: list[str]) -> list[dict[str, str | int]]:
    """将字符级 BIO 标签合并为实体。"""
    entities: list[dict[str, str | int]] = []
    start: int | None = None
    entity_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, entity_type
        if start is not None and entity_type is not None:
            entities.append(
                {
                    "text": text[start:end],
                    "type": entity_type,
                    "start": start,
                    "end": end,
                }
            )
        start, entity_type = None, None

    for index, label in enumerate(labels):
        if label == "O" or "-" not in label:
            close(index)
            continue
        prefix, current_type = label.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            close(index)
            start, entity_type = index, current_type
        elif prefix != "I":
            close(index)
            start, entity_type = index, current_type
    close(len(text))
    return entities


def render_entities(text: str, entities: list[dict[str, str | int]]) -> str:
    """生成带颜色标记的 HTML 实体结果。"""
    parts: list[str] = []
    cursor = 0
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        entity_type = str(entity["type"])
        parts.append(text[cursor:start])
        color = LABEL_COLORS.get(entity_type, "#eeeeee")
        label_name = LABEL_NAMES.get(entity_type, entity_type)
        parts.append(
            f'<mark style="background-color:{color}; padding:2px 4px; '
            f'border-radius:4px">{text[start:end]} '
            f'<small>[{label_name}]</small></mark>'
        )
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def predict(model_name: str, text: str) -> tuple[list[str], list[dict[str, str | int]]]:
    model = load_model(model_name)
    features = [sentence_to_features(text)]
    labels = model.predict(features)[0]
    entities = decode_entities(text, list(labels))
    return list(labels), entities


def main() -> None:
    st.set_page_config(page_title="中文命名实体识别系统", page_icon="🔎", layout="wide")
    st.title("中文命名实体识别系统")
    st.caption("基于 CRF、HMM 和朴素贝叶斯的中文实体识别实验平台")

    with st.sidebar:
        st.header("模型设置")
        model_name = st.selectbox("选择识别模型", list(MODEL_PATHS))
        st.markdown("实体类型：CLUENER2020原始10类")

    default_text = "北京大学校长张三在中国石油大学参加会议。"
    text = st.text_area("请输入待识别的中文文本", value=default_text, height=140)

    if st.button("开始识别", type="primary"):
        if not text.strip():
            st.warning("请输入文本后再识别。")
        else:
            try:
                _, entities = predict(model_name, text)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.subheader("识别结果")
                st.markdown(render_entities(text, entities), unsafe_allow_html=True)
                if entities:
                    table = pd.DataFrame(
                        [
                            {
                                "实体": item["text"],
                                "类别": LABEL_NAMES.get(str(item["type"]), item["type"]),
                                "起始位置": item["start"],
                                "结束位置": int(item["end"]) - 1,
                            }
                            for item in entities
                        ]
                    )
                    st.dataframe(table, use_container_width=True, hide_index=True)
                else:
                    st.info("未识别到目标实体。")

    metrics_path = ROOT / "results" / "baseline_metrics.csv"
    if metrics_path.is_file():
        st.subheader("模型实验指标")
        metrics = pd.read_csv(metrics_path)
        dev_metrics = metrics[metrics["split"] == "dev"].copy()
        columns = ["model", "precision", "recall", "f1"]
        columns.extend(
            sorted(
                column
                for column in dev_metrics.columns
                if column.endswith("_f1") and column not in columns
            )
        )
        st.dataframe(
            dev_metrics[[column for column in columns if column in dev_metrics.columns]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
