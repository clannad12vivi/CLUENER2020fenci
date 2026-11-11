"""中文 NER 的字符级特征提取与 BIO 数据读取。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import jieba.posseg as pseg


_ASCII_LETTER = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")
_CHINESE = re.compile(r"[\u4e00-\u9fff]")


def _char_type(character: str) -> str:
    if not character:
        return "BOUNDARY"
    if _CHINESE.fullmatch(character):
        return "CHINESE"
    if _DIGIT.fullmatch(character):
        return "DIGIT"
    if _ASCII_LETTER.fullmatch(character):
        return "LETTER"
    if character.isspace():
        return "SPACE"
    return "PUNCT"


def _word_features(text: str) -> list[tuple[str, str, str]]:
    """为每个字符返回 (词语、词性、词内位置)。"""
    features = [("", "x", "SINGLE") for _ in text]
    offset = 0
    for word, part_of_speech in pseg.cut(text):
        if not word:
            continue
        end = offset + len(word)
        if end > len(text):
            raise ValueError("jieba 分词结果超出文本长度")
        for index in range(offset, end):
            if len(word) == 1:
                position = "SINGLE"
            elif index == offset:
                position = "BEGIN"
            elif index == end - 1:
                position = "END"
            else:
                position = "MIDDLE"
            features[index] = (word, part_of_speech or "x", position)
        offset = end
    if offset != len(text):
        raise ValueError("jieba 分词结果未覆盖完整文本")
    return features


def sentence_to_features(text: str) -> list[dict[str, str | bool]]:
    """将一条文本转换为 sklearn-crfsuite 使用的特征序列。"""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not text:
        return []

    word_features = _word_features(text)
    result: list[dict[str, str | bool]] = []
    for index, character in enumerate(text):
        previous = text[index - 1] if index > 0 else ""
        next_character = text[index + 1] if index + 1 < len(text) else ""
        word, part_of_speech, word_position = word_features[index]
        item: dict[str, str | bool] = {
            "bias": "1.0",
            "char": character,
            "char_type": _char_type(character),
            "prev_char": previous or "BOS",
            "next_char": next_character or "EOS",
            "prev_char_type": _char_type(previous),
            "next_char_type": _char_type(next_character),
            "is_first": index == 0,
            "is_last": index == len(text) - 1,
            "word": word,
            "word_pos": word_position,
            "part_of_speech": part_of_speech,
            "char_pos": f"{character}_{part_of_speech}",
            "char_word_pos": f"{character}_{word_position}",
            "word_part_of_speech": f"{word}_{part_of_speech}",
        }
        if index > 1:
            item["prev2_char"] = text[index - 2]
        else:
            item["prev2_char"] = "BOS"
        if index + 2 < len(text):
            item["next2_char"] = text[index + 2]
        else:
            item["next2_char"] = "EOS"
        if index > 0:
            item["bigram_prev"] = previous + character
        if index + 1 < len(text):
            item["bigram_next"] = character + next_character
        result.append(item)
    return result


def read_bio(path: str | Path) -> list[tuple[list[str], list[str]]]:
    """读取字符 TAB 标签格式的 BIO 文件。"""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到 BIO 文件：{input_path}")

    sentences: list[tuple[list[str], list[str]]] = []
    characters: list[str] = []
    labels: list[str] = []
    with input_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                if characters:
                    sentences.append((characters, labels))
                    characters, labels = [], []
                continue
            parts = stripped.split("\t")
            if len(parts) != 2 or not parts[0]:
                raise ValueError(f"{input_path} 第 {line_number} 行不是“字符<TAB>标签”格式")
            characters.append(parts[0])
            labels.append(parts[1])
    if characters:
        sentences.append((characters, labels))
    return sentences


def bio_to_features(
    sentences: Iterable[tuple[list[str], list[str]]],
) -> tuple[list[list[dict[str, str | bool]]], list[list[str]]]:
    """将 BIO 样本转换为 CRF 的 X、y。"""
    feature_sequences: list[list[dict[str, str | bool]]] = []
    label_sequences: list[list[str]] = []
    for characters, labels in sentences:
        text = "".join(characters)
        if len(text) != len(labels):
            raise ValueError("字符数量和标签数量不一致")
        feature_sequences.append(sentence_to_features(text))
        label_sequences.append(labels)
    return feature_sequences, label_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 BIO 文件的特征提取结果")
    parser.add_argument("input", type=Path, help="BIO 文件，例如 data\\train.bio")
    parser.add_argument("--limit", type=int, default=1, help="显示的样本数量")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit 必须大于 0")
    sentences = read_bio(args.input)
    features, labels = bio_to_features(sentences[: args.limit])
    print(f"样本数：{len(sentences)}")
    for index, (items, targets) in enumerate(zip(features, labels), start=1):
        print(f"第 {index} 条：{len(items)} 个字符")
        for item, target in zip(items, targets):
            print(item, target)


if __name__ == "__main__":
    main()
