# 数据集来源记录

## 选用数据集

本项目选用 **CLUENER2020 中文细粒度命名实体识别数据集**。它与课程设计目标匹配，原因是：

- 面向中文命名实体识别任务，原始数据采用 JSONL 格式；
- 每条样本包含文本和实体区间标注，便于转换为字符级 BIO 标注；
- 官方提供训练集、验证集和测试文本，适合进行模型训练、调参和结果展示；
- 数据集包含人名、地址、公司、政府、组织机构等类别，可以映射为本项目需要的 `PER`、`LOC` 和 `ORG`。

## 官方来源

- 官方 GitHub 仓库：<https://github.com/CLUEbenchmark/CLUENER2020>
- 官方数据压缩包：<https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip>
- 数据集技术报告：<https://arxiv.org/abs/2001.04351>
- 数据集名称：CLUENER2020: Fine-grained Name Entity Recognition for Chinese
- 作者：Xu Liang, Dong Qianqian, Yu Cong, Tian Yin, Liu Weitang, Li Lu, Zhang Xuanwei
- 发布年份：2020
- 本项目记录日期：2026-11-03

## 本地文件与规模

数据已下载到项目的 `data` 目录：

| 文件 | 样本数 | 标签情况 | 用途 |
|---|---:|---|---|
| `train.json` | 10748 | 有实体标注 | 按80%/10%/10%划分训练、调参验证和内部测试 |
| `dev.json` | 1343 | 有实体标注 | 最终独立评估 |
| `test.json` | 1345 | 官方测试文本无公开实体标注 | 仅用于无标签预测展示 |

项目同时保存了对应的 `train.bio`、`dev.bio` 和 `test.bio` 转换文件，以及原始压缩包 `cluener_public.zip`。

## 本项目的标签映射

| CLUENER2020 原始类别 | 项目类别 |
|---|---|
| `address` | `LOC` |
| `book` | `BOOK` |
| `company` | `COMP` |
| `game` | `GAME` |
| `government` | `GOV` |
| `movie` | `MOV` |
| `name` | `PER` |
| `organization` | `ORG` |
| `position` | `POS` |
| `scene` | `SCENE` |

官方仓库说明了数据字段、类别分布和数据规模；本项目使用全部10类，不再把多个原始类别合并。由于普通BIO不能表示嵌套实体，预处理时保留较长的外层实体；部分重叠标注会报错。项目没有将无标签的 `test.json` 当作有监督测试集，也没有据此计算虚假的 Precision、Recall 或 F1。
