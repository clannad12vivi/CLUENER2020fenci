# 基于 CRF 的中文命名实体识别系统

这是一个使用 Python、字符级特征和条件随机场（CRF）实现的中文命名实体识别课程设计项目。

当前使用CLUENER2020原始10类实体：`address`、`book`、`company`、`game`、`government`、`movie`、`name`、`organization`、`position`、`scene`，程序内部对应为`LOC`、`BOOK`、`COMP`、`GAME`、`GOV`、`MOV`、`PER`、`ORG`、`POS`、`SCENE`。

## 项目结构

```text
data/       原始数据集和转换后的标注数据
models/     训练后的模型
src/        数据处理、特征工程、训练、评估和预测代码
app.py      Streamlit 交互界面
```

## 环境准备

建议使用 Python 3.10 或更高版本，并创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 实现顺序

1. 准备 CLUENER2020 数据集并验证 JSON 格式。
2. 将 JSON 实体标注转换为字符级 BIO 标注。
3. 提取 CRF 特征并训练模型。
4. 在有标签的验证集上评估 Precision、Recall、F1。
5. 启动 Streamlit 页面进行交互式实体识别。

注意：官方 `test.json` 不公开实体标签，因此只用于无标签预测展示，不能用于计算真实的监督评价指标。

## 数据集来源与数据预处理

本项目选用 CLUENER2020 中文细粒度命名实体识别数据集。数据集来源、官方仓库、直接下载地址、技术报告、文件规模和标签映射已记录在
[`DATASET_SOURCES.md`](DATASET_SOURCES.md) 中。

从 [CLUENER2020 官方仓库](https://github.com/CLUEbenchmark/CLUENER2020)
下载 `train.json`、`dev.json` 和 `test.json`，放入 `data/` 目录。
这些文件是 JSONL 格式，每行一条样本。

在项目根目录执行：

```powershell
python -m src.data_process data\train.json data\train.bio
python -m src.data_process data\dev.json data\dev.bio
python -m src.data_process data\test.json data\test.bio
```

转换程序保留CLUENER2020全部10类标签。由于普通BIO序列不能表示嵌套实体，转换时保留较长外层实体；部分重叠标注会明确报错，避免静默生成错误标签。

## 特征工程

`src\feature_extraction.py` 实现了训练和预测共用的字符级特征，包括：

- 当前字符、前后字符和前后两位字符；
- 字符类型（中文、数字、英文、空格、标点）；
- 相邻字符组合；
- jieba 分词结果、词性和字符在词中的位置；
- 文本首尾标记。

每个字符最终表示为一个特征字典，CRF 会基于这些特征学习 BIO 标签。
例如，训练时调用 `read_bio()` 读取样本，再调用 `bio_to_features()` 得到
CRF 所需的 `X` 和 `y`。

## 三模型快速训练与比较

当前实现了 CRF、HMM 和朴素贝叶斯三个模型。由于CLUENER官方
`test.json` 不提供实体标签，无法计算真实测试集 F1，因此程序会将有标签的
最终运行使用官方`train.json`内部的90%数据训练，在剩余10%内部测试集上检查泛化，并使用有标签的`dev.json`作为最终独立评估集。
需要快速调试或启用内部测试时可以显式传参：

```powershell
python -m src.train_compare
```

快速调试并启用内部测试：

```powershell
python -m src.train_compare --train-limit 2000 --use-internal-test
```

完整实验划分：

```powershell
python -m src.tune_models --train-limit 0 --internal-validation-ratio 0.1 --internal-test-ratio 0.1
python -m src.train_compare --train-limit 0 --use-internal-test --internal-validation-ratio 0.1 --internal-test-ratio 0.1 --params-file results\best_params.json
```

先进行小范围参数选择：

```powershell
python -m src.tune_models
python -m src.train_compare --train-limit 0 --params-file results\best_params.json
```

启动交互界面：

```powershell
streamlit run app.py
```

页面支持在 CRF、HMM 和朴素贝叶斯三个模型之间切换，输入中文文本后显示
识别出的实体、实体类别和字符位置，并展示 `dev` 集上的模型对比指标。

设置 `--train-limit 0` 可以使用完整训练集：

```powershell
python -m src.train_compare --train-limit 0
```

运行后会生成：

```text
models\crf_model.pkl
models\hmm_model.pkl
models\naivebayes_model.pkl
results\baseline_metrics.csv
results\dev_predictions.jsonl
results\internal_test_predictions.jsonl
results\best_params.json
results\tuning_metrics.csv
```
