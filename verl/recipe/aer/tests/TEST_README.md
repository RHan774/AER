# 相似度计算算法测试脚本使用说明

## 概述

本目录包含两个测试脚本，用于测试 AER 中实现的各种相似度计算算法：

1. **test_similarity.py** - 单算法测试脚本
2. **benchmark_similarity.py** - 多算法对比测试脚本

## 环境准备

在运行测试前，请确保已激活 conda 环境：

```bash
conda activate aer
cd /home/ruanruihan/adaptive-exploration-reward/verl/recipe/aer/tests
```

## 可用算法

| 算法 | 说明 | 计算时间 | 语义理解 | 顺序敏感 |
|------|------|---------|---------|---------|
| token_match | Token 精确匹配 | ~1ms | 无 | 是 |
| ngram_overlap | N-gram 重叠度 | ~5ms | 低 | 否 |
| char_ngram | 字符级 N-gram | ~10ms | 低 | 否 |
| levenshtein | 编辑距离 | ~50ms | 中 | 是 |
| tfidf_cosine | TF-IDF 余弦相似度 | ~20ms | 中 | 否 |
| semantic_embedding | 语义嵌入相似度 | ~500ms | 高 | 否 |
| **compression_ratio** | **压缩比相似度** | **~5ms** | **中** | **是** |
| **rouge_l** | **ROUGE-L 相似度** | **~30ms** | **中** | **是** |

## test_similarity.py 使用方法

### 基本用法

```bash
# 测试 token_match 算法
python test_similarity.py --algorithm token_match

# 测试 ngram_overlap 算法（默认 n=3）
python test_similarity.py --algorithm ngram_overlap

# 测试 ngram_overlap 算法（自定义 n 值）
python test_similarity.py --algorithm ngram_overlap --n 4

# 测试 char_ngram 算法
python test_similarity.py --algorithm char_ngram --n 4

# 测试 levenshtein 算法
python test_similarity.py --algorithm levenshtein

# 测试 tfidf_cosine 算法
python test_similarity.py --algorithm tfidf_cosine
```

### 输出结果

脚本会输出以下内容：

1. **相似度矩阵**：展示所有 16 条输出两两之间的相似度
2. **探索奖励**：每条输出的探索奖励计算结果
3. **统计信息**：平均探索奖励、最小值、最大值、标准差

### 示例输出

```
================================================================================
相似度矩阵 (batch_size=16)
================================================================================
        Out01  Out02  Out03  ...
Out01 1.000 0.429 0.426  ...
Out02 0.429 1.000 0.414  ...
...

================================================================================
探索奖励 (Exploration Rewards)
================================================================================
输出         相似度之和           探索奖励           
----------------------------------------
Out 01   7.292590        0.137125       
Out 02   7.843553        0.127493       
...

================================================================================
统计信息
================================================================================
平均探索奖励: 0.129320
最小探索奖励: 0.124197
最大探索奖励: 0.137125
探索奖励标准差: 0.003611
```

### 保存结果

使用 `--output` 参数保存结果到 JSON 文件：

```bash
python test_similarity.py --algorithm token_match --output results.json
```

### 列出所有可用算法

```bash
python test_similarity.py --list-algorithms
```

## benchmark_similarity.py 使用方法

该脚本会自动运行所有可用算法并输出对比结果。

```bash
python benchmark_similarity.py
```

### 输出结果

1. **算法对比表格**：各算法的平均探索奖励、标准差、计算时间
2. **相对性能分析**：以 token_match 为基准的相对性能

### 示例输出

```
算法                   平均探索奖励          标准差          计算时间(s)
-------------------------------------------------------------------------------------
token_match          0.129320        0.003611     0.024
ngram_overlap        0.250384        0.012006     0.042
char_ngram           0.125719        0.002745     0.097
levenshtein          0.100229        0.002419     0.049
tfidf_cosine         0.070006        0.001080     0.014

相对性能分析 (以 token_match 为基准)
算法                   相对探索奖励               相对计算时间
--------------------------------------------------
token_match            +0.0%                +0.0%
ngram_overlap         +93.6%               +74.4%
char_ngram             -2.8%              +300.0%
levenshtein           -22.5%              +101.4%
tfidf_cosine          -45.9%               -42.7%
```

## 算法选择建议

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 快速实验/调试 | token_match | 最快，baseline |
| 平衡性能和质量 | ngram_overlap | 探索奖励更高，计算快 |
| 数学表达式相似度 | char_ngram | 对符号差异鲁棒 |
| 结构相似性 | levenshtein 或 rouge_l | 捕捉编辑操作或顺序相似性 |
| 长文本推理 | tfidf_cosine | 考虑词重要性 |
| 最高语义质量 | semantic_embedding | 最佳语义理解 |
| **快速多样性评估** | **compression_ratio** | **论文证明与 n-gram 指标高度相关，速度快 10-100 倍** |
| **顺序敏感的结构相似** | **rouge_l** | **基于最长公共子序列，对推理步骤顺序敏感** |

## 新增算法详细说明

### Compression Ratio (compression_ratio)

**原理**：基于 arXiv:2403.00553 "Standardizing the Measurement of Text Diversity" 论文

使用 gzip/zlib 压缩算法计算文本相似度。核心思想是：两个文本越相似，它们的拼接版本压缩后的大小越小。

**公式**：
```
CR(text) = compressed_size / original_size
similarity(i, j) = 1 - (CR_combined - min(CR_i, CR_j)) / max(CR_i, CR_j)
```

**参数**：
- `compression_type`: 压缩算法类型 ("gzip" 或 "zlib"，默认 "gzip")
- `normalize`: 是否归一化到 [0, 1] (默认 True)

**性能**：
- 计算时间: ~5ms for batch_size=32
- 内存占用: < 10MB
- 无需额外依赖

**使用示例**：
```bash
# 使用默认 gzip 压缩
python test_similarity.py --algorithm compression_ratio

# 使用 zlib 压缩
python test_similarity.py --algorithm compression_ratio --compression_type zlib
```

---

### ROUGE-L (rouge_l)

**原理**：基于最长公共子序列（LCS）的相似度算法

保留顺序信息，对结构性相似敏感。与 NgramOverlap 不同，它考虑了序列的顺序。

**公式**：
```
LCS(i, j) = LongestCommonSubsequence(text_i, text_j)
R = LCS / len(text_i)  # 召回率
P = LCS / len(text_j)  # 精确率
F = (1 + beta²) × R × P / (R + beta² × P)  # F-beta 分数
```

**参数**：
- `use_char_level`: 是否使用字符级别 (默认 False，使用词级别)
- `beta`: F-beta 分数的 beta 参数 (默认 1.0，即 F1 分数)
  - beta > 1: 召回率更重要
  - beta < 1: 精确率更重要
- `tokenize_by_space`: 是否按空格分词 (默认 True)

**性能**：
- 计算时间: ~30ms for batch_size=32
- 内存占用: ~50MB
- 无需额外依赖

**使用示例**：
```bash
# 使用默认参数（词级别，F1 分数）
python test_similarity.py --algorithm rouge_l

# 使用字符级别（对数学符号更鲁棒）
python test_similarity.py --algorithm rouge_l --use_char_level

# 重视召回率（F2 分数）
python test_similarity.py --algorithm rouge_l --beta 2.0
```

## 依赖说明

- **token_match, ngram_overlap, char_ngram**: 无额外依赖
- **levenshtein**: 推荐 `rapidfuzz`（自动回退到内置实现）
- **tfidf_cosine**: 需要 `scikit-learn`
- **semantic_embedding**: 需要 `sentence-transformers`

## 常见问题

### 1. 如何选择合适的算法？

- 如果需要快速 baseline：使用 `token_match`
- 如果需要更高的探索奖励（更多样性）：使用 `ngram_overlap`
- 如果需要语义级别的相似度：使用 `semantic_embedding`

### 2. 探索奖励越高越好吗？

探索奖励的计算公式是 `1 / 相似度之和`，因此：
- 相似度越高 → 探索奖励越低（多样性低）
- 相似度越低 → 探索奖励越高（多样性高）

在 AER 训练中，探索奖励用于鼓励模型生成更多样化的响应。

### 3. 不同算法的探索奖励可以比较吗？

不同算法计算相似度的尺度不同，因此探索奖励的绝对值不能直接比较。但同一算法在不同轮次的结果可以比较。

## 测试数据说明

当前测试使用的数据文件是 `rollout_example.jsonl`，包含：
- 16 条针对同一数学问题的 rollout 输出
- 每条输出都是完整的推理过程
- 所有输出都正确回答了问题（acc=1.0）

这模拟了实际训练中的情况：batch_size=1, rollout_n=16。
