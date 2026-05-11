# 相似度计算算法文档

本文档介绍了 AER (Adaptive Exploration Reward) 中可用的各种相似度计算算法。

## 概述

相似度计算用于衡量同一 prompt 生成的不同 response 之间的相似程度。较低的相似度意味着更高的多样性，应该获得更高的探索奖励。

## 可用算法

### 1. Token Match (token_match)

**原理**: Token 精确匹配 + 长度归一化

**公式**:
```
similarity(i, j) = sum(token_i^k == token_j^k) / sqrt(L_i * L_j)
```

**优点**:
- 计算速度最快 (~1ms)
- 易于理解和调试
- 无需额外依赖

**缺点**:
- 只考虑精确匹配，无法捕捉语义相似性
- 对噪声敏感，单个 token 差异会导致相似度下降

**适用场景**:
- 需要 fast baseline 对比
- Token 序列质量较高，噪声较少

**参数**: 无

---

### 2. N-gram Overlap (ngram_overlap)

**原理**: 计算 token 级 n-gram 计数的 multiset Jaccard 相似度

**公式**:
```
similarity(i, j) = sum(min(count_i(g), count_j(g))) / sum(max(count_i(g), count_j(g)))
```

**优点**:
- 捕捉局部上下文相似性
- 对位置变化不敏感（基于 n-gram 计数）
- 能惩罚重复片段，避免集合去重导致相似度被高估
- 比精确匹配更鲁棒

**缺点**:
- 丢失了顺序信息
- n 值选择对结果影响较大
- 长度小于 n 的输出不产生 n-gram，相似度为 0

**适用场景**:
- 检测相同的推理步骤片段
- 对顺序变化不敏感的相似度判断

**参数**:
- `n` (int): n-gram 的窗口大小，默认为 3
  - n=1: unigram（单词级别）
  - n=2: bigram
  - n=3: trigram（推荐）
  - n=4: 4-gram

**性能**: ~5ms for batch_size=32

---

### 3. Character-level N-gram (char_ngram)

**原理**: 计算字符级 n-gram 的 Jaccard 或 Dice 相似度

**公式** (Jaccard):
```
similarity(i, j) = |char_ngram_i ∩ char_ngram_j| / |char_ngram_i ∪ char_ngram_j|
```

**公式** (Dice):
```
similarity(i, j) = 2 * |char_ngram_i ∩ char_ngram_j| / (|char_ngram_i| + |char_ngram_j|)
```

**优点**:
- 对数学表达式中的符号差异更鲁棒
- 能捕捉拼写错误或格式变化
- 对 token 切分不敏感

**缺点**:
- 需要解码文本，比 token 级别慢
- 对非常短的文本可能不够准确

**适用场景**:
- 表达式形式不同但语义相同的情况
- 需要对符号差异鲁棒的相似度判断
- 数学公式相似度计算

**参数**:
- `n` (int): 字符 n-gram 的窗口大小，默认为 4
- `metric` (str): 相似度度量，"jaccard" 或 "dice"，默认为 "jaccard"

**性能**: ~10ms for batch_size=32

---

### 4. Normalized Levenshtein (levenshtein)

**原理**: 使用归一化的 Levenshtein 编辑距离衡量文本相似度

**公式**:
```
similarity(i, j) = 1 - edit_distance(text_i, text_j) / norm_factor
```

其中 `norm_factor` 根据 `normalize_method` 参数选择：
- "max": max(len(text_i), len(text_j))（默认）
- "avg": (len(text_i) + len(text_j)) / 2
- "min": min(len(text_i), len(text_j))

**优点**:
- 衡量序列编辑成本，捕捉结构性相似
- 对拼写错误和小的修改鲁棒
- RapidFuzz 加速后计算效率高

**缺点**:
- O(n*m) 时间复杂度，对于非常长的文本较慢
- 需要解码文本，比 token 级别慢

**适用场景**:
- 检测推理路径的结构相似性
- 需要对小的编辑操作鲁棒的相似度判断

**参数**:
- `normalize_method` (str): 归一化方法，"max", "avg", 或 "min"，默认为 "max"
  - 比较前会固定做 NFC Unicode 规范化，避免组合字符形式不同导致误判。
**性能**: ~50ms for batch_size=32 (使用 RapidFuzz)

**依赖**: `rapidfuzz`

---

### 5. TF-IDF Cosine (tfidf_cosine)

**原理**: 基于 TF-IDF 向量化的余弦相似度

**公式**:
```
TF(t, d) = 词 t 在文档 d 中的频率
IDF(t) = log(总文档数 / 包含词 t 的文档数)
TF-IDF(t, d) = TF(t, d) * IDF(t)
similarity(i, j) = cosine(tfidf_i, tfidf_j)
```

**优点**:
- 考虑词的重要性，降低常见词权重
- 对长文本推理过程效果好
- 余弦相似度对向量长度不敏感

**缺点**:
- 需要解码文本
- 对于非常短的文本可能不够准确
- 需要安装 scikit-learn

**适用场景**:
- 长文本推理过程的相似度
- 需要考虑词重要性的场景

**参数**:
- `max_features` (int): TF-IDF 最大特征数，默认为 1000
- `min_df` (int): 最小文档频率，默认为 1
- `max_df` (float): 最大文档频率（比例），默认为 1.0
- `ngram_range` (tuple): n-gram 范围，默认为 (1, 2)

**性能**: ~20ms for batch_size=32

**依赖**: `scikit-learn`

---

### 6. Semantic Embedding (semantic_embedding)

**原理**: 使用预训练的句子编码器生成嵌入，计算余弦相似度

**公式**:
```
embedding_i = Encoder(text_i)
embedding_j = Encoder(text_j)
similarity(i, j) = cosine(embedding_i, embedding_j)
```

**优点**:
- 最佳的语义理解能力
- 能捕捉等价但表述不同的数学推理
- 对格式变化鲁棒
- **Qwen3-Embedding 支持 32K 长文本，适合数学推理 CoT**

**缺点**:
- 需要加载预训练模型（约 1-2GB）
- 计算相对较慢
- 需要安装 sentence-transformers

**适用场景**:
- 需要高质量的语义相似度判断
- 表达方式多样但语义相同的情况
- **长文本数学推理（推荐使用 Qwen3-Embedding）**

**参数**:
- `model_name` (str): 预训练模型名称，默认为 "Qwen/Qwen3-Embedding-0.6B"
  - **"Qwen/Qwen3-Embedding-0.6B"**: 32K 长文本，适合数学推理（推荐）
  - "Qwen/Qwen3-Embedding-4B": 更高精度，32K 长文本
  - "Qwen/Qwen3-Embedding-8B": 最高精度，32K 长文本
  - "all-MiniLM-L6-v2": 最快，但 256 token 限制
  - "all-mpnet-base-v2": 质量不错，但 512 token 限制
- `batch_size` (int): 批量编码大小，默认为 32
- `max_length` (int): 最大序列长度，默认为 4096
  - **注意**: 这是输入嵌入模型的最大长度，不是截断整个 response
  - 如果 response 超过这个长度，只会截断输入到嵌入模型的部分
- `device` (str): 计算设备，"cpu" 或 "cuda"，默认为 "cpu"
  - 推荐 CPU 以避免与训练 GPU 竞争

**性能**:
- 模型加载时间: ~5s（首次，之后使用缓存）
- 计算时间: ~500ms for batch_size=32 on CPU
- 内存占用: ~3GB（模型 + 缓存）

**依赖**: `sentence-transformers` (>= 2.7.0), `transformers` (>= 4.51.0)

**参考文献**: [Qwen3-Embedding GitHub](https://github.com/QwenLM/Qwen3-Embedding), arXiv:2506.05176

---

### 7. Compression Ratio (compression_ratio)

**原理**: 基于压缩算法（gzip/zlib）的文本相似度计算

**公式**:
```
CR(text) = compressed_size / original_size
similarity(i, j) = 1 - (CR(text_i + text_j) - min(CR(text_i), CR(text_j))) / max(CR(text_i), CR(text_j))
```

**核心思想**: 两个文本越相似，它们的拼接版本压缩后的大小越小。

**优点**:
- 计算速度极快 (~5ms)
- 内存占用低 (< 10MB)
- 与人工多样性评估相关性高
- 对长文本效果好
- 无需额外依赖

**缺点**:
- 对非常短的文本可能不够准确
- 压缩算法的内部实现影响结果

**适用场景**:
- 快速多样性评估
- 长文本推理过程
- 计算资源受限的场景

**参数**:
- `compression_type` (str): 压缩算法类型，"gzip" 或 "zlib"，默认为 "gzip"
- `normalize` (bool): 是否归一化到 [0, 1]，默认为 True

**性能**: ~5ms for batch_size=32

**参考文献**: Shaib et al., "Standardizing the Measurement of Text Diversity: A Tool and Comparative Analysis", arXiv:2403.00553

---

### 8. ROUGE-L (rouge_l)

**原理**: 基于最长公共子序列（LCS）的相似度算法

**公式**:
```
LCS(i, j) = LongestCommonSubsequence(text_i, text_j)
R = LCS / len(text_i)  # 召回率
P = LCS / len(text_j)  # 精确率
F = (1 + beta²) × R × P / (R + beta² × P)  # F-beta 分数
```

**核心思想**: LCS 保留顺序信息，对结构性相似敏感。

**优点**:
- 对顺序敏感（与 NgramOverlap 不同）
- 捕捉结构性相似
- 对数学表达式比较有效
- 无需额外依赖

**缺点**:
- O(n×m) 时间复杂度
- 需要解码文本
- 分词方式影响结果

**适用场景**:
- 需要考虑顺序的相似度判断
- 数学推理步骤比较
- 结构性文本分析

**参数**:
- `use_char_level` (bool): 是否使用字符级别，默认为 False
  - False: 使用空格分词（默认）
  - True: 使用字符级别（对数学符号更鲁棒）
- `beta` (float): F-beta 分数的 beta 参数，默认为 1.0
  - beta = 1: F1 分数（R 和 P 等权重）
  - beta > 1: 召回率更重要
  - beta < 1: 精确率更重要
- `tokenize_by_space` (bool): 是否按空格分词，默认为 True

**性能**: ~30ms for batch_size=32

**参考文献**: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", 2004

---

### 9. SimHash (simhash)

**原理**: 将 token n-gram 特征压缩为固定长度 SimHash 指纹，再用 Hamming 距离估计近重复程度

**公式**:
```
raw_similarity(i, j) = 1 - hamming(simhash_i, simhash_j) / hash_bits
similarity(i, j) = max(0, 2 * raw_similarity(i, j) - 1)
```

**核心思想**: 共享大量局部片段的 response 会得到相近指纹；随机无关文本的指纹期望 Hamming 相似度约为 0.5，默认会校准到 0，避免探索奖励把无关文本误判为中等相似。

**优点**:
- 只保存固定长度整数指纹，内存占用极低
- 两两比较只需 XOR 和 bit_count，适合 rollout 数较多或长 CoT
- 无需模型和额外依赖

**缺点**:
- 是近似算法，不如精确 n-gram Jaccard 可解释
- 对真正的语义等价无法建模

**适用场景**:
- 快速近重复检测
- 长文本数学推理的低成本多样性估计
- 希望比 `ngram_overlap` 更省内存，同时比 `token_match` 更鲁棒

**参数**:
- `n` (int): token n-gram 窗口大小，默认为 3
- `hash_bits` (int): 指纹位数，默认为 64
- `use_counts` (bool): 是否保留重复 n-gram 权重，默认为 True
- `calibrate_random` (bool): 是否将随机指纹期望相似度校准为 0，默认为 True

**性能**: ~2-5ms for batch_size=32，长文本下通常比精确集合两两 Jaccard 更稳定

---

## 算法对比

| 算法 | 内存占用 | 计算时间 | OOM 风险 | GPU 使用 | 语义理解 | 顺序敏感 |
|------|---------|---------|---------|---------|---------|---------|
| token_match | < 10MB | ~1ms | 无 | 无 | 无 | 是 |
| ngram_overlap | ~50MB | ~5ms | 无 | 无 | 低 | 否 |
| char_ngram | ~100MB | ~10ms | 无 | 无 | 低 | 否 |
| levenshtein | ~200MB | ~50ms | 低 | 无 | 中 | 是 |
| tfidf_cosine | ~500MB | ~20ms | 低 | 无 | 中 | 否 |
| semantic_embedding | ~3GB | ~500ms | 低 | 是(可选) | 高 | 否 |
| **compression_ratio** | **< 10MB** | **~5ms** | **无** | **无** | **中** | **是** |
| **rouge_l** | **~50MB** | **~30ms** | **无** | **无** | **中** | **是** |
| **simhash** | **< 10MB** | **~2-5ms** | **无** | **无** | **低** | **否** |

## 使用示例

### 在 run.sh 中配置

```bash
# 使用 Token 精确匹配（最快）
similarity_algorithm="token_match"

# 使用 N-gram 重叠度（平衡）
similarity_algorithm="ngram_overlap"
similarity_n=3

# 使用压缩比相似度（快速多样性评估）
similarity_algorithm="compression_ratio"
similarity_compression_type="gzip"  # 可选: gzip 或 zlib

# 使用 ROUGE-L（顺序敏感）
similarity_algorithm="rouge_l"
similarity_beta=1.0  # F-beta 分数的 beta 参数

# 使用 SimHash（快速近重复检测）
similarity_algorithm="simhash"
similarity_n=3

# 使用语义嵌入（最佳质量，支持长文本）
similarity_algorithm="semantic_embedding"
similarity_model="Qwen/Qwen3-Embedding-0.6B"
similarity_device="cpu"
```

### 在 Python 代码中使用

```python
from recipe.aer.src.similarity import get_similarity_computer

# 创建相似度计算器
computer = get_similarity_computer("ngram_overlap", n=3)

# 计算相似度
similarity_matrix = computer.compute(data, tokenizer=tokenizer)

# 使用压缩比相似度
computer = get_similarity_computer("compression_ratio", compression_type="gzip")

# 使用 ROUGE-L 相似度
computer = get_similarity_computer("rouge_l", beta=2.0)

# 使用 SimHash 相似度
computer = get_similarity_computer("simhash", n=3, hash_bits=64)
```

## 依赖安装

```bash
# 基础算法（token_match, ngram_overlap, char_ngram, simhash）
# 无需额外依赖

# 编辑距离算法（推荐）
pip install rapidfuzz

# TF-IDF 余弦相似度
pip install scikit-learn

# 语义嵌入相似度
pip install sentence-transformers
```

## 选择建议

| 场景 | 推荐算法 |
|------|---------|
| 快速实验/调试 | token_match |
| 平衡性能和质量 | ngram_overlap (n=3) |
| 数学表达式相似度 | char_ngram (n=4) |
| 结构相似性 | levenshtein 或 rouge_l |
| 长文本推理 | tfidf_cosine |
| 最高语义质量 | semantic_embedding (Qwen3-Embedding-0.6B) |
| **快速多样性评估** | **compression_ratio** |
| **顺序敏感的结构相似** | **rouge_l** |
| **长文本近重复快速检测** | **simhash** |
