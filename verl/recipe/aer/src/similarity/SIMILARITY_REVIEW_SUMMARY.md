# 相似度模块审查总结

## 范围
本文档总结了对 `verl/recipe/aer/src/similarity` 目录下相似度算法实现所完成的代码审查与优化工作。

## 主要修改

### 1. 保留 `token_match` 作为基准算法
- 保持 `token_match` 的核心相似度计算逻辑不变，继续作为 AER 的基准实现。
- 其他算法仅替换“组内两两 response 的相似度定义”，后续探索奖励聚合流程与基准路径保持一致。

### 2. 统一按 Group 计算的公共逻辑
- 重构了 `src/similarity/base.py` 中的共享基础层。
- 新增可复用辅助方法，用于：
  - 提取有效 response token；
  - 对每个样本只解码一次 response；
  - 仅在相同 `uid` 的组内计算相似度。
- 避免先做整批两两计算、再通过 group mask 清零的无效开销。

### 3. 修正各算法定义与实现
- `ngram_overlap.py`：实现 token n-gram 计数型 multiset Jaccard 相似度；长度小于 n 的序列不产生 n-gram。
- `char_ngram.py`：实现字符级 n-gram 相似度，支持并校验 `jaccard` / `dice` 两种模式。
- `levenshtein.py`：将原先的模糊匹配近似改为严格的归一化 Levenshtein 相似度；仅保留 `rapidfuzz` 加速实现，并在比较前做 NFC Unicode 规范化。
- `tfidf_cosine.py`：对齐标准 TF-IDF + 余弦相似度定义；仅保留 `scikit-learn` 标准实现。
- `compression_ratio.py`：将原启发式实现改为基于归一化压缩距离思想的对称压缩相似度实现。
- `rouge_l.py`：对齐基于 LCS 的 ROUGE-L F 分数定义，并补充 `beta` 参数校验。
- `semantic_embedding.py`：改进参数校验、模型缓存、向量归一化、重复文本去重编码和组内矩阵计算；移除共享模型线程并发路径。

### 4. 改进结构与鲁棒性
- 统一了参数校验和错误提示。
- 移除了相似度模块中的运行时回退分支，相关依赖改为显式要求。
- `semantic_embedding` 直接作为正式算法注册，运行环境要求已安装 `sentence-transformers`。

### 5. 增加单元测试
- 新增 `verl/recipe/aer/tests/test_similarity_algorithms.py`。
- 覆盖内容包括：
  - 各本地算法的数学正确性；
  - 组内计算、组间清零的行为；
  - 正式算法的工厂构造行为。

## 验证结果
- 已通过：
  - `python -m unittest verl/recipe/aer/tests/test_similarity_algorithms.py`
  - `python -m ruff check verl/recipe/aer/src/similarity verl/recipe/aer/tests/test_similarity_algorithms.py`

## 可量化性能提升
基于本地小规模基准测试，按 Group 分块计算带来了明显加速：

- `ngram_overlap`：约 `6.90x`
- `char_ngram`：约 `5.16x`
- `rouge_l`：约 `10.82x`

## 说明
- 当前代码假设训练环境已安装 `rapidfuzz`、`scikit-learn` 与 `sentence-transformers`。
- `semantic_embedding` 的工厂路径已纳入测试，真实模型推理仍建议结合本地模型缓存或训练环境做专项验证。
