# Repository Guidelines

## Project Structure & Module Organization
This repository contains the AER extension on top of the upstream `verl` framework. Core AER code lives in `verl/recipe/aer/src/`: `main_ppo.py` is the training entry point, `aer_ray_trainer.py` implements reward shaping, `reward_manager.py` contains reward logic, and `similarity/` holds pluggable similarity metrics. Training scripts and local utilities live in `verl/recipe/aer/`, including `run.sh`, `tests/test_similarity.py`, and `tests/benchmark_similarity.py`. Large outputs are intentionally excluded from Git under `save/`, `verl/recipe/aer/wandb/`, and `verl/recipe/aer/outputs/`.

## Build, Test, and Development Commands
Activate the environment before any work:

```bash
conda activate aer
```

Common commands:

```bash
cd verl && pip install -e .
cd verl/recipe/aer && bash run.sh
cd verl/recipe/aer && python tests/test_similarity.py --algorithm rouge_l
cd verl/recipe/aer && python tests/benchmark_similarity.py
cd verl && pre-commit run --all-files
```

Use `run.sh` for training and resume behavior; update paths, GPUs, and hyperparameters there before launching.

## Coding Style & Naming Conventions
Python is the primary language. Follow 4-space indentation, snake_case for functions and variables, and descriptive module names matching existing patterns such as `reward_manager.py` and `semantic_embedding.py`. Linting and formatting are handled by Ruff through `verl/.pre-commit-config.yaml`; line length is currently configured to 300 in `verl/pyproject.toml`. Keep new similarity backends isolated under `verl/recipe/aer/src/similarity/`.

## Testing Guidelines
Use the local AER scripts for functional checks when changing similarity or reward code. Name new tests with the `test_*.py` pattern and keep them near the AER recipe unless they belong in upstream `verl` tests. Run targeted checks first, for example `python tests/test_similarity.py --algorithm token_match`, then run `python tests/benchmark_similarity.py` when comparing behavior or performance.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries in Chinese, for example `优化rougeL算法` or `下载embedding模型到本地`. Keep commit messages concise and specific to one change. Pull requests should state the experiment or bug being addressed, list modified configs or scripts, and include evidence for behavior changes such as test output, benchmark results, or training log snippets. When `run.sh` changes affect GPUs, model paths, or checkpoints, call that out explicitly.

## 语言
用中文回复用户，且代码注释用中文。