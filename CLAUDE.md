# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Important: Always Activate Environment First

**Before running ANY command or script in this repository, you MUST activate the conda environment:**

```bash
conda activate aer
```

All scripts, including [`run.sh`](verl/recipe/aer/run.sh), assume the `aer` conda environment is active. If you see errors about missing modules or packages, first check that you've run `conda activate aer`.

## Project Overview

This repository implements **Adaptive Exploration Reward (AER)** for mathematical reasoning using the [verl](verl/README.md) framework (Volcano Engine Reinforcement Learning for LLMs). The project extends PPO/GRPO training with a custom exploration reward mechanism that encourages diverse reasoning paths while maintaining accuracy.

The core innovation is the dynamic weighting between accuracy rewards and exploration rewards. The exploration reward promotes diversity by computing token similarity between responses in the same group (where each prompt generates `n` responses), with higher similarity leading to lower exploration reward.

## Quick Reference

```bash
# Always activate environment first!
conda activate aer

# Start training
cd verl/recipe/aer && bash run.sh

# Stop training
pkill -f main_ppo

# Resume training (auto-detects latest checkpoint)
cd verl/recipe/aer && bash run.sh

# Monitor logs
tail -f verl/recipe/aer/log.txt
```

## Running Training

```bash
conda activate aer
cd verl/recipe/aer

# Configure GPU allocation at the top of run.sh (e.g., export CUDA_VISIBLE_DEVICES=4,5,6,7)
# Edit run.sh to set model path, hyperparameters, and data paths
bash run.sh
```

### Resume Training

Auto-resume is enabled by default (`resume_mode: auto` in config). Training automatically continues from the latest checkpoint:

```bash
conda activate aer
bash run.sh  # Automatically detects and resumes from latest checkpoint
```

To resume from a specific checkpoint, add to `run.sh`:
```bash
trainer.resume_from_path="<checkpoint_path>"
```

### Key Parameters in `run.sh`

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to base model (e.g., `/data/models/Qwen/Qwen3-4B-Base`) |
| `tau` | Target exploration reward level (affects dynamic weighting) |
| `rollout_n` | Number of responses per prompt for diversity computation (default: 16) |
| `train_batch_size` | Training batch size |
| `ppo_mini_batch_size` | PPO mini-batch size |
| `max_token_len_per_gpu` | Maximum tokens per GPU |
| `total_epochs` | Number of training epochs |
| `save_freq` | Checkpoint save frequency |
| `test_freq` | Validation frequency |

## Code Architecture

### Entry Point
- **[`main_ppo.py`](verl/recipe/aer/src/main_ppo.py)**: Hydra-based entry point. Initializes Ray, creates workers, and instantiates `RayAERTrainer`. Uses a remote `TaskRunner` class to avoid scheduling on the Ray head node.

### Custom Trainer
- **[`aer_ray_trainer.py`](verl/recipe/aer/src/aer_ray_trainer.py)**: `RayAERTrainer` extends `RayPPOTrainer`. Key methods:
  - `_compute_aer_reward()`: Computes accuracy + exploration rewards
  - `_update_aer_weight()`: Dynamically adjusts weight based on exploration reward vs target `tau`
  - `_compute_reward_colocate()`: Overrides parent class to use AER reward computation
  - `_validate()`: Validation mode uses accuracy-only reward

### Reward System
- **[`reward_manager.py`](verl/recipe/aer/src/reward_manager.py)**:
  - `math_accuracy_reward()`: Computes accuracy reward via `math_verify`
  - `compute_token_similarity()`: Computes pairwise token similarity matrix, grouped by prompt UID
  - `RLRewardManager`: Validation reward manager (accuracy only)
  - `AERRewardManager`: Training reward manager with exploration bonus

### Data Preparation
- **[`data_preparation.py`](verl/recipe/aer/src/data_preparation.py)**: Converts HuggingFace datasets to parquet format with standardized prompt format. Handles answer extraction from `\boxed{}` notation.

### Configuration
- **[`src/config/ppo_trainer.yaml`](verl/recipe/aer/src/config/ppo_trainer.yaml)**: Hydra configuration with extensive defaults for data, models, algorithm, and trainer settings.

## AER Algorithm Implementation

### Mathematical Formulation

**Accuracy Reward:**
```
r_acc(i) = 1.0 if answer_i == ground_truth else 0.0
```

**Token Similarity:**
```
sim(i, j) = sum(token_i^k == token_j^k) / sqrt(L_i * L_j)  if uid_i == uid_j
sim(i, j) = 0                                              otherwise
```
where `L_i` is the valid token length of response `i`.

**Exploration Reward:**
```
sim_sum(i) = sum_j sim(i, j)
r_exp(i) = 1.0 / sim_sum(i)  if sim_sum(i) > 0 else 0.0
```

**Dynamic Weight Update:**
```
w_{t+1} = clip(w_t + (tau - r̄_exp), 0, 1)
```
where `tau` is the target exploration reward level and `r̄_exp` is the mean exploration reward of the current batch.

**Final Reward:**
```
r_final(i) = r_acc(i) + w_t * r_exp(i)
```

### Key Implementation Details

1. **Group-based computation**: Responses are grouped by prompt UID (`rollout_n` responses per prompt). Similarity is only computed within groups.

2. **Token-level similarity**: Unlike answer-level similarity, token-level similarity rewards diverse reasoning paths, not just different final answers.

3. **Adaptive weighting**: The weight automatically adjusts during training:
   - If exploration < target → increase weight → encourage more diversity
   - If exploration > target → decrease weight → focus on accuracy

4. **Training vs Validation**:
   - Training: Uses `AERRewardManager` (accuracy + exploration)
   - Validation: Uses `RLRewardManager` (accuracy only)

## verl Framework Integration

This project uses verl with the following components:

- **GRPO (Group Relative Policy Optimization)**: Advantage estimator that normalizes advantages within groups of responses to the same prompt
- **HybridEngine**: Seamlessly transitions between training (FSDP) and inference (vLLM) modes
- **Ray-based distributed training**: Uses remote actors to avoid head node scheduling issues

### File Locations

- **Training logs**: [`verl/recipe/aer/log.txt`](verl/recipe/aer/log.txt)
- **Validation samples**: `save/rollout/validation/`
- **WandB logs**: `verl/recipe/aer/wandb/`
- **Checkpoints**: `checkpoints/AER/{experiment_name}/`
