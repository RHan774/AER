#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的实验参数。
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(baseline_entropy_exp_name)}"
ALGORITHM="${ALGORITHM:-${BASELINE_SIMILARITY_ALGORITHM:-token_match}}"
TAU="${TAU:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-${TOTAL_TRAINING_STEPS}}"
ENTROPY_COEFF="${ENTROPY_COEFF:-${ENTROPY_BASELINE_COEFF:-0.0}}"
EXPLORATION_ALGORITHMS="${EXPLORATION_ALGORITHMS:-}"

run_single_training_experiment "${EXPERIMENT_NAME}" "${ALGORITHM}" "${TAU}" "${TOTAL_STEPS}" "${ENTROPY_COEFF}" "${EXPLORATION_ALGORITHMS}"
