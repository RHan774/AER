#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的实验参数。
# 如需复用 gamma-search 的最佳 run，不需要运行本脚本；本脚本用于强制单独训练一条主实验。
ALGORITHM="${ALGORITHM:-ngram_overlap}"
GAMMA="${GAMMA:-$(resolve_gamma_best)}"
require_tau_plan "${ALGORITHM}"
TAU="${TAU:-$(read_tau_for_gamma "${ALGORITHM}" "${GAMMA}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(main_aer_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"
TOTAL_STEPS="${TOTAL_STEPS:-${TOTAL_TRAINING_STEPS}}"

run_single_training_experiment "${EXPERIMENT_NAME}" "${ALGORITHM}" "${TAU}" "${TOTAL_STEPS}" "0.0" ""
