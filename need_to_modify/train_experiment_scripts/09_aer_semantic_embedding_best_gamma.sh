#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的实验参数。
ALGORITHM="${ALGORITHM:-semantic_embedding}"
GAMMA="${GAMMA:-$(resolve_gamma_best)}"
require_tau_plan "${ALGORITHM}"
TAU="${TAU:-$(read_tau_for_gamma "${ALGORITHM}" "${GAMMA}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(main_aer_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"
TOTAL_STEPS="${TOTAL_STEPS:-${TOTAL_TRAINING_STEPS}}"

# semantic_embedding 设备也可在此单脚本覆盖：
# SIMILARITY_DEVICE="cuda"
# SIMILARITY_CUDA_VISIBLE_DEVICES="[4,5,6,7]"
# SIMILARITY_NUM_PROCESSES=4

run_single_training_experiment "${EXPERIMENT_NAME}" "${ALGORITHM}" "${TAU}" "${TOTAL_STEPS}" "0.0" ""
