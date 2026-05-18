#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的实验参数。
ALGORITHM="${ALGORITHM:-${TARGET_SIMILARITY_FOR_GAMMA_SEARCH:-ngram_overlap}}"
GAMMA="${GAMMA:-1.3}"
require_tau_plan "${ALGORITHM}"
TAU="${TAU:-$(read_tau_for_gamma "${ALGORITHM}" "${GAMMA}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(gamma_search_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"
TOTAL_STEPS="${TOTAL_STEPS:-${TOTAL_TRAINING_STEPS}}"

run_single_training_experiment "${EXPERIMENT_NAME}" "${ALGORITHM}" "${TAU}" "${TOTAL_STEPS}" "0.0" ""
