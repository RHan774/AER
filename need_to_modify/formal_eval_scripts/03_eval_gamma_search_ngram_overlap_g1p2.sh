#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的评测参数。
ALGORITHM="${ALGORITHM:-${TARGET_SIMILARITY_FOR_GAMMA_SEARCH:-ngram_overlap}}"
GAMMA="${GAMMA:-1.2}"
require_tau_plan "${ALGORITHM}"
TAU="${TAU:-$(read_tau_for_gamma "${ALGORITHM}" "${GAMMA}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(gamma_search_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"

run_single_formal_eval "${EXPERIMENT_NAME}"
