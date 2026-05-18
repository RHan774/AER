#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的评测参数。
ALGORITHM="${ALGORITHM:-token_match}"
GAMMA="${GAMMA:-$(resolve_gamma_best)}"
require_tau_plan "${ALGORITHM}"
TAU="${TAU:-$(read_tau_for_gamma "${ALGORITHM}" "${GAMMA}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(main_aer_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"

run_single_formal_eval "${EXPERIMENT_NAME}"
