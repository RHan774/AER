#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的评测参数。
# USE_GAMMA_SEARCH_RUN=1 时评测复用的最佳 gamma-search run；设为 0 时评测单独训练的 aer-ngram_overlap run。
ALGORITHM="${ALGORITHM:-ngram_overlap}"
GAMMA="${GAMMA:-$(resolve_gamma_best)}"
USE_GAMMA_SEARCH_RUN="${USE_GAMMA_SEARCH_RUN:-${FORMAL_EVAL_REUSE_GAMMA_SEARCH_FOR_TARGET:-1}}"
require_tau_plan "${ALGORITHM}"
TAU="${TAU:-$(read_tau_for_gamma "${ALGORITHM}" "${GAMMA}")}"
if bool_is_true "${USE_GAMMA_SEARCH_RUN}"; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(gamma_search_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(main_aer_exp_name "${ALGORITHM}" "${GAMMA}" "${TAU}")}"
fi

run_single_formal_eval "${EXPERIMENT_NAME}"
