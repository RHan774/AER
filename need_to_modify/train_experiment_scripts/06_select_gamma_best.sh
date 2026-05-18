#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的选择参数。
TARGET_SIMILARITY_FOR_GAMMA_SEARCH="${TARGET_SIMILARITY_FOR_GAMMA_SEARCH:-ngram_overlap}"
GAMMA_SELECTION_PRIMARY_METRIC="${GAMMA_SELECTION_PRIMARY_METRIC:-pass@8}"
GAMMA_SELECTION_TIEBREAK_METRIC="${GAMMA_SELECTION_TIEBREAK_METRIC:-correct_rate}"

refresh_run_paths
require_tau_plan "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}"
select_gamma_best
