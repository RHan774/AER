#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# 可单独修改的评测参数。
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(baseline_naive_exp_name)}"
FORMAL_EVAL_CHECKPOINT_STEP="${FORMAL_EVAL_CHECKPOINT_STEP:-${CALIBRATION_STEPS}}"

run_single_formal_eval "${EXPERIMENT_NAME}"
