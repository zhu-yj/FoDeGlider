#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage_common.sh"
TRAIN_FILES="${TRAIN_FILES:-$(stage_file_list stage_a train)}"
TEST_FILES="${TEST_FILES:-$(stage_file_list stage_a test)}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/stage_a_to_e}"
"${PYTHON_BIN}" "${IDENT_DIR}/SI_awug_a.py"   --out-dir "${OUT_DIR}" --device "${DEVICE:-cpu}" --sample-step "${SAMPLE_STEP:-1}"   --dt-base "${DT_BASE:-0.011111111111111}" --norm-mode "${NORM_MODE:-minmax}"   --norm-stats-json "${CONFIG_DIR}/norm_stats_minmax_local.json"   --train-files "${TRAIN_FILES}" --test-files "${TEST_FILES}" --skip-old-eval
