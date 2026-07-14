#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AWUG_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IDENT_DIR="${ROOT}/code/identification"
CONFIG_DIR="${ROOT}/config"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

stage_file_list() {
  local stage="$1"
  local split="$2"
  "${PYTHON_BIN}" - "$ROOT" "$stage" "$split" <<'PY'
import re
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
stage = sys.argv[2]
split = sys.argv[3]
index_path = root / "data" / "index" / "index.csv"
df = pd.read_csv(index_path)


def parse_name(data_file_name):
    name = Path(str(data_file_name)).name
    match = re.match(r"^(\d{4})_(\d+)_(\d+)-", name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def sort_key(row):
    parsed = parse_name(row["data_file_name"])
    if parsed is None:
        return ("", 10**9, 10**9, int(row["row_begin"]), str(row["data_file_name"]))
    return (*parsed, int(row["row_begin"]), str(row["data_file_name"]))


def selected(row):
    data_file_name = str(row["data_file_name"]).replace("\\", "/")
    if not data_file_name.startswith("processed_mocap/stage_a_to_e/"):
        return False
    parsed = parse_name(data_file_name)
    if parsed is None:
        return False
    _, condition, trial = parsed
    if stage in {"stage_a", "stage_b"}:
        conditions = {2, 3, 4, 7, 8, 9, 14, 15, 16}
        return condition in conditions and trial in ({1, 2, 3} if split == "train" else {4})
    if stage in {"stage_c", "stage_d"}:
        conditions = set(range(17, 44))
        return condition in conditions and trial in ({1, 2, 3} if split == "train" else {4})
    if stage == "stage_e" and split == "train":
        return 44 <= condition <= 61 and trial in {1, 2, 3}
    if stage == "stage_e" and split == "test_scheme41":
        return 44 <= condition <= 61 and trial in {4}
    raise SystemExit(f"Unsupported stage/split: {stage}/{split}")

rows = [row for _, row in df.iterrows() if selected(row)]
rows.sort(key=sort_key)
file_args = []
for row in rows:
    path = root / "data" / str(row["data_file_name"])
    file_args.append(f"{path}@{int(row['row_begin'])}:{int(row['row_end'])}")
print(",".join(file_args))
PY
}
