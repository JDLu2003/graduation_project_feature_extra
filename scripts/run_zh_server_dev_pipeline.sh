#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-security}"
STEP="${1:-all}"

DATA_ROOT="/data16T_1/sunshengzhe/lujiading/data_zh"
TRAIN_TXT="${DATA_ROOT}/train/train.txt"
DEV_TXT="${DATA_ROOT}/dev/dev.txt"
DEV_OUTPUT_DIR="${DATA_ROOT}/dev"

DATASET_BUILD_CONFIG="${ROOT}/server_configs/zh/dataset_build_train_zh_server.yaml"
FACE_TRAIN_CONFIG="${ROOT}/server_configs/zh/facenet_fr_train_zh_server.yaml"
MAIN_CONFIG="${ROOT}/server_configs/zh/main_dev_zh_server.yaml"
FEATURE_ROOT="${ROOT}/artifacts_zh/features/dev/Video_dev_face_scene_fr"
FACE_CHECKPOINT="${ROOT}/face_name_id/artifacts_zh/train_speaker_model/facenet_fr/outputs/checkpoints/best.pt"
MERGED_EMBEDDING="${DEV_OUTPUT_DIR}/video_embedding_dev.npy"
MERGED_MAPPING="${DEV_OUTPUT_DIR}/video_id_mapping_dev.npy"

run_py() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -u "$@"
}

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing file: ${path}" >&2
    exit 1
  fi
  echo "[ok] file exists: ${path}"
}

check_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[error] missing directory: ${path}" >&2
    exit 1
  fi
  echo "[ok] directory exists: ${path}"
}

run_step() {
  local name="$1"
  shift
  echo
  echo "=== ${name} ==="
  echo "[cmd] $*"
  "$@"
}

verify_inputs() {
  echo "=== Verify Inputs ==="
  check_dir "${DATA_ROOT}"
  check_file "${TRAIN_TXT}"
  check_file "${DEV_TXT}"
  check_file "${DATASET_BUILD_CONFIG}"
  check_file "${FACE_TRAIN_CONFIG}"
  check_file "${MAIN_CONFIG}"
}

role_stats() {
  run_step "Role Stats" \
    run_py "${ROOT}/scripts/stat_role_frequencies.py" \
    --txt-path "${TRAIN_TXT}" \
    --out-csv "${ROOT}/logs/zh_train_role_stats.csv" \
    --out-json "${ROOT}/logs/zh_train_role_stats.json"
  check_file "${ROOT}/logs/zh_train_role_stats.csv"
  check_file "${ROOT}/logs/zh_train_role_stats.json"
}

build_face_dataset() {
  run_step "Build Face Dataset" \
    run_py "${ROOT}/face_name_id/scripts/build_dataset.py" \
    --config "${DATASET_BUILD_CONFIG}"
  check_dir "${ROOT}/face_name_id/artifacts_zh/train_speaker_model/dataset/images"
}

train_face_model() {
  run_step "Train Face Model" \
    run_py "${ROOT}/face_name_id/scripts/train_facenet_fr.py" \
    --config "${FACE_TRAIN_CONFIG}"
  check_file "${FACE_CHECKPOINT}"
}

extract_smoke() {
  check_file "${FACE_CHECKPOINT}"
  run_step "Extract Smoke" \
    run_py "${ROOT}/main.py" \
    --config "${MAIN_CONFIG}" \
    --smoke \
    --max-dialogues 2
  check_dir "${FEATURE_ROOT}"
}

extract_full() {
  check_file "${FACE_CHECKPOINT}"
  run_step "Extract Full" \
    run_py "${ROOT}/main.py" \
    --config "${MAIN_CONFIG}"
  check_dir "${FEATURE_ROOT}"
}

merge_outputs() {
  run_step "Merge Outputs" \
    run_py "${ROOT}/scripts/merge_video_dev_features.py" \
    --config "${MAIN_CONFIG}" \
    --output-dir "${DEV_OUTPUT_DIR}"
  check_file "${MERGED_EMBEDDING}"
  check_file "${MERGED_MAPPING}"
}

verify_outputs() {
  run_step "Verify Outputs" \
    run_py "${ROOT}/scripts/run_zh_feature_workflow.py" \
    --dataset-root "${DATA_ROOT}" \
    --speaker-split train \
    --target-split dev \
    --feature-root "${ROOT}/artifacts_zh/features/dev" \
    --merged-output-dir "${DEV_OUTPUT_DIR}" \
    --face-work-dir "${ROOT}/face_name_id/artifacts_zh/train_speaker_model" \
    --run verify
}

verify_inputs

case "${STEP}" in
  role-stats)
    role_stats
    ;;
  build-face-dataset)
    build_face_dataset
    ;;
  train-face-model)
    train_face_model
    ;;
  extract-smoke)
    extract_smoke
    ;;
  extract-full)
    extract_full
    ;;
  merge)
    merge_outputs
    ;;
  verify)
    verify_outputs
    ;;
  all)
    role_stats
    build_face_dataset
    train_face_model
    extract_smoke
    extract_full
    merge_outputs
    verify_outputs
    ;;
  *)
    echo "Usage: bash scripts/run_zh_server_dev_pipeline.sh [all|role-stats|build-face-dataset|train-face-model|extract-smoke|extract-full|merge|verify]" >&2
    exit 1
    ;;
esac
