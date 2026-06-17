#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/test_videos}"
URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet-rgb.avi"
OUT_FILE="${OUT_DIR}/tum_freiburg3_cabinet_rgb.avi"

mkdir -p "${OUT_DIR}"
curl -L --fail --progress-bar -o "${OUT_FILE}" "${URL}"
echo "${OUT_FILE}"
