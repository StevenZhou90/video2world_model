#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/test_videos}"
RGB_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet-rgb.avi"
DEPTH_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet-depth.avi"
RGB_OUT="${OUT_DIR}/tum_freiburg3_cabinet_rgb.avi"
DEPTH_OUT="${OUT_DIR}/tum_freiburg3_cabinet_depth.avi"

mkdir -p "${OUT_DIR}"
curl -L --fail --progress-bar -o "${RGB_OUT}" "${RGB_URL}"
curl -L --fail --progress-bar -o "${DEPTH_OUT}" "${DEPTH_URL}"
printf '%s\n%s\n' "${RGB_OUT}" "${DEPTH_OUT}"
