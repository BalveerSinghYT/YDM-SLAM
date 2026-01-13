#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p pcd pcd_dyn

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/max/Dataset/TUM/rgbd_dataset_freiburg3_sitting_xyz /home/max/Dataset/TUM/associations/fr3_sitting_xyz.txt "${1-}"

# Note: result.pcd is meaningful only if you run with: export YDM_PCD_GLOBAL=1
# pcl_viewer result.pcd