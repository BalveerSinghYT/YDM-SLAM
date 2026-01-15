#!/usr/bin/env bash
# Process optical flow with ego-motion compensation and background removal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/FastFlowNet"

# 激活FastFlowNet虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate FastFlowNet

# TUM dataset configuration
TUM_DATASET="/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz"
TUM_ASSOC="/home/max/Dataset/TUM/associations/fr3_walking_xyz.txt"
TUM_TRAJECTORY="/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt"  # Leave empty for static camera assumption, or provide path to trajectory

# Output configuration
OUTPUT_DIR="./data/ego_compensated_flow"
mkdir -p "$OUTPUT_DIR"

# Processing parameters
NUM_PAIRS=${1:-5}         # Number of frame pairs (default: 5)
STRIDE=${2:-1}            # Frame stride (default: 1)

# TUM3 camera intrinsics (from TUM3.yaml)
FX=535.4
FY=539.2
CX=320.1
CY=247.6

# MAD iteration parameters (same as YDM_RESIDUAL_* in run_dynamic.sh)
ITERATIONS=5              # Number of MAD iterations
SIGMA_SCALE=2.5           # Threshold = median + sigma_scale * 1.4826 * MAD
MIN_PX=1.5                # Minimum threshold in pixels

echo "=========================================="
echo "Ego-Motion Compensated Flow Processing"
echo "=========================================="
echo "Dataset: $TUM_DATASET"
echo "Associations: $TUM_ASSOC"
echo "Trajectory: ${TUM_TRAJECTORY:-None (static camera)}"
echo "Output: $OUTPUT_DIR"
echo "Processing: $NUM_PAIRS frame pairs (stride=$STRIDE)"
echo ""
echo "Camera Intrinsics:"
echo "  fx=$FX, fy=$FY, cx=$CX, cy=$CY"
echo ""
echo "MAD Parameters:"
echo "  iterations=$ITERATIONS, sigma_scale=$SIGMA_SCALE, min_px=$MIN_PX"
echo ""

TRAJECTORY_ARG=""
if [ -n "$TUM_TRAJECTORY" ]; then
    TRAJECTORY_ARG="--trajectory $TUM_TRAJECTORY"
fi

python process_flow_with_ego_compensation.py \
    --dataset "$TUM_DATASET" \
    --assoc "$TUM_ASSOC" \
    $TRAJECTORY_ARG \
    --num-pairs "$NUM_PAIRS" \
    --stride "$STRIDE" \
    --output "$OUTPUT_DIR" \
    --fx "$FX" --fy "$FY" --cx "$CX" --cy "$CY" \
    --iterations "$ITERATIONS" \
    --sigma-scale "$SIGMA_SCALE" \
    --min-px "$MIN_PX"

echo ""
echo "✅ Done! View results:"
echo "  Grid view:       eog $OUTPUT_DIR/*_grid.png"
echo "  Motion mask:     eog $OUTPUT_DIR/*_motion_mask.png"
echo "  Motion overlay:  eog $OUTPUT_DIR/*_motion_overlay.png"
echo ""
echo "Results saved in: $OUTPUT_DIR"
