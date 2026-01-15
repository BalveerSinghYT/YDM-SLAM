#!/usr/bin/env bash
# Process optical flow with NeuFlowV2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/FastFlowNet"

# 激活FastFlowNet虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate FastFlowNet

# TUM dataset configuration
TUM_DATASET="/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz"
TUM_ASSOC="/home/max/Dataset/TUM/associations/fr3_walking_xyz.txt"
TUM_TRAJECTORY="/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt"

# Output configuration
OUTPUT_DIR="./data/neuflowv2_output"
mkdir -p "$OUTPUT_DIR"

# Processing parameters
NUM_PAIRS=${1:-5}
STRIDE=${2:-1}
MODEL=${3:-neuflow_mixed}  # neuflow_mixed, neuflow_sintel, neuflow_things

# TUM3 camera intrinsics
FX=535.4
FY=539.2
CX=320.1
CY=247.6

# MAD parameters
ITERATIONS=5
SIGMA_SCALE=2.5
MIN_PX=1.5

echo "=========================================="
echo "NeuFlowV2 Optical Flow Processing"
echo "=========================================="
echo "Dataset: $TUM_DATASET"
echo "Associations: $TUM_ASSOC"
echo "Trajectory: $TUM_TRAJECTORY"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL"
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

python process_neuflowv2.py \
    --dataset "$TUM_DATASET" \
    --assoc "$TUM_ASSOC" \
    $TRAJECTORY_ARG \
    --num-pairs "$NUM_PAIRS" \
    --stride "$STRIDE" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL" \
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
echo ""
echo "💡 NeuFlowV2 vs FastFlowNet comparison:"
echo "  Compare with:    eog ./data/ego_compensated_flow/*_grid.png"
