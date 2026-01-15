#!/usr/bin/env bash
# Quick script to run FastFlowNet on TUM dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/FastFlowNet"

# 激活FastFlowNet虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate FastFlowNet

# TUM dataset configuration
TUM_DATASET="/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz"
TUM_ASSOC="/home/max/Dataset/TUM/associations/fr3_walking_xyz.txt"

# Output configuration
OUTPUT_DIR="./data/tum_flow_output"
mkdir -p "$OUTPUT_DIR"

# Flow computation parameters
NUM_PAIRS=${1:-10}        # Number of frame pairs to process (default: 10)
STRIDE=${2:-1}            # Frame stride (default: 1 = consecutive frames)

echo "=========================================="
echo "FastFlowNet TUM Dataset Flow Inference"
echo "=========================================="
echo "Dataset: $TUM_DATASET"
echo "Associations: $TUM_ASSOC"
echo "Output: $OUTPUT_DIR"
echo "Processing: $NUM_PAIRS frame pairs (stride=$STRIDE)"
echo ""

python demo_tum.py \
    --dataset "$TUM_DATASET" \
    --assoc "$TUM_ASSOC" \
    --num-pairs "$NUM_PAIRS" \
    --stride "$STRIDE" \
    --output "$OUTPUT_DIR"

echo ""
echo "✅ Done! View results in: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  eog $OUTPUT_DIR/*_comparison.png"
echo "  eog $OUTPUT_DIR/*_flow.png"
