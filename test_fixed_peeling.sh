#!/bin/bash
# Test FIXED DIO peeling in C++ SLAM system

echo "=========================================="
echo "Testing FIXED DIO Peeling in SLAM"
echo "=========================================="
echo ""

# Set environment variables
export YDM_BG_PEEL=1  # Enable background peeling
export YDM_VERBOSE=1
export YDM_DEBUG_DYNAMIC=1
export YDM_FLOW_NET_MODEL_PATH=/home/max/DynamicSLAM/YOLOv8-ORBSLAM/FastFlowNet/export/fastflownet.onnx
export YDM_FLOW_NET_KIND=FASTFLOWNET

# Test on TUM fr3_walking_xyz (has dynamic person)
DATASET_PATH=/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz
VOCAB_PATH=/home/max/DynamicSLAM/YOLOv8-ORBSLAM/Vocabulary/ORBvoc.txt
SETTINGS_PATH=/home/max/DynamicSLAM/YOLOv8-ORBSLAM/Examples/RGB-D/TUM3.yaml
ASSOC_FILE=$DATASET_PATH/freiburg3_walking_xyz.txt

echo "Dataset: $DATASET_PATH"
echo "Using FIXED DIO peeling (YDM_BG_PEEL=1)"
echo ""
echo "Key improvements:"
echo "  ✅ SE(3): Sophus log/exp (numerically stable)"
echo "  ✅ Auto pose direction detection"
echo "  ✅ Strict depth validity (Z>0, finite, in-bounds)"
echo "  ✅ True BG peeling (only update bg_mask pixels)"
echo "  ✅ 7 iterations (DIO-SLAM paper)"
echo ""
echo "Starting test (will process first 100 frames)..."
echo ""

cd /home/max/DynamicSLAM/YOLOv8-ORBSLAM

# Run SLAM with FIXED peeling
./Examples/RGB-D/rgbd_tum \
    $VOCAB_PATH \
    $SETTINGS_PATH \
    $DATASET_PATH \
    $ASSOC_FILE \
    100 \
    2>&1 | tee test_fixed_peeling.log

echo ""
echo "=========================================="
echo "Test Complete!"
echo "Log saved to: test_fixed_peeling.log"
echo ""
echo "Look for these messages in the log:"
echo "  [YDM][BG_PEEL] Auto-corrected pose direction"
echo "  [YDM][BG_PEEL] Pose direction correct"
echo "  [YDM][BG_PEEL] iter=X/7 thr=...px"
echo "=========================================="
