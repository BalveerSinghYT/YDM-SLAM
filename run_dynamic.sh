#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p pcd pcd_dyn

# FlowNet 配置
export YDM_FLOW_NET_MODEL_PATH="$SCRIPT_DIR/FastFlowNet/export/fastflownet_things3d_dynamic.onnx"
export YDM_FLOW_NET_KIND=FASTFLOWNET

# ✅ ROS 实时发布（直接发布，不需要 PCD 磁盘保存）
export YDM_ROS=1
export YDM_PCD_DUMP_DYNAMIC=0      # 关闭 PCD 磁盘保存（实时发布已涵盖）

# 调试可视化
export YDM_VIEW_DEBUG=1
export YDM_VIEW_DEBUG_EVERY=1        # 每1帧刷新一次
export YDM_VIEW_RESIDUAL_MAX=50      # Residual 归一化上限（像素）
export YDM_VIEW_FLOW_MAX=15          # Flow 归一化上限（像素）

# ✅ 新增：残差计算参数（这些是关键！）
export YDM_RESIDUAL_ITERATIONS=5   # 迭代次数（默认7，可调 1-20）
export YDM_RESIDUAL_SIGMA_SCALE=3  # Sigma 倍数（默认2.5，建议 1.5-3.5）
export YDM_RESIDUAL_MIN_PX=2.5       # 最小阈值（默认1.0px，建议 0.5-2.0）

# 抑制裂纹/闪烁：丢掉深度边缘像素 + 腐蚀有效深度区域
export YDM_DEPTH_EDGE_TH=0.20
export YDM_VALID_ERODE=1

# 形态学：填掉裂纹孔洞/去孤立噪点（可选但建议开）
export YDM_MOTION_MASK_CLOSE=2
export YDM_MOTION_MASK_OPEN=1

# ✅ 新增：动态物体检测参数
export YDM_OVERLAP_DYNAMIC_TH=0.70   # YOLO 重叠阈值（默认0.70，建议 0.5-0.9）
export YDM_REFINE_DILATE=3           # 膨胀核大小（默认3，建议 1-5）

# ✅ 新增：调试输出
export YDM_DEBUG_DYNAMIC=1           # 启用详细日志
export YDM_VERBOSE=1                 # 启用通用调试信息

# ✅ Option B: 背景迭代剥离 + GN 位姿精修（实验性功能，默认关闭）
export YDM_BG_PEEL=1                       # 启用背景剥离模式（0=关闭，1=开启）
export YDM_BG_PEEL_ITERS=7                 # 迭代次数（默认7）
export YDM_BG_PEEL_STRIDE=4                # GN/统计采样步长（越大越快但thr更抖）
export YDM_BG_PEEL_SIGMA_SCALE=3.5         # thr = median + sigma*1.4826*MAD（更大=更“宽容”=背景更纯）
export YDM_BG_PEEL_MIN_PX=1.5              # thr 最小值（防止某些帧thr过小导致背景乱闪）
export YDM_BG_PEEL_HUBER_DELTA=5.0         # Huber鲁棒核（若你用于位姿/权重）
export YDM_BG_PEEL_MIN_INLIER_RATIO=0.3    # early stop 背景占比下限

./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM3.yaml \
    /home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz \
    /home/max/Dataset/TUM/associations/fr3_walking_xyz.txt \
    "${1-}"