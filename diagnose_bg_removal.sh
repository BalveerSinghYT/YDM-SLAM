#!/usr/bin/env bash
# 快速诊断背景剔除算法问题

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/FastFlowNet"

# 激活FastFlowNet虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate FastFlowNet

# 查找最新的处理结果
OUTPUT_DIR="./data/ego_compensated_flow"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 输出目录不存在: $OUTPUT_DIR"
    echo "请先运行: ./run_ego_flow.sh"
    exit 1
fi

# 获取第一个pair的flow文件（注意：匹配不含ego的flow文件）
FLOW_FILE=$(ls "$OUTPUT_DIR"/pair_0000_*_flow.npy 2>/dev/null | grep -v "_ego_flow" | head -1)

if [ -z "$FLOW_FILE" ]; then
    echo "错误: 找不到处理结果"
    echo "请先运行: ./run_ego_flow.sh"
    exit 1
fi

# 提取基础名: pair_XXXX_tYYYYYY.YYY
# 方法: 去掉最后的 _flow.npy
BASE_NAME=$(basename "$FLOW_FILE" | sed 's/_flow\.npy$//')

# 构建所有文件路径
FLOW_FILE="$OUTPUT_DIR/${BASE_NAME}_flow.npy"
EGO_FLOW_FILE="$OUTPUT_DIR/${BASE_NAME}_ego_flow.npy"
RESIDUAL_FILE="$OUTPUT_DIR/${BASE_NAME}_residual.npy"
MOTION_MASK_FILE="$OUTPUT_DIR/${BASE_NAME}_motion_mask.npy"

# 查找对应的深度图
# 从文件名提取时间戳
TIMESTAMP=$(echo "$BASE_NAME" | grep -oP 't\K[0-9.]+')

# 在TUM数据集中查找对应深度图
TUM_DATASET="/home/max/Dataset/TUM/rgbd_dataset_freiburg3_walking_xyz"
TUM_ASSOC="/home/max/Dataset/TUM/associations/fr3_walking_xyz.txt"

DEPTH_PATH=$(awk -v ts="$TIMESTAMP" 'BEGIN{mindiff=999; best=""} 
    {diff=($1>ts?$1-ts:ts-$1); if(diff<mindiff && diff<0.05){mindiff=diff; best=$4}} 
    END{print best}' "$TUM_ASSOC")

if [ -z "$DEPTH_PATH" ]; then
    echo "警告: 找不到对应的深度图"
    DEPTH_FILE=""
else
    DEPTH_FILE="$TUM_DATASET/$DEPTH_PATH"
    if [ ! -f "$DEPTH_FILE" ]; then
        echo "警告: 深度图不存在: $DEPTH_FILE"
        DEPTH_FILE=""
    fi
fi

echo "=========================================="
echo "背景剔除算法诊断"
echo "=========================================="
echo "分析文件: $BASE_NAME"
echo "光流:     $FLOW_FILE"
echo "EgoFlow:  $EGO_FLOW_FILE"
echo "残差:     $RESIDUAL_FILE"
echo "运动mask: $MOTION_MASK_FILE"
echo "深度图:   ${DEPTH_FILE:-未找到}"
echo ""

if [ -z "$DEPTH_FILE" ]; then
    echo "错误: 需要深度图进行完整诊断"
    echo "请检查 TUM_DATASET 和 TUM_ASSOC 路径"
    exit 1
fi

# 从grid图像中提取阈值信息（如果有）
GRID_FILE="$OUTPUT_DIR/${BASE_NAME}_grid.png"
THR=0.0
if [ -f "$GRID_FILE" ]; then
    # 尝试从文件名或日志中提取阈值
    echo "提示: 如果知道MAD阈值，可以用 --threshold 参数传入"
fi

python diagnose_background_removal.py \
    --flow "$FLOW_FILE" \
    --ego-flow "$EGO_FLOW_FILE" \
    --residual "$RESIDUAL_FILE" \
    --motion-mask "$MOTION_MASK_FILE" \
    --depth "$DEPTH_FILE" \
    --threshold 0.0

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "💡 调整参数建议:"
echo ""
echo "如果动态像素太少:"
echo "  export YDM_RESIDUAL_SIGMA_SCALE=2.0  # 降低阈值"
echo "  export YDM_RESIDUAL_MIN_PX=1.0       # 降低最小阈值"
echo ""
echo "如果动态像素太多:"
echo "  export YDM_RESIDUAL_SIGMA_SCALE=3.5  # 提高阈值"
echo "  export YDM_RESIDUAL_MIN_PX=2.5       # 提高最小阈值"
echo ""
echo "如果深度边缘噪声多:"
echo "  export YDM_DEPTH_EDGE_TH=0.15        # 更激进的边缘剔除"
echo "  export YDM_VALID_ERODE=2             # 腐蚀有效区域"
echo ""
