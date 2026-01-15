#!/bin/bash
# Quick viewer for Bonn Balloon dataset results

OUTPUT_DIR="FastFlowNet/data/bonn_balloon_output"

echo "🎨 查看Bonn Balloon数据集背景剔除结果"
echo "========================================"
echo ""
echo "选项："
echo "  1) 查看所有可视化网格 (grid.png)"
echo "  2) 查看运动叠加图 (overlay.png)"
echo "  3) 查看特定帧 (输入帧号 0-14)"
echo "  4) 生成GIF动画"
echo ""

if [ "$1" == "grid" ] || [ "$1" == "1" ]; then
    echo "📊 查看所有网格可视化..."
    eog ${OUTPUT_DIR}/pair_*_grid.png &
elif [ "$1" == "overlay" ] || [ "$1" == "2" ]; then
    echo "🔴 查看运动叠加图..."
    eog ${OUTPUT_DIR}/pair_*_motion_overlay.png &
elif [ "$1" == "gif" ] || [ "$1" == "4" ]; then
    echo "🎬 生成GIF动画..."
    if command -v convert &> /dev/null; then
        convert -delay 50 -loop 0 ${OUTPUT_DIR}/pair_*_grid.png ${OUTPUT_DIR}/animation.gif
        echo "✅ 已生成: ${OUTPUT_DIR}/animation.gif"
        eog ${OUTPUT_DIR}/animation.gif &
    else
        echo "❌ 需要安装 imagemagick: sudo apt install imagemagick"
    fi
elif [ ! -z "$1" ]; then
    # Specific frame number
    FRAME=$(printf "%04d" $1)
    echo "🖼️  查看帧 ${FRAME}..."
    eog ${OUTPUT_DIR}/pair_${FRAME}_*_grid.png ${OUTPUT_DIR}/pair_${FRAME}_*_overlay.png &
else
    echo "用法:"
    echo "  $0 grid          # 查看所有网格图"
    echo "  $0 overlay       # 查看所有叠加图"  
    echo "  $0 5             # 查看第5帧"
    echo "  $0 gif           # 生成GIF动画"
fi
