#!/bin/bash

# ============================================================================
# 相机对齐与相似度验证脚本
# ============================================================================

echo "========================================"
echo "相机对齐与相似度验证脚本"
echo "========================================"

CAMERA1_IMAGE="/home/xihuidong/codetree/repo/visual_mapping_test/OT_1_4/fisheye1/1763019575.165889.png"
CAMERA4_IMAGE="/home/xihuidong/codetree/repo/visual_mapping_test/OT_1_4/fisheye3/1763019374.265722.png"

# 检查图像文件是否存在
if [ ! -f "$CAMERA1_IMAGE" ]; then
    echo "错误: 找不到camera1图像: $CAMERA1_IMAGE"
    exit 1
fi

if [ ! -f "$CAMERA4_IMAGE" ]; then
    echo "错误: 找不到camera4图像: $CAMERA4_IMAGE"
    exit 1
fi

# 切换到项目目录
cd /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test

# 检查可执行文件是否存在
if [ ! -f "build/test_remap_image" ]; then
    echo "错误: 找不到可执行文件 build/test_remap_image"
    echo "请先运行: ./build.sh"
    exit 1
fi

echo ""
echo "处理图像:"
echo "  camera1: $CAMERA1_IMAGE"
echo "  camera4: $CAMERA4_IMAGE"
echo ""

# 运行程序
./build/test_remap_image "$CAMERA1_IMAGE" "$CAMERA4_IMAGE"

echo ""
echo "========================================"
echo "处理完成！"
echo "========================================"
echo ""
echo "输出文件:"
echo "  - camera1_remapped.jpg           (camera1重映射后的图像)"
echo "  - camera4_remapped.jpg           (camera4重映射后的图像)"
echo "  - camera1_remapped_grid.jpg      (camera1带网格标记)"
echo "  - camera4_remapped_grid.jpg      (camera4带网格标记)"
echo "  - comparison.jpg                 (对比图)"
echo ""

