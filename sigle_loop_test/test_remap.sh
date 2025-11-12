#!/bin/bash

# 测试相机图像对齐程序
# 用于对齐camera_1和camera_4在车辆旋转180度时看到的相同场景

echo "========================================"
echo "测试相机图像对齐程序"
echo "========================================"

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <camera1_image> <camera4_image>"
    echo ""
    echo "说明:"
    echo "  camera1_image: camera_1拍摄的图像路径"
    echo "  camera4_image: 车辆旋转180度后camera_4拍摄的图像路径"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/camera1.jpg /path/to/camera4.jpg"
    exit 1
fi

CAMERA1_IMAGE="$1"
CAMERA4_IMAGE="$2"

# 检查图像文件是否存在
if [ ! -f "$CAMERA1_IMAGE" ]; then
    echo "错误: camera1图像文件不存在: $CAMERA1_IMAGE"
    exit 1
fi

if [ ! -f "$CAMERA4_IMAGE" ]; then
    echo "错误: camera4图像文件不存在: $CAMERA4_IMAGE"
    exit 1
fi

# 运行程序
cd build
./test_remap_image "$CAMERA1_IMAGE" "$CAMERA4_IMAGE"

echo ""
echo "========================================"
echo "测试完成!"
echo "========================================"

