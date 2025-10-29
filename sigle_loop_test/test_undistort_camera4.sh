#!/bin/bash

echo "测试 camera_4 图像去畸变..."

# 设置测试图像路径（请修改为你的实际图像路径）
IMAGE_PATH="/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera4/1747822563.930000.png"

# 检查图像是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "错误: 图像文件不存在: $IMAGE_PATH"
    echo "请修改脚本中的 IMAGE_PATH 变量为实际图像路径"
    exit 1
fi

cd build
./test_distort_image \
    "$IMAGE_PATH" \
    camera_4 \
    ../config/cameras.cfg \
    ../config/config.yaml

echo "测试完成!"

