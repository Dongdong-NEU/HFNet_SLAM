#!/bin/bash

echo "测试 panoramic_3 图像去畸变..."

# 设置测试图像路径（请修改为你的实际图像路径）
IMAGE_PATH="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/debug_image/fisheye_test/front_query_1763019572.065888.png"

# 检查图像是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "错误: 图像文件不存在: $IMAGE_PATH"
    echo "请修改脚本中的 IMAGE_PATH 变量为实际图像路径"
    exit 1
fi

cd build
./test_distort_image \
    "$IMAGE_PATH" \
    panoramic_3 \
    ../config/cameras_ot_1_4.cfg \
    ../config/config.yaml

echo "测试完成!"

