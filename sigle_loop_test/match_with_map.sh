#!/bin/bash

# ============================================================================
# 回环检测脚本 - 支持地图保存和加载
# ============================================================================
# 
# 功能说明：
# 1. 第一次运行：构建关键帧数据库，提取特征，并保存地图到指定路径
# 2. 后续运行：直接加载已保存的地图，跳过耗时的构建过程
#
# 使用方法：
#   bash match_with_map.sh
# ============================================================================

# 配置参数
DATASET_FRONT="/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera1"
DATASET_REAR="/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera4"
MODEL_PATH="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/EigenPlaces/ONNX"
GT_POSES="/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/dr_pose.txt"
CAMERA_CFG="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras.cfg"
CONFIG_YAML="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/config.yaml"

# 地图保存路径
MAP_FILE="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/maps/keyframe_db.bin"

# 创建地图目录（如果不存在）
MAP_DIR=$(dirname "$MAP_FILE")
mkdir -p "$MAP_DIR"

echo "=========================================="
echo "回环检测 - 地图保存/加载模式"
echo "=========================================="
echo "地图文件: $MAP_FILE"

if [ -f "$MAP_FILE" ]; then
    echo "检测到已有地图文件，将直接加载"
    echo "文件大小: $(du -h $MAP_FILE | cut -f1)"
    echo "如需重新构建地图，请删除该文件：rm $MAP_FILE"
else
    echo "未找到地图文件，将构建新地图"
fi
echo "=========================================="

cd build

# 运行程序（会自动判断是加载还是构建）
./test_match_global_feats \
    "$DATASET_FRONT" \
    "$DATASET_REAR" \
    "$MODEL_PATH" \
    "$GT_POSES" \
    "$CAMERA_CFG" \
    "$CONFIG_YAML" \
    "$MAP_FILE"

echo ""
echo "=========================================="
echo "程序执行完成"
if [ -f "$MAP_FILE" ]; then
    echo "地图文件: $MAP_FILE"
    echo "文件大小: $(du -h $MAP_FILE | cut -f1)"
fi
echo "=========================================="

