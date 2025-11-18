#!/bin/bash

# ============================================================================
# 回环检测脚本 - 离线描述子 + 地图保存/加载
# ============================================================================
# 
# 功能说明：
# 1. 使用预计算的离线描述子（更快速）
# 2. 支持地图保存和加载，进一步加速后续运行
#
# 使用方法：
#   bash match_with_map_offline.sh
# ============================================================================

# 配置参数
DATASET_FRONT="/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/fisheye1"
DATASET_REAR="/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/fisheye3"
MODEL_PATH="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/checkpointout_end"
GT_POSES="/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/dr_pose.txt"
CAMERA_CFG="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras_eigenplaces_1.cfg"
CONFIG_YAML="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/config.yaml"

# 离线描述子路径
OFFLINE_DESC_PATH="/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/descriptors"

# 地图保存路径
MAP_FILE="/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/maps/keyframe_db_offline.bin"

# 创建地图目录（如果不存在）
MAP_DIR=$(dirname "$MAP_FILE")
mkdir -p "$MAP_DIR"

echo "=========================================="
echo "回环检测 - 离线描述子 + 地图模式"
echo "=========================================="
echo "离线描述子路径: $OFFLINE_DESC_PATH"
echo "地图文件: $MAP_FILE"

if [ -f "$MAP_FILE" ]; then
    echo "检测到已有地图文件，将直接加载"
    echo "文件大小: $(du -h $MAP_FILE | cut -f1)"
    echo "如需重新构建地图，请删除该文件：rm $MAP_FILE"
else
    echo "未找到地图文件，将使用离线描述子构建新地图"
fi
echo "=========================================="

cd build

# 运行程序（离线描述子 + 地图）
./test_match_global_feats \
    "$DATASET_FRONT" \
    "$DATASET_REAR" \
    "$MODEL_PATH" \
    "$GT_POSES" \
    "$CAMERA_CFG" \
    "$CONFIG_YAML" \
    "$OFFLINE_DESC_PATH" \
    "$MAP_FILE"

echo ""
echo "=========================================="
echo "程序执行完成"
if [ -f "$MAP_FILE" ]; then
    echo "地图文件: $MAP_FILE"
    echo "文件大小: $(du -h $MAP_FILE | cut -f1)"
fi
echo "=========================================="

