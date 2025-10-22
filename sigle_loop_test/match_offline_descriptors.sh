#!/bin/bash

echo "使用离线描述子进行回环检测测试..."

cd build
./test_match_global_feats \
/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/fisheye1 \
/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/fisheye3 \
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/EigenPlaces/ONNX \
/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/dr_pose.txt \
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras.cfg \
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/config.yaml \
/home/xihuidong/Documents/visual_place_recognization_files/glb_save

echo "离线描述子回环检测测试完成!"

