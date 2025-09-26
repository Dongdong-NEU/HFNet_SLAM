#!/bin/bash

echo "使用EigenPlaces模型进行回环检测测试..."

cd build
# ./test_match_global_feats \
# /home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/fisheye1 \
# /home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/fisheye3 \
# /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/EigenPlaces/ONNX \
# /home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/dr_pose.txt \
# /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras-indoor.cfg \
# /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/config.yaml

cd build
./test_match_global_feats \
/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/fisheye1 \
/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/fisheye3 \
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/EigenPlaces/ONNX \
/home/xihuidong/codetree/repo/visual_mapping_test/eigenplaces_test_1/dr_pose.txt \
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras_eigenplaces_1.cfg \
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/config.yaml

echo "EigenPlaces回环检测测试完成!"
