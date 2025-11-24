#!/bin/bash
echo "构建 test_match_global_feats 独立版本..."
mkdir -p build && cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
echo "构建完成! 可执行文件: build/test_match_global_feats"
