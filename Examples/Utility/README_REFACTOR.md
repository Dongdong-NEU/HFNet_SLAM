# 代码重构说明

## 文件组织结构

经过重构，原始的 `test_match_global_feats.cc` 现在被分解为以下几个模块：

### 1. common.h / common.cpp
**功能**: 通用数据结构和算法函数
**包含内容**:
- 数据结构定义
  - `KeyFrameNetVlad`: 关键帧数据结构
  - `TUMTrajectoryData`: TUM格式轨迹数据
  - `AlignedTUMData`: 对齐的轨迹数据
  - `KeyFramePointCloud`: KDTree点云适配器
- 文件读取函数
  - `GetTimesFromFileKitti()` / `GetGTPosesFromFileKitti()`: KITTI格式数据读取
  - `ReadTUMTrajectory()` / `AlignTUMTrajectoryToImages()`: TUM格式数据读取
- 回环检测算法
  - `GetNCandidateLoopFrameCV()`: OpenCV版本回环检测
  - `GetNCandidateLoopFrameEigen()`: Eigen版本回环检测（更快）
- 工具函数
  - `ShowImageWithText()`: 显示带文本的图像

### 2. loop_visual.h / loop_visual.cpp
**功能**: Pangolin 3D可视化模块
**包含内容**:
- `TrajectoryViewer` 类: 主要的3D可视化类
  - 轨迹显示（灰色线条）
  - 关键帧显示（绿色点）
  - 当前位置显示（红色点）
  - KDTree候选帧显示（蓝色点和连线）
  - Eigen候选帧显示（黄色点和连线）
  - 坐标轴显示（RGB轴）
  - 交互式控制面板
- 全局可视化管理函数
  - `StartVisualization()`: 启动可视化线程
  - `UpdateVisualization()`: 更新可视化数据

### 3. test_match_global_feats.cc
**功能**: 主程序入口
**包含内容**:
- `main()` 函数: 程序主逻辑
- 参数解析和数据加载
- 关键帧数据库构建
- KDTree构建和搜索
- 主循环: 逐帧处理和可视化
- 图像拼接和显示

## 编译说明

由于代码被分解为多个文件，需要确保CMakeLists.txt包含所有新文件：

```cmake
# 在CMakeLists.txt中添加新的源文件
set(SOURCES
    test_match_global_feats.cc
    common.cpp
    loop_visual.cpp
)
```

## 功能特性

### 性能对比
- **OpenCV版本**: 使用cv::norm()计算L2距离，较慢
- **Eigen版本**: 使用Eigen矩阵运算，约5倍速度提升

### 可视化功能
- **实时3D轨迹显示**: 显示相机运动轨迹
- **多候选帧对比**: 同时显示KDTree和Eigen检测结果
- **交互式控制**: 可暂停/继续，调整显示参数
- **图像对比窗口**: 7宫格显示查询帧和候选帧

### 数据格式支持
- **KITTI格式**: 12元素位姿矩阵 + 单独时间戳文件
- **TUM格式**: 时间戳 + 平移 + 四元数格式

## 使用方法

```bash
# 编译
cd build
make

# 运行 (KITTI格式)
./test_match_global_feats /path/to/images /path/to/model /path/to/poses.txt /path/to/times.txt

# 运行 (TUM格式)
./test_match_global_feats /path/to/images /path/to/model /path/to/groundtruth.txt
```

## 重构收益

1. **代码组织更清晰**: 功能模块化，便于维护
2. **编译更快**: 头文件依赖减少
3. **复用性更强**: 可视化和算法模块可独立使用
4. **调试更方便**: 模块职责明确
5. **扩展更容易**: 新增功能时只需修改对应模块
