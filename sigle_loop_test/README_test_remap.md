# 相机图像对齐程序说明

## 功能概述

这个程序用于解决**前置摄像头camera_1**和**后置摄像头camera_4**在车辆旋转180度后看到相同场景时的图像对齐问题。

### 问题背景

- 车辆配备两个摄像头：
  - `camera_1`：前置摄像头（3840x2160）
  - `camera_4`：后置摄像头（1920x1080）
- 两个摄像头的安装位置和朝向不完全相反
- 坐标系：以车辆后轴中心为基准，x方向指向车头，z方向指向天空
- 当车辆在某位置用camera_1看到物体A，然后旋转180度后用camera_4看到同一个物体A时，由于外参和内参的差异，A在两张图像中的位置不一致

### 解决方案

程序通过以下步骤实现图像对齐：

1. **解析相机参数**：从配置文件中读取camera_1和camera_4的内参（焦距、畸变系数等）和外参（位置、姿态）
2. **计算相对变换**：考虑车辆旋转180度后，计算从camera_1到camera_4的相对旋转和平移
3. **生成重映射表**：
   - 对camera_4进行标准的鱼眼去畸变
   - 对camera_1在去畸变的同时应用相对旋转变换，使其视角与camera_4对齐
4. **图像处理**：使用生成的映射表对输入图像进行remap，得到对齐后的结果

## 编译

```bash
cd /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test
./build.sh
```

## 使用方法

### 方式1：直接运行可执行文件

```bash
cd /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/build
./test_remap_image <camera1_image> <camera4_image>
```

### 方式2：使用测试脚本

```bash
cd /home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test
./test_remap.sh <camera1_image> <camera4_image>
```

### 参数说明

- `camera1_image`: camera_1在某个位置拍摄的图像路径
- `camera4_image`: 车辆在同一位置旋转180度后，camera_4拍摄的图像路径

### 示例

```bash
./test_remap.sh /path/to/scene_camera1.jpg /path/to/scene_camera4.jpg
```

## 输出结果

程序会生成以下文件：

1. `camera1_remapped.jpg` - camera_1处理后的图像
2. `camera4_remapped.jpg` - camera_4处理后的图像
3. `camera1_remapped_grid.jpg` - camera_1处理后的图像（带网格）
4. `camera4_remapped_grid.jpg` - camera_4处理后的图像（带网格）
5. `comparison.jpg` - 对比图（上方：两张处理后的原图，下方：带网格的图像）

## 验证方法

1. **观察网格对齐**：在两张带网格的图像中，相同的场景特征（如建筑物、地标等）应该在网格中处于相似的位置
2. **对比图观察**：在comparison.jpg中，左右两张图像中的同一物体应该在图像中的相对位置基本一致
3. **中心十字**：两张图像的中心位置都有红色十字标记，场景中心的物体应该接近这个标记

## 技术细节

### 相机参数

**camera_1 (前置)**
- 分辨率: 3840x2160
- 内参: fx=1886.09, fy=1896.02, cx=1921.50, cy=1097.82
- 畸变: k1=-0.0367, k2=-0.0131, k3=0.0273, k4=-0.0157
- 位置: [0.923, 0.001, -0.249]米
- 姿态: 朝向车头前方

**camera_4 (后置)**
- 分辨率: 1920x1080
- 内参: fx=1146.82, fy=1145.55, cx=961.63, cy=559.58
- 畸变: k1=-0.0767, k2=0.0301, k3=-0.0256, k4=0.0050
- 位置: [-2.527, -0.064, -0.779]米
- 姿态: 朝向车尾后方

### 相对变换（车辆旋转180度后）

- **相对旋转**: 单位矩阵（朝向完全对齐）
- **相对平移**: [0.0628, -0.530, -1.605]米

这说明在理想情况下，旋转180度后两个相机的光轴方向是平行的，主要差异在于位置偏移。

## 配置文件

相机参数配置文件位于：
```
/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras-indoor.cfg
```

## 原理说明

### 坐标系统

1. **车辆坐标系**（Sensor）：
   - 原点：车辆后轴中心
   - X轴：指向车头
   - Y轴：指向左侧
   - Z轴：指向天空

2. **相机坐标系**：
   - 原点：相机光心
   - Z轴：指向相机前方（光轴方向）
   - X轴：指向相机右方
   - Y轴：指向相机下方

### 变换流程

1. 车辆在位置P，camera_1拍摄物体A
2. 车辆旋转180度（绕Z轴旋转π），仍在位置P
3. 此时camera_4拍摄到同一物体A
4. 程序计算：
   ```
   T_vehicle_rot = Rotation(Z, 180°)
   T_cam1_rotated = T_vehicle_rot * T_cam1_original
   T_relative = T_cam4^(-1) * T_cam1_rotated
   ```
5. 使用T_relative对camera_1的图像进行rectify，使其与camera_4对齐

## 注意事项

1. 输入图像必须是对应相机拍摄的原始分辨率图像（camera_1: 3840x2160, camera_4: 1920x1080）
2. 两张图像应该是在车辆完成180度旋转前后，在相同位置拍摄的同一场景
3. 程序使用Kannala-Brandt鱼眼畸变模型进行去畸变
4. 输出图像的分辨率统一为camera_4的分辨率（1920x1080）

## 常见问题

### Q: 为什么输出尺寸是1920x1080？
A: 因为camera_4的分辨率较小，为了避免上采样导致的图像质量下降，程序选择使用较小的分辨率作为输出。

### Q: 如果对齐效果不好怎么办？
A: 可能的原因：
1. 检查输入图像是否真的是同一场景（车辆是否真的旋转了180度）
2. 检查相机标定参数是否准确
3. 检查外参配置是否正确

### Q: 可以用于实时处理吗？
A: 映射表的计算只需要进行一次，之后可以保存并重复使用。在实际应用中，可以预先计算好映射表，实时处理时只需要执行remap操作，速度很快。

## 作者

开发日期：2025年11月11日

