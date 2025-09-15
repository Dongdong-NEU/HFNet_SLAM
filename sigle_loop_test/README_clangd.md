# clangd 配置说明

本项目已经配置了 clangd Language Server，以提供更好的 C++ 代码编辑体验。

## 配置文件

### 1. `.clangd` - 主配置文件
- **位置**: 项目根目录
- **功能**: 
  - 配置编译数据库路径 (`build/compile_commands.json`)
  - 设置诊断选项（ClangTidy 规则）
  - 启用代码补全、悬停提示、类型推导等功能
  - 配置头文件路径和编译器参数

### 2. `.clang-format` - 代码格式化配置
- **位置**: 项目根目录  
- **功能**: 统一 C++ 代码格式风格
- **基于**: LLVM 风格，进行了自定义调整

### 3. `compile_commands.json` - 编译数据库
- **位置**: 项目根目录（符号链接到 `build/compile_commands.json`）
- **功能**: 提供编译命令信息，让 clangd 了解项目结构

## 支持的功能

### 代码跳转
- **转到定义**: 支持函数、类、变量的定义跳转
- **转到声明**: 快速跳转到声明位置  
- **查找引用**: 查找符号的所有使用位置

### 头文件解析
项目配置了以下头文件路径：
- `include/` - 项目头文件
- `include/Extractors/` - 提取器相关头文件
- `/usr/local/include/eigen3` - Eigen3 数学库
- `/usr/include/opencv4` - OpenCV 计算机视觉库
- `/usr/local/cuda/include` - CUDA 头文件
- `/home/xihuidong/usr_lib/TensorRT-8.5.3.1/include` - TensorRT 头文件

### 语法高亮和诊断
- 实时语法检查
- ClangTidy 代码质量检查
- 错误和警告提示
- 代码补全建议

### 代码格式化
- 自动代码格式化
- 统一的代码风格
- 支持保存时自动格式化

## 使用方法

### 在 VSCode 中使用
1. 安装 `clangd` 扩展
2. 禁用 `C/C++` 扩展（避免冲突）
3. 重启编辑器

### 在 Neovim 中使用
配置 LSP 客户端指向 clangd

### 在其他编辑器中使用
大多数现代编辑器都支持 Language Server Protocol (LSP)

## 故障排除

### 1. 代码补全不工作
- 确保 `compile_commands.json` 存在且有效
- 检查 `.clangd` 配置文件语法
- 重新构建项目生成新的编译数据库

### 2. 头文件找不到
- 检查 CMakeLists.txt 中的 include_directories
- 确保依赖库（OpenCV、Eigen3、TensorRT）已正确安装
- 更新 `.clangd` 配置文件中的头文件路径

### 3. 重新生成编译数据库
```bash
cd build
cmake ..
make
```

### 4. 验证配置
```bash
# 检查 clangd 是否安装
clangd --version

# 检查编译数据库
cat compile_commands.json | jq '.[0]'
```

## 依赖项目
- OpenCV 4.x
- Eigen3 3.1.0+
- Pangolin
- TensorRT 8.5.3.1
- CUDA

## 编译器标准
- C++14 标准
- GCC 编译器
- OpenMP 支持























