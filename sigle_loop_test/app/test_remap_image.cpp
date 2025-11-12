#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "../include/Extractors/fisheye_utils.h"

using namespace std;
using namespace cv;

// 相机外参结构体
struct CameraExtrinsic {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Matrix4d T_sensor_to_cam;
};

// 从配置文件解析外参
bool ParseExtrinsic(const string& config_file, const string& camera_name, CameraExtrinsic& extrinsic) {
    ifstream file(config_file);
    if (!file.is_open()) {
        cerr << "无法打开配置文件: " << config_file << endl;
        return false;
    }
    
    string line;
    bool found_camera = false;
    bool in_position = false;
    
    while (getline(file, line)) {
        // 查找camera_dev
        if (line.find("camera_dev:") != string::npos && line.find(camera_name) != string::npos) {
            found_camera = true;
            continue;
        }
        
        if (!found_camera) continue;
        
        // 解析外参 - 位置
        if (line.find("position") != string::npos) {
            in_position = true;
        } else if (in_position && line.find("x:") != string::npos) {
            double x, y, z;
            sscanf(line.c_str(), " x: %lf", &x);
            getline(file, line);
            sscanf(line.c_str(), " y: %lf", &y);
            getline(file, line);
            sscanf(line.c_str(), " z: %lf", &z);
            extrinsic.position = Eigen::Vector3d(x, y, z);
            in_position = false;
        }
        // 解析外参 - 姿态
        else if (line.find("qx:") != string::npos) {
            double qx, qy, qz, qw;
            sscanf(line.c_str(), " qx: %lf", &qx);
            getline(file, line);
            sscanf(line.c_str(), " qy: %lf", &qy);
            getline(file, line);
            sscanf(line.c_str(), " qz: %lf", &qz);
            getline(file, line);
            sscanf(line.c_str(), " qw: %lf", &qw);
            extrinsic.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
            
            // 构建变换矩阵
            extrinsic.T_sensor_to_cam = Eigen::Matrix4d::Identity();
            extrinsic.T_sensor_to_cam.block<3, 3>(0, 0) = extrinsic.orientation.toRotationMatrix();
            extrinsic.T_sensor_to_cam.block<3, 1>(0, 3) = extrinsic.position;
            
            break;  // 读取完一个相机配置
        }
    }
    
    if (!found_camera) {
        cerr << "未找到相机: " << camera_name << endl;
        return false;
    }
    
    return true;
}

// 打印相机外参
void printCameraExtrinsic(const string& camera_name, const CameraExtrinsic& extrinsic, const FisheyeCameraParams& params) {
    cout << "\n=== 相机参数: " << camera_name << " ===" << endl;
    cout << "分辨率: " << params.imageSize.width << "x" << params.imageSize.height << endl;
    cout << "内参矩阵 K:\n" << params.K << endl;
    cout << "畸变系数 D:\n" << params.D.t() << endl;
    cout << "位置: [" << extrinsic.position.transpose() << "]" << endl;
    cout << "姿态(四元数): [w=" << extrinsic.orientation.w() << ", x=" 
         << extrinsic.orientation.x() << ", y=" << extrinsic.orientation.y() << ", z="
         << extrinsic.orientation.z() << "]" << endl;
    
    Eigen::Matrix3d R = extrinsic.orientation.toRotationMatrix();
    cout << "旋转矩阵:\n" << R << endl;
}

// 生成从camera1到camera4的重映射
// 当车辆旋转180度时，将camera1的视角映射到camera4的视角
void computeRemapForAlignment(const FisheyeCameraParams& cam1_params,
                               const CameraExtrinsic& cam1_ext,
                               const FisheyeCameraParams& cam4_params,
                               const CameraExtrinsic& cam4_ext,
                               Mat& map1_x, Mat& map1_y,
                               Mat& map4_x, Mat& map4_y,
                               Size& output_size) {
    
    // 车辆旋转180度的变换矩阵（绕z轴旋转180度）
    Eigen::Matrix4d T_vehicle_rot = Eigen::Matrix4d::Identity();
    T_vehicle_rot(0, 0) = -1;  // x' = -x
    T_vehicle_rot(1, 1) = -1;  // y' = -y
    // z不变
    
    // 计算相对变换
    // 当车辆旋转180度后，camera1的位置和姿态在新坐标系下的表示
    Eigen::Matrix4d T_cam1_original = cam1_ext.T_sensor_to_cam;
    Eigen::Matrix4d T_cam4_original = cam4_ext.T_sensor_to_cam;
    
    // 车辆旋转后，camera1在新位置的变换
    Eigen::Matrix4d T_cam1_rotated = T_vehicle_rot * T_cam1_original;
    
    // 从rotated camera1到camera4的相对变换
    Eigen::Matrix4d T_relative = T_cam4_original.inverse() * T_cam1_rotated;
    
    Eigen::Matrix3d R_rel = T_relative.block<3, 3>(0, 0);
    Eigen::Vector3d t_rel = T_relative.block<3, 1>(0, 3);
    
    cout << "\n=== 车辆旋转180度后的相对变换 ===" << endl;
    cout << "从camera1到camera4的相对旋转:\n" << R_rel << endl;
    cout << "从camera1到camera4的相对平移: [" << t_rel.transpose() << "]米" << endl;
    
    // 选择输出尺寸（使用cam4的尺寸）
    output_size = cam4_params.imageSize;
    
    // 生成camera4的去畸变映射（作为参考）
    auto maps4 = UndistortFisheyeParam(cam4_params);
    map4_x = maps4.first;
    map4_y = maps4.second;
    
    // 为camera1生成考虑相对变换的映射
    // 将相对旋转用于rectify
    Mat R_rel_cv;
    cv::eigen2cv(R_rel, R_rel_cv);
    
    cv::fisheye::initUndistortRectifyMap(
        cam1_params.K, cam1_params.D, R_rel_cv,
        cam4_params.K,  // 使用cam4的内参作为目标
        output_size,
        CV_32FC1, map1_x, map1_y
    );
    
    cout << "输出尺寸: " << output_size << endl;
}

// 在图像上绘制网格，便于观察对齐效果
void drawGrid(Mat& img, int grid_size = 50) {
    // 绘制竖线
    for (int j = 0; j < img.cols; j += grid_size) {
        line(img, Point(j, 0), Point(j, img.rows), Scalar(0, 255, 0), 1);
    }
    // 绘制横线
    for (int i = 0; i < img.rows; i += grid_size) {
        line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1);
    }
    
    // 绘制中心十字
    line(img, Point(img.cols/2 - 50, img.rows/2), Point(img.cols/2 + 50, img.rows/2), 
         Scalar(0, 0, 255), 2);
    line(img, Point(img.cols/2, img.rows/2 - 50), Point(img.cols/2, img.rows/2 + 50), 
         Scalar(0, 0, 255), 2);
    
    // 在四个角绘制标记
    circle(img, Point(20, 20), 10, Scalar(255, 0, 0), 2);
    circle(img, Point(img.cols-20, 20), 10, Scalar(255, 0, 0), 2);
    circle(img, Point(20, img.rows-20), 10, Scalar(255, 0, 0), 2);
    circle(img, Point(img.cols-20, img.rows-20), 10, Scalar(255, 0, 0), 2);
}

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << "相机图像对齐测试程序" << endl;
    cout << "用于对齐camera_1和camera_4在车辆旋转180度时看到的相同场景" << endl;
    cout << "========================================" << endl;
    
    // 配置文件路径
    string config_file = "/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/config/cameras-inverse.cfg";
    
    // 加载camera_1的参数
    FisheyeCameraParams cam1_params;
    CameraExtrinsic cam1_ext;
    
    if (!LoadFisheyeCameraParams(config_file, "camera_1", cam1_params)) {
        cerr << "加载camera_1内参失败" << endl;
        return -1;
    }
    if (!ParseExtrinsic(config_file, "camera_1", cam1_ext)) {
        cerr << "解析camera_1外参失败" << endl;
        return -1;
    }
    
    // 加载camera_4的参数
    FisheyeCameraParams cam4_params;
    CameraExtrinsic cam4_ext;
    
    if (!LoadFisheyeCameraParams(config_file, "camera_4", cam4_params)) {
        cerr << "加载camera_4内参失败" << endl;
        return -1;
    }
    if (!ParseExtrinsic(config_file, "camera_4", cam4_ext)) {
        cerr << "解析camera_4外参失败" << endl;
        return -1;
    }
    
    // 打印参数
    printCameraExtrinsic("camera_1", cam1_ext, cam1_params);
    printCameraExtrinsic("camera_4", cam4_ext, cam4_params);
    
    // 生成重映射
    Mat map1_x, map1_y, map4_x, map4_y;
    Size output_size;
    
    cout << "\n计算重映射..." << endl;
    computeRemapForAlignment(cam1_params, cam1_ext, cam4_params, cam4_ext,
                             map1_x, map1_y, map4_x, map4_y, output_size);
    
    cout << "\n重映射参数计算完成！" << endl;
    
    // 如果提供了测试图像，进行处理
    if (argc >= 3) {
        string img1_path = argv[1];
        string img4_path = argv[2];
        
        Mat img1 = imread(img1_path);
        Mat img4 = imread(img4_path);
        
        if (img1.empty()) {
            cerr << "无法读取图像: " << img1_path << endl;
            return -1;
        }
        if (img4.empty()) {
            cerr << "无法读取图像: " << img4_path << endl;
            return -1;
        }
        
        cout << "\n处理图像..." << endl;
        cout << "camera1原始图像尺寸: " << img1.cols << "x" << img1.rows << endl;
        cout << "camera4原始图像尺寸: " << img4.cols << "x" << img4.rows << endl;
        
        // 应用重映射
        Mat img1_remapped, img4_remapped;
        remap(img1, img1_remapped, map1_x, map1_y, INTER_LINEAR);
        remap(img4, img4_remapped, map4_x, map4_y, INTER_LINEAR);
        
        // 在图像上绘制网格
        Mat img1_grid = img1_remapped.clone();
        Mat img4_grid = img4_remapped.clone();
        drawGrid(img1_grid);
        drawGrid(img4_grid);
        
        // 创建对比图
        Mat comparison_top, comparison_bottom;
        hconcat(img1_remapped, img4_remapped, comparison_top);
        hconcat(img1_grid, img4_grid, comparison_bottom);
        
        Mat comparison;
        vconcat(comparison_top, comparison_bottom, comparison);
        
        // 显示结果
        namedWindow("Camera1 (remapped)", WINDOW_NORMAL);
        namedWindow("Camera4 (remapped)", WINDOW_NORMAL);
        namedWindow("Comparison (上:原图 下:网格)", WINDOW_NORMAL);
        
        imshow("Camera1 (remapped)", img1_grid);
        imshow("Camera4 (remapped)", img4_grid);
        imshow("Comparison (上:原图 下:网格)", comparison);
        
        // 保存结果
        imwrite("camera1_remapped.jpg", img1_remapped);
        imwrite("camera4_remapped.jpg", img4_remapped);
        imwrite("camera1_remapped_grid.jpg", img1_grid);
        imwrite("camera4_remapped_grid.jpg", img4_grid);
        imwrite("comparison.jpg", comparison);
        
        cout << "\n结果已保存:" << endl;
        cout << "  - camera1_remapped.jpg (camera1处理后的图像)" << endl;
        cout << "  - camera4_remapped.jpg (camera4处理后的图像)" << endl;
        cout << "  - camera1_remapped_grid.jpg (带网格)" << endl;
        cout << "  - camera4_remapped_grid.jpg (带网格)" << endl;
        cout << "  - comparison.jpg (对比图)" << endl;
        
        cout << "\n按任意键退出..." << endl;
        waitKey(0);
    } else {
        cout << "\n未提供测试图像。" << endl;
        cout << "用法: " << argv[0] << " <camera1_image> <camera4_image>" << endl;
        cout << "\n说明:" << endl;
        cout << "  camera1_image: camera_1拍摄的图像路径" << endl;
        cout << "  camera4_image: 车辆旋转180度后camera_4拍摄的图像路径" << endl;
        cout << "\n示例:" << endl;
        cout << "  " << argv[0] << " /path/to/camera1.jpg /path/to/camera4.jpg" << endl;
    }
    
    return 0;
}
