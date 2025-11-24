#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "../include/Extractors/fisheye_utils.h"

using namespace std;
using namespace cv;

// 简化版的 LoadConfigYaml，只读取图像尺寸
bool LoadConfigYaml(const string &configPath, cv::Size &ImSize) {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << configPath << std::endl;
        return false;
    }
    
    int height = (int)fs["height"];
    int width = (int)fs["width"];
    
    if (height <= 0 || width <= 0) {
        std::cerr << "Invalid height or width in config file." << std::endl;
        return false;
    }
    
    ImSize = cv::Size(width, height);
    fs.release();
    return true;
}


int main(int argc, char** argv)
{
    if (argc != 5 && argc != 9) {
        std::cerr << "Usage: test_distort_image <image_path> <camera_name> <camera_cfg> <config_yaml> [crop_x crop_y crop_width crop_height]" << std::endl;
        std::cerr << "Example: ./test_distort_image /path/to/image.png camera_1 config/cameras.cfg config/config.yaml" << std::endl;
        std::cerr << "Example with crop: ./test_distort_image /path/to/image.png camera_1 config/cameras.cfg config/config.yaml 100 100 800 600" << std::endl;
        return -1;
    }

    const string strImagePath = string(argv[1]);
    const string strCameraName = string(argv[2]);
    const string strCamerasCfgPath = string(argv[3]);
    const string strConfigYamlPath = string(argv[4]);
    
    // 可选的裁切参数
    bool enableCrop = (argc == 9);
    cv::Rect cropROI;
    if (enableCrop) {
        int crop_x = std::stoi(argv[5]);
        int crop_y = std::stoi(argv[6]);
        int crop_w = std::stoi(argv[7]);
        int crop_h = std::stoi(argv[8]);
        cropROI = cv::Rect(crop_x, crop_y, crop_w, crop_h);
        std::cout << "Crop enabled: x=" << crop_x << ", y=" << crop_y 
                  << ", w=" << crop_w << ", h=" << crop_h << std::endl;
    }

    FisheyeCameraParams camParams;
    if (!LoadFisheyeCameraParams(strCamerasCfgPath, strCameraName, camParams)) {
        std::cerr << "Failed to load camera intrinsics for " << strCameraName << " from: " << strCamerasCfgPath << std::endl;
        return -1;
    }

    std::cout << "\n=== Camera Parameters ===" << std::endl;
    std::cout << "Camera: " << strCameraName << std::endl;
    std::cout << "Image size: " << camParams.imageSize.width << "x" << camParams.imageSize.height << std::endl;
    std::cout << "Intrinsics K:\n" << camParams.K << std::endl;
    std::cout << "Distortion D (shape: " << camParams.D.rows << "x" << camParams.D.cols << "):\n" << camParams.D << std::endl;

    // 生成去畸变映射表（使用鱼眼相机去畸变函数）
    std::cout << "\n=== Generating Undistortion Maps ===" << std::endl;
    std::cout << "Using KANNALA_BRANDT fisheye undistortion model..." << std::endl;
    
    // 参数说明：
    // balance: 1.0 = 去除黑边，最小化无效区域
    // fov_offset_y: 150.0 = 视野向上抬150像素，减少底部车辆前盖
    // fov_offset_x: 0.0 = 不进行水平偏移
    double balance = 1.0;
    double fov_offset_y = 100.0;  // 正值=向上抬，可根据实际效果调整（建议范围: 50-300）
    double fov_offset_x = 0.0;
    
    std::cout << "Parameters: balance=" << balance 
              << ", fov_offset_y=" << fov_offset_y 
              << ", fov_offset_x=" << fov_offset_x << std::endl;
    
    auto t1 = chrono::steady_clock::now();
    pair<cv::Mat, cv::Mat> undistortMaps = UndistortFisheyeParam(camParams, cv::Size(1920, 1440), balance, fov_offset_y, fov_offset_x);
    auto t2 = chrono::steady_clock::now();
    auto time_map = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    std::cout << "Map generation time: " << time_map << " ms" << std::endl;

    // 读取图像
    std::cout << "\n=== Loading Image ===" << std::endl;
    cv::Mat image = imread(strImagePath, IMREAD_COLOR);
    // cv::Size imageSize = cv::Size(1920, 1080);
    // cv::resize(image, image, imageSize);
    if (image.empty()) {
        std::cerr << "Failed to load image from: " << strImagePath << std::endl;
        return -1;
    }
    std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;


    // cv::Mat undistorted_full_new = UndistortFisheyeImageNew(image, camParams);
    // cv::imshow("undistorted_full_new", undistorted_full_new);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // 执行去畸变
    std::cout << "\n=== Undistorting Image ===" << std::endl;
    auto t3 = chrono::steady_clock::now();
    cv::Mat undistorted_full = UndistortImage(image, undistortMaps.first, undistortMaps.second, camParams, cv::Size(960, 720));
    auto t4 = chrono::steady_clock::now();
    auto time_undistort = chrono::duration_cast<chrono::milliseconds>(t4 - t3).count();
    string path = "/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/debug_image/fisheye_test/";
    cv::imwrite(path + "undistorted_fisheye_front.png", undistorted_full);
    

    return 0;
}
