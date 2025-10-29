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

cv::Mat CropImage(const cv::Mat& image,int top_cut, int bottom_cut, int side_cut_left, int side_cut_right)
{
    if (image.empty()) {
        std::cerr << "Error: input image is empty!" << std::endl;
        return cv::Mat();
    }

    // Step 2: 定义ROI区域
    cv::Rect roi(side_cut_left, top_cut, side_cut_right, bottom_cut);

    // 边界检查（防止越界）
    roi &= cv::Rect(0, 0, image.cols, image.rows);

    cv::Mat image_cropped = image(roi).clone();
    cv::resize(image_cropped, image_cropped, cv::Size(640, 360));

    cv::imshow("image_cropped", image_cropped);
    cv::imshow("image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    
    return image_cropped;
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

    // 读取目标输出尺寸
    cv::Size ImSizeFinal;
    LoadConfigYaml(strConfigYamlPath, ImSizeFinal);
    std::cout << "Target output size: " << ImSizeFinal.width << "x" << ImSizeFinal.height << std::endl;

    // 加载相机参数
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
    auto t1 = chrono::steady_clock::now();
    pair<cv::Mat, cv::Mat> undistortMaps = UndistortFisheyeParam(camParams);
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


    cv::Mat image_camera1 = imread("/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera1_test/camera1/1747822690.045384.png", IMREAD_COLOR);
    cv::Mat image_camera4 = imread("/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera1_test/camera4/1747822793.566925.png", IMREAD_COLOR);

    cv::Mat image_camera1_crop = CropImage(image_camera1, 0,1440,640,1920);
    cv::Mat image_camera4_crop = CropImage(image_camera4, 0, 720, 320, 1280);
    cv::imwrite("/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera1_test/image_camera1_crop.png", image_camera1_crop);
    cv::imwrite("/home/xihuidong/codetree/repo/visual_mapping_test/avp-15/camera1_test/image_camera4_crop.png", image_camera4_crop);




    
    // 执行裁切（如果启用）
    if (enableCrop) {
        std::cout << "\n=== Cropping Image ===" << std::endl;
        
        // 检查裁切区域是否有效
        if (cropROI.x < 0 || cropROI.y < 0 || 
            cropROI.x + cropROI.width > image.cols || 
            cropROI.y + cropROI.height > image.rows) {
            std::cerr << "Error: Crop ROI is out of image bounds!" << std::endl;
            std::cerr << "  Image size: " << image.cols << "x" << image.rows << std::endl;
            std::cerr << "  Crop ROI: x=" << cropROI.x << ", y=" << cropROI.y 
                      << ", w=" << cropROI.width << ", h=" << cropROI.height << std::endl;
            return -1;
        }
        
        cv::Mat originalImage = image.clone();  // 保存原始图像用于对比
        image = image(cropROI).clone();
        std::cout << "Cropped image size: " << image.cols << "x" << image.rows << std::endl;
        
        // 保存裁切后的图像
        string cropOutputPath = strImagePath.substr(0, strImagePath.find_last_of('.')) + "_cropped.png";
        cv::imwrite(cropOutputPath, image);
        std::cout << "Saved cropped image to: " << cropOutputPath << std::endl;
    }

    // 检查图像尺寸是否匹配
    // if (image.cols != camParams.imageSize.width || image.rows != camParams.imageSize.height) {
    //     std::cerr << "Warning: Image size mismatch!" << std::endl;
    //     std::cerr << "  Expected: " << camParams.imageSize.width << "x" << camParams.imageSize.height << std::endl;
    //     std::cerr << "  Got: " << image.cols << "x" << image.rows << std::endl;
    // }

    // 执行去畸变
    std::cout << "\n=== Undistorting Image ===" << std::endl;
    auto t3 = chrono::steady_clock::now();
    cv::Mat undistorted_full = UndistortImage(image, undistortMaps.first, undistortMaps.second, camParams, ImSizeFinal);
    auto t4 = chrono::steady_clock::now();
    auto time_undistort = chrono::duration_cast<chrono::milliseconds>(t4 - t3).count();
    
    if (undistorted_full.empty()) {
        std::cerr << "Undistortion failed!" << std::endl;
        return -1;
    }
    
    std::cout << "Full undistorted size: " << undistorted_full.cols << "x" << undistorted_full.rows << std::endl;
    
    // 手动裁剪到目标尺寸（以光心为中心）
    float cx = camParams.K.at<double>(0, 2);
    float cy = camParams.K.at<double>(1, 2);
    int x = cx - ImSizeFinal.width / 2;
    int y = cy - ImSizeFinal.height / 2;
    
    // 确保裁剪区域在图像范围内
    x = std::max(0, x);
    y = std::max(0, y);
    x = std::min(x, undistorted_full.cols - ImSizeFinal.width);
    y = std::min(y, undistorted_full.rows - ImSizeFinal.height);
    
    cv::Rect crop_roi(x, y, ImSizeFinal.width, ImSizeFinal.height);
    cv::Mat undistorted = undistorted_full(crop_roi).clone();
    
    std::cout << "Crop ROI: x=" << x << ", y=" << y << ", w=" << ImSizeFinal.width << ", h=" << ImSizeFinal.height << std::endl;
    std::cout << "Undistortion time: " << time_undistort << " ms" << std::endl;
    std::cout << "Output image size: " << undistorted.cols << "x" << undistorted.rows << std::endl;

    // 保存结果
    string outputPath = strImagePath.substr(0, strImagePath.find_last_of('.')) + "_undistorted.png";
    cv::imwrite(outputPath, undistorted);
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Saved undistorted image to: " << outputPath << std::endl;

    // 显示对比
    std::cout << "\n=== Displaying Comparison ===" << std::endl;
    std::cout << "Press any key to close windows..." << std::endl;
    
    // 调整原图尺寸以便对比显示
    cv::Mat original_resized;
    cv::resize(image, original_resized, ImSizeFinal);
    
    // 并排显示
    cv::Mat comparison;
    cv::hconcat(original_resized, undistorted, comparison);
    
    // 添加标注
    cv::putText(comparison, "Original (Distorted)", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(comparison, "Undistorted", cv::Point(ImSizeFinal.width + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Comparison", cv::WINDOW_NORMAL);
    cv::resizeWindow("Comparison", 1280, 480);
    cv::imshow("Comparison", comparison);
    
    // 单独显示原图和去畸变图
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Undistorted", cv::WINDOW_NORMAL);
    cv::imshow("Original", original_resized);
    cv::imshow("Undistorted", undistorted);
    
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Camera: " << strCameraName << std::endl;
    std::cout << "Model: KANNALA_BRANDT (Fisheye)" << std::endl;
    std::cout << "Input size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Output size: " << undistorted.cols << "x" << undistorted.rows << std::endl;
    std::cout << "Processing time: " << time_undistort << " ms" << std::endl;
    std::cout << "\nDone!" << std::endl;

    return 0;
}
