#include "fisheye_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>

// 通用工具函数：提取一行中的数值
bool ExtractValue(const std::string& line, const std::string& key, double& out_val) {
    std::regex pattern(key + R"(\s*:\s*([-0-9.eE+]+))");
    std::smatch match;
    if (std::regex_search(line, match, pattern)) {
        out_val = std::stod(match[1]);
        return true;
    }
    return false;
}

bool ExtractValue(const std::string& line, const std::string& key, int& out_val) {
    std::regex pattern(key + R"(\s*:\s*([0-9]+))");
    std::smatch match;
    if (std::regex_search(line, match, pattern)) {
        out_val = std::stoi(match[1]);
        return true;
    }
    return false;
}

// 从配置文件中加载指定 camera_name 的相机参数
bool LoadFisheyeCameraParams(const std::string& cfgPath, const std::string& camera_name, FisheyeCameraParams& params) {
    std::ifstream fin(cfgPath);
    if (!fin.is_open()) {
        std::cerr << "failed to open config file: " << cfgPath << std::endl;
        return false;
    }

    std::string line;
    bool found_camera = false;
    std::regex cam_regex("camera_dev:\\s*\"" + camera_name + "\"");

    while (std::getline(fin, line)) {
        if (std::regex_search(line, cam_regex)) {
            found_camera = true;
            break;
        }
    }

    if (!found_camera) {
        std::cerr << "failed to find camera name: " << camera_name << std::endl;
        return false;
    }

    // 开始读取 intrinsic 区块
    bool in_intrinsic = false;
    double fx=0, fy=0, cx=0, cy=0;
    double k1=0, k2=0, k3=0, k4=0;
    int width=0, height=0;

    while (std::getline(fin, line)) {
        if (line.find("intrinsic {") != std::string::npos) {
            in_intrinsic = true;
            continue;
        }
        if (in_intrinsic && line.find("}") != std::string::npos) break;
        if (in_intrinsic) {
            ExtractValue(line, "img_width", width);
            ExtractValue(line, "img_height", height);
            ExtractValue(line, "f_x", fx);
            ExtractValue(line, "f_y", fy);
            ExtractValue(line, "o_x", cx);
            ExtractValue(line, "o_y", cy);
            ExtractValue(line, "k_1", k1);
            ExtractValue(line, "k_2", k2);
            ExtractValue(line, "k_3", k3);
            ExtractValue(line, "k_4", k4);
        }
    }

    if (width == 0 || height == 0) {
        std::cerr << "failed to read image size, width=" << width << " height=" << height << std::endl;
        return false;
    }

    params.K = (cv::Mat_<double>(3,3) << fx, 0, cx,
                                          0, fy, cy,
                                          0, 0, 1);
    params.D = (cv::Mat_<double>(4,1) << k1, k2, k3, k4);
    params.imageSize = cv::Size(width, height);
    return true;
}

std::pair<cv::Mat, cv::Mat> UndistortFisheyeParam(const FisheyeCameraParams& params, cv::Size size, double balance, double fov_offset_y, double fov_offset_x) {
    // balance参数控制去畸变的模式:
    // 0.0 - 保留所有源像素，边缘会有严重拉伸
    // 1.0 - 最小化黑色区域，但会裁剪掉边缘信息
    // 0.3-0.5 - 推荐的折中值，平衡视野和图像质量
    cv::Matx33d newK;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        params.K, params.D, size, cv::Matx33d::eye(), newK, balance);

    // 调整视野方向：修改新相机内参的光心位置
    // 增加cy：视野向上抬（减少底部内容）
    // 减少cy：视野向下看（减少顶部内容）
    // 增加cx：视野向右偏
    // 减少cx：视野向左偏
    if (fov_offset_y != 0.0 || fov_offset_x != 0.0) {
        newK(1, 2) += fov_offset_y;  // 调整cy（垂直方向）
        newK(0, 2) += fov_offset_x;  // 调整cx（水平方向）
        std::cout << "FOV adjusted: offset_y=" << fov_offset_y 
                  << ", offset_x=" << fov_offset_x << std::endl;
        std::cout << "New camera matrix (newK) after adjustment:\n" << cv::Mat(newK) << std::endl;
    }

    cv::Mat map1, map2;  

    // 使用目标尺寸size创建映射表
    cv::fisheye::initUndistortRectifyMap(params.K, params.D, cv::Matx33d::eye(), newK,
            size, CV_32FC1, map1, map2);

    return std::make_pair(map1, map2);
}

cv::Mat UndistortImage(const cv::Mat& image, const cv::Mat & Mapx, const cv::Mat & Mapy, const FisheyeCameraParams& params, const cv::Size& size) {

    cv::Mat undistorted;
    if (Mapx.empty() || Mapy.empty()) {
        std::cerr << "Error: Mapx or Mapy is empty!" << std::endl;
        return undistorted; // 返回空矩阵
    }
    cv::remap(image, undistorted, Mapx, Mapy, cv::INTER_LINEAR);

    float cx = params.K.at<double>(0, 2);
    float cy = params.K.at<double>(1, 2);
    int x = cx - size.width / 2;
    int y = cy - size.height / 2;
    
    // 确保裁剪区域在图像范围内
    x = std::max(0, x);
    y = std::max(0, y);
    x = std::min(x, undistorted.cols - size.width);
    y = std::min(y, undistorted.rows - size.height);
    
    cv::Rect crop_roi(x, y, size.width, size.height);
    cv::Mat undistorted_cropped = undistorted(crop_roi).clone();
    cv::Mat undistorted_cropped_resized;
    cv::resize(undistorted_cropped, undistorted_cropped_resized, cv::Size(400, 300), cv::INTER_AREA);
    
    // cv::imshow("undistorted", undistorted);
    // cv::imshow("undistorted_cropped", undistorted_cropped);
    // cv::imshow("undistorted_cropped_resized", undistorted_cropped_resized);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return undistorted_cropped_resized;
}


cv::Mat UndistortFisheyeImageMy(const cv::Mat& distorted, const FisheyeCameraParams& params, double balance , const cv::Rect& crop_roi) {
    std::cerr << "Error: UndistortFisheyeImageMy is not implemented!" << std::endl;
    return cv::Mat();
    cv::Mat input = distorted;
    cv::Size remap_size = cv::Size(1920,1080); 
    // 靠fx fy 来裁剪区域，靠cx cy来控制视野范围
    cv::Mat new_intrinsic_matrix = (cv::Mat_<double>(3, 3) << 600, 0, 960,
                                    0, 600, 720,
                                    0, 0, 1);
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(params.K, params.D, cv::Matx33d::eye(), new_intrinsic_matrix,
            remap_size, CV_32FC1, map1, map2);
     cv::Mat undistorted;
    cv::remap(input, undistorted, map1, map2, cv::INTER_CUBIC);
    return undistorted;
}
