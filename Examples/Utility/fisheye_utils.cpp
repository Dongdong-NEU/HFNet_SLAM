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

std::pair<cv::Mat, cv::Mat> UndistortFisheyeParam(const FisheyeCameraParams& params) {
    cv::Size size = cv::Size(1920, 1080); 
    cv::Matx33d newK;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        params.K, params.D, size, cv::Matx33d::eye(), newK, 0);

    cv::Mat map1, map2;  

    cv::fisheye::initUndistortRectifyMap(params.K, params.D, cv::Matx33d::eye(), newK,
            params.imageSize, CV_32FC1, map1, map2);

    return std::make_pair(map1, map2);
}

cv::Mat UndistortImage(const cv::Mat& image, const cv::Mat & Mapx, const cv::Mat & Mapy, const FisheyeCameraParams& params, const cv::Size& size) {

    float w = static_cast<float>(size.width);
    float h = static_cast<float>(size.height);
    float cx = params.K.at<double>(0, 2);
    float cy = params.K.at<double>(1, 2);
    int x = cx - w / 2;
    int y = cy - h / 2;
    cv::Rect crop_roi = cv::Rect(x, y, w, h);

    cv::Mat undistorted;
    if (Mapx.empty() || Mapy.empty()) {
        std::cerr << "Error: Mapx or Mapy is empty!" << std::endl;
        return undistorted; // 返回空矩阵
    }
    cv::remap(image, undistorted, Mapx, Mapy, cv::INTER_LINEAR);

    if (crop_roi.width > 0 && crop_roi.height > 0 && 
        crop_roi.x >= 0 && crop_roi.y >= 0 && 
        crop_roi.x + crop_roi.width <= undistorted.cols && 
        crop_roi.y + crop_roi.height <= undistorted.rows) {
        cv::Mat cropped = undistorted(crop_roi).clone();
        return cropped;
    }
    return undistorted;
}


cv::Mat UndistortFisheyeImageMy(const cv::Mat& distorted, const FisheyeCameraParams& params, double balance , const cv::Rect& crop_roi) {
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
