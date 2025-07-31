#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct FisheyeCameraParams {
    cv::Mat K; // 内参矩阵 3x3
    cv::Mat D; // 畸变参数 4x1
    cv::Size imageSize;
};


// 解析相机参数配置文件，仅加载指定相机名（如 panoramic_1）
bool LoadFisheyeCameraParams(const std::string& cfgPath, const std::string& camera_name, FisheyeCameraParams& params);

// 鱼眼去畸变
std::pair<cv::Mat, cv::Mat> UndistortFisheyeParam(const FisheyeCameraParams& params);
cv::Mat UndistortImage(const cv::Mat& image, const cv::Mat & Mapx, const cv::Mat & Mapy, const FisheyeCameraParams& params, const cv::Size& size);
cv::Mat UndistortFisheyeImageMy(const cv::Mat& distorted, const FisheyeCameraParams& params, double balance , const cv::Rect& crop_roi);

