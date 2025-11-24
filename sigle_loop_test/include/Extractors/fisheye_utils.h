#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct FisheyeCameraParams {
    cv::Mat K; 
    cv::Mat D;
    cv::Size imageSize;
};


bool LoadFisheyeCameraParams(const std::string& cfgPath, const std::string& camera_name, FisheyeCameraParams& params);

// 鱼眼去畸变
// balance参数: 0.0=保留所有像素(边缘扭曲严重), 1.0=去除黑边(损失视野), 0.3-0.5=推荐折中值
// fov_offset_y: 视野Y方向偏移，正值=向上抬，负值=向下看，单位像素，建议范围[-300, 300]
// fov_offset_x: 视野X方向偏移，正值=向右偏，负值=向左偏，单位像素
std::pair<cv::Mat, cv::Mat> UndistortFisheyeParam(const FisheyeCameraParams& params, cv::Size size, double balance = 0.5, double fov_offset_y = 0.0, double fov_offset_x = 0.0);
cv::Mat UndistortImage(const cv::Mat& image, const cv::Mat & Mapx, const cv::Mat & Mapy, const FisheyeCameraParams& params, const cv::Size& size);
cv::Mat UndistortFisheyeImageMy(const cv::Mat& distorted, const FisheyeCameraParams& params, double balance , const cv::Rect& crop_roi);

