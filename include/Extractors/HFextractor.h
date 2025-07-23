#ifndef HFNETEXTRACTOR_H
#define HFNETEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseModel.h"

namespace ORB_SLAM3
{

class BaseModel;

class HFextractor
{
public:
    
    HFextractor(int nfeatures, float threshold, BaseModel* pModels);
    // 特征点提取器 特征点个数 阈值 尺度 金字塔层数 使用的模型
    HFextractor(int nfeatures, float threshold, float scaleFactor, 
                int nlevels, const std::vector<BaseModel*>& vpModels);

    ~HFextractor(){}

    // Compute the features and descriptors on an image.
    int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint>& _keypoints,
                   cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors);
    // 获取金字塔层数
    int inline GetLevels(void) {
        return nlevels;}
    // 获取缩放因子
    float inline GetScaleFactor(void) {
        return scaleFactor;}
    // 获取每一层的缩放因子
    std::vector<float> inline GetScaleFactors(void) {
        return mvScaleFactor;
    }
    // 获取缩放因子的倒数
    std::vector<float> inline GetInverseScaleFactors(void) {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(void) {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(void) {
        return mvInvLevelSigma2;
    }
    // 金字塔图像容器；
    std::vector<cv::Mat> mvImagePyramid;
    // 每一层金字塔内的特征点；
    std::vector<int> mnFeaturesPerLevel;
    
    int nfeatures;
    // 响应值；
    float threshold;
    // basemodle模型
    std::vector<BaseModel*> mvpModels;

protected:

    double scaleFactor;
    int nlevels;
    bool bUseOctTree;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<int> umax;

    void ComputePyramid(const cv::Mat &image);
    // 单独在一层上提取特征点
    int ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                           cv::Mat &localDescriptors, cv::Mat &globalDescriptors);
    // 在多层上提取特征点
    int ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                           cv::Mat &localDescriptors, cv::Mat &globalDescriptors);
    // 并行在多层金字塔上提取特征点
    int ExtractMultiLayersParallel(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                                   cv::Mat &localDescriptors, cv::Mat &globalDescriptors);
};

} //namespace ORB_SLAM

#endif
