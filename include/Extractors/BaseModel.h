#ifndef BASEMODEL_H
#define BASEMODEL_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{
// 枚举使用模型类型
enum ModelType {
    kHFNetTFModel,
    kHFNetRTModel,
    kHFNetVINOModel,
};
// 检测描述子类型
enum ModelDetectionMode {
    kImageToLocalAndGlobal,  // 局部描述子 与 全局描述子
    kImageToLocal,           // 局部描述子
    kImageToLocalAndIntermediate, // 中间局部描述子？
    kIntermediateToGlobal // 中间描述子-》全局描述子？
};

const std::string gStrModelDetectionName[] = {"ImageToLocalAndGlobal", "ImageToLocal", 
                                              "ImageToLocalAndIntermediate", "IntermediateToGlobal"};
// 提取特征点类
class ExtractorNode
{
public:
    // 默认bNoMore为false
    ExtractorNode():bNoMore(false){}
    // 划分节点？
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
    // 特征点容器
    std::vector<cv::KeyPoint> vKeys;
    // 特征点边界
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    // ？
    bool bNoMore;
};

class BaseModel
{
public:
    virtual ~BaseModel(void) = default;
    // 检测特征点函数抽象类，提取全局描述子；
    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints,
                        cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) = 0;
    // 检测特征点函数抽象类，提取局部描述子；
    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, 
                        cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) = 0;
    // 将中间量转化成全局描述子？
    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) = 0;

    // 是否有效？
    virtual bool IsValid(void) = 0;
    // 检测类型，如果开启RT条件编译，那么返回数值是RTModle
    virtual ModelType Type(void) = 0;
};


class Settings;

void InitAllModels(Settings* settings);

void InitAllModels(const std::string& strModelPath, ModelType modelType, cv::Size ImSize, int nLevels, float scaleFactor);

std::vector<BaseModel*> GetModelVec(void);

BaseModel* GetGlobalModel(void);

BaseModel* InitTFModel(const std::string& strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

BaseModel* InitRTModel(const std::string& strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

BaseModel* InitVINOModel(const std::string &strModelPath, ModelDetectionMode mode, cv::Vec4i inputShape);

std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int minX,
                                           const int maxX, const int minY, const int maxY, const int N);

std::vector<cv::KeyPoint> NMS(const std::vector<cv::KeyPoint> &vToDistributeKeys, int width, int height, int radius);

void Resampler(const float* data, const float* warp, float* output,
                const int batch_size, const int data_height, 
                const int data_width, const int data_channels, const int num_sampling_points);

} // namespace ORB_SLAM3

#endif