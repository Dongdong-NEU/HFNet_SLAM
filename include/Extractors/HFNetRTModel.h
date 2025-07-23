#ifndef HFNETRTMODEL_H
#define HFNETRTMODEL_H

#include <string>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseModel.h"

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include "Extractors/TensorRTBuffers.h"
#endif // USE_TENSORRT

namespace ORB_SLAM3
{

#ifdef USE_TENSORRT

class RTLogger : public nvinfer1::ILogger
{
public:
    RTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING) : level(severity) {}

    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

    nvinfer1::ILogger::Severity level;
};
// d：一个void*类型的指针，用于初始化data成员变量。
// s：一个nvinfer1::Dims类型的对象，用于初始化shape成员变量。
class RTTensor
{
public:
    RTTensor(void* d, nvinfer1::Dims s) : data(d), shape(s) {} 
    void* data;
    nvinfer1::Dims shape;
};

class HFNetRTModel : public BaseModel
{
public:
    HFNetRTModel(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape);
    virtual ~HFNetRTModel(void) = default;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                         int nKeypointsNum, float threshold) override;

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override;

    bool IsValid(void) override { return mbVaild; }

    ModelType Type(void) override { return kHFNetTFModel; }
    /*nvinfer1::ICudaEngine 提供了一系列方法和功能，用于加载、编译、优化和执行深度学习模型。
    它可以从序列化的模型文件（如 ONNX、Caffe、TensorFlow 等）或动态构建的网络定义中创建。一
    旦创建了 nvinfer1::ICudaEngine 对象，就可以将输入数据提供给引擎并执行推理操作，获得输出
    结果。*/
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;

protected:

    bool LoadHFNetTRModel(void);

    void LoadTimingCacheFile(const std::string& strFileName, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    void UpdateTimingCacheFile(const std::string& strFileName, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    std::string DecideEigenFileName(const std::string& strEngineSaveDir, ModelDetectionMode mode, const nvinfer1::Dims4 inputShape);

    bool SaveEngineToFile(const std::string& strEngineSaveFile, const std::unique_ptr<nvinfer1::IHostMemory>& serializedEngine);

    bool LoadEngineFromFile(const std::string& strEngineSaveFile);

    void PrintInputAndOutputsInfo(std::unique_ptr<nvinfer1::INetworkDefinition>& network);

    bool Run(void);

    void GetLocalFeaturesFromTensor(const RTTensor &tScoreDense, const RTTensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, 
                                    int nKeypointsNum, float threshold);

    void GetGlobalDescriptorFromTensor(const RTTensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &mat, RTTensor &tensor);

    void Tensor2Mat(const RTTensor &tensor, cv::Mat &mat);

    void ResamplerRT(const RTTensor &data, const cv::Mat &warp, cv::Mat &output);

    nvinfer1::Dims4 mInputShape;
    ModelDetectionMode mMode;
    std::string mStrTRModelDir;
    std::string mStrONNXFile;
    std::string mStrCacheFile;
    bool mbVaild = false;
    RTLogger mLogger;
    std::unique_ptr<BufferManager> mpBuffers;
    std::vector<RTTensor> mvInputTensors;
    // mvOutputTensors[0]
    // mvOutputTensors[1]
    // mvOutputTensors[2]是全局描述子；
    std::vector<RTTensor> mvOutputTensors;
    // nvinfer1::IExecutionContext 是 NVIDIA TensorRT 库中的一个接口类，用于执行已编译的深度学习模型。
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
};

#else // USE_TENSORRT

class HFNetRTModel : public BaseModel
{
public:
    HFNetRTModel(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape)
    {
        std::cerr << "You must set USE_TENSORRT in CMakeLists.txt to enable tensorRT function." << std::endl;
        exit(-1);
    }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                        int nKeypointsNum, float threshold) override { return false; }

    virtual bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override { return false; }

    bool IsValid(void) override { return false; }

    ModelType Type(void) override { return kHFNetRTModel; }
};

#endif // USE_TENSORRT

} // namespace ORB_SLAM3

#endif // HFNETRTMODEL_H