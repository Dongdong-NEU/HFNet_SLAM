#ifndef EIGENPLACES_EXTRACTOR_H
#define EIGENPLACES_EXTRACTOR_H

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../../EigenPlaces/tensorrt_engine.h"
#include "../../EigenPlaces/image_processor.h"

namespace DeepRoute
{

/**
 * EigenPlaces特征提取器
 * 封装EigenPlaces TensorRT推理引擎，提供与HFNet兼容的接口
 */
class EigenPlacesExtractor
{
public:
    /**
     * 构造函数
     * @param onnx_model_path ONNX模型文件路径
     * @param engine_cache_path TensorRT引擎缓存文件路径（可选）
     * @param input_size 输入图像尺寸（默认512x512）
     * @param use_nhwc_layout 是否使用NHWC布局（自动检测或手动指定）
     */
    EigenPlacesExtractor(const std::string& onnx_model_path, 
                        cv::Size input_size,
                        const std::string& engine_cache_path = "",
                        bool use_nhwc_layout = false);
    
    /**
     * 析构函数
     */
    ~EigenPlacesExtractor();
    
    /**
     * 初始化提取器
     * @return 是否成功初始化
     */
    bool Initialize();
    
    /**
     * 提取全局描述符（兼容HFNet接口）
     * @param image 输入图像（灰度或彩色）
     * @param globalDescriptors 输出的全局描述符
     * @return 是否成功提取
     */
    bool ExtractGlobalDescriptor(const cv::Mat& image, cv::Mat& globalDescriptors);
    
    /**
     * 检查提取器是否有效
     * @return 是否有效
     */
    bool IsValid() const { return is_valid_; }

private:
    std::string onnx_model_path_;
    std::string engine_cache_path_;
    cv::Size input_size_;
    bool use_nhwc_layout_;
    bool is_valid_;
    
    std::unique_ptr<TensorRTEngine> engine_;
    std::unique_ptr<ImageProcessor> processor_;
    
    /**
     * 检测ONNX模型的输入格式（NCHW或NHWC）
     * @return true表示NHWC格式，false表示NCHW格式
     */
    bool detectInputLayout();
    
    /**
     * 将OpenCV Mat转换为EigenPlaces需要的格式
     */
    bool preprocessImage(const cv::Mat& image, std::vector<float>& input_data);
    
    /**
     * 将EigenPlaces输出转换为OpenCV Mat格式
     */
    bool postprocessDescriptor(const std::vector<float>& raw_output, cv::Mat& descriptor);
};

} // namespace DeepRoute

#endif // EIGENPLACES_EXTRACTOR_H
