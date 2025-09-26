#include "../../include/Extractors/EigenPlacesExtractor.h"
#include <iostream>
#include <chrono>

namespace DeepRoute
{

EigenPlacesExtractor::EigenPlacesExtractor(const std::string& onnx_model_path, 
                                         cv::Size input_size,
                                         const std::string& engine_cache_path)
    : onnx_model_path_(onnx_model_path)
    , engine_cache_path_(engine_cache_path)
    , input_size_(input_size)
    , is_valid_(false)
{
}

EigenPlacesExtractor::~EigenPlacesExtractor()
{
}

bool EigenPlacesExtractor::Initialize()
{
    try {
        // 初始化TensorRT引擎
        engine_ = std::make_unique<TensorRTEngine>(onnx_model_path_, engine_cache_path_, 1, true);
        if (!engine_->initialize()) {
            std::cerr << "Failed to initialize TensorRT engine for EigenPlaces" << std::endl;
            return false;
        }
        
        // 初始化图像处理器
        processor_ = std::make_unique<ImageProcessor>(
            input_size_.height,  // 高度
            input_size_.width    // 宽度
        );
        
        is_valid_ = true;
        std::cout << "EigenPlaces extractor initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during EigenPlaces initialization: " << e.what() << std::endl;
        is_valid_ = false;
        return false;
    }
}

bool EigenPlacesExtractor::ExtractGlobalDescriptor(const cv::Mat& image, cv::Mat& globalDescriptors)
{
    if (!is_valid_) {
        std::cerr << "EigenPlaces extractor not initialized" << std::endl;
        return false;
    }
    
    if (image.empty()) {
        std::cerr << "Input image is empty" << std::endl;
        return false;
    }
    
    try {
        // 预处理图像
        std::vector<float> input_data;
        if (!preprocessImage(image, input_data)) {
            std::cerr << "Failed to preprocess image" << std::endl;
            return false;
        }
        
        // 执行推理
        std::vector<float> raw_output;
        if (!engine_->infer(input_data, raw_output)) {
            std::cerr << "Failed to run inference" << std::endl;
            return false;
        }
        
        // 后处理描述符
        if (!postprocessDescriptor(raw_output, globalDescriptors)) {
            std::cerr << "Failed to postprocess descriptor" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during feature extraction: " << e.what() << std::endl;
        return false;
    }
}

bool EigenPlacesExtractor::preprocessImage(const cv::Mat& image, std::vector<float>& input_data)
{
    try {
        // 如果输入是灰度图，转换为RGB
        cv::Mat rgb_image;
        if (image.channels() == 1) {
            cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 3) {
            // OpenCV使用BGR，需要转换为RGB
            cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        } else {
            std::cerr << "Unsupported image format: " << image.channels() << " channels" << std::endl;
            return false;
        }
        
        // 使用ImageProcessor进行预处理
        return processor_->preprocessImage(rgb_image, input_data);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during image preprocessing: " << e.what() << std::endl;
        return false;
    }
}

bool EigenPlacesExtractor::postprocessDescriptor(const std::vector<float>& raw_output, cv::Mat& descriptor)
{
    try {
        // 使用ImageProcessor进行L2归一化
        std::vector<float> normalized_descriptor;
        processor_->postprocessDescriptor(raw_output, normalized_descriptor);
        
        // 转换为OpenCV Mat格式 (1 x descriptor_size, CV_32F)
        descriptor = cv::Mat(1, normalized_descriptor.size(), CV_32F);
        std::memcpy(descriptor.data, normalized_descriptor.data(), 
                   normalized_descriptor.size() * sizeof(float));
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during descriptor postprocessing: " << e.what() << std::endl;
        return false;
    }
}

} // namespace DeepRoute
