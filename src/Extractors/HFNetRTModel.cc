#include "Extractors/HFNetRTModel.h"

namespace ORB_SLAM3
{

#ifdef USE_TENSORRT
#include <fstream>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace nvinfer1;

void RTLogger::log(Severity severity, AsciiChar const* msg) noexcept
{
    if (severity > level) return;

    using namespace std;
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING";
            break;
        case Severity::kINFO:
            std::cerr << "INFO";
            break;
        case Severity::kVERBOSE:
            std::cerr << "VERBOSE";
            break;
    }
    std::cerr << ": " << msg << endl;
}

// 在这里完成特征点和描述子检测的前奏
HFNetRTModel::HFNetRTModel(const std::string &strModelDir, ModelDetectionMode mode, 
                           const cv::Vec4i inputShape)
{
    mStrTRModelDir = strModelDir + "/";
    mMode = mode;
    mInputShape = {inputShape(0), inputShape(1), inputShape(2), inputShape(3)};
    mStrONNXFile = mStrTRModelDir + "HF-Net.onnx";
    mStrCacheFile = mStrTRModelDir + "HF-Net.cache";
    /// 加载模型，若加载成功则返回true，否则返回false ,若没有加载则中断程序 ///
    mbVaild = LoadHFNetTRModel();

    if (!mbVaild) return;
    // 使用 new 运算符创建一个 BufferManager 对象，并通过 
    // std::shared_ptr 的 reset 函数将其分配给 mpBuffers。
    // 通过mpBuffers管理mEngine？
    mpBuffers.reset(new BufferManager(mEngine));

    // ? 怎么分辨要检测特征点还是局部描述子还是全局描述子？
    if (mMode == kImageToLocalAndGlobal)
    {
        // 在该代码块中，首先使用 mpBuffers 对象的 getHostBuffer 函数获取名为 "image:0" 
        // 的数据，并使用 mEngine 对象的 getTensorShape 函数获取 "image:0" 张量的
        // 形状。然后，将获取到的主机缓冲区和张量形状作为参数，通过 emplace_back 函数将它们作
        // 为一个 RTTensor对象添加到 mvInputTensors 容器中。
        mvInputTensors.emplace_back(mpBuffers->getHostBuffer("image:0"), 
                                    mEngine->getTensorShape("image:0"));
        // 接下来，使用相同的方式，将名为 "scores_dense_nms:0" 的主机缓冲区和张量形状添加到
        // mvOutputTensors 容器中。
        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("scores_dense_nms:0"), 
                                     mEngine->getTensorShape("scores_dense_nms:0"));
        // 然后，将名为 "local_descriptor_map:0" 的主机缓冲区和张量形状添加到 mvOutputTensors 容器中。
        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("local_descriptor_map:0"), 
                                     mEngine->getTensorShape("local_descriptor_map:0"));
        // 将名为 "global_descriptor:0" 的主机缓冲区和张量形状添加到 mvOutputTensors 容器中。
        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("global_descriptor:0"),
                                     mEngine->getTensorShape("global_descriptor:0"));
    }
    else if (mMode == kImageToLocal)
    {
        mvInputTensors.emplace_back(mpBuffers->getHostBuffer("image:0"), 
                                    mEngine->getTensorShape("image:0"));

        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("scores_dense_nms:0"), 
                                     mEngine->getTensorShape("scores_dense_nms:0"));

        mvOutputTensors.emplace_back(mpBuffers->getHostBuffer("local_descriptor_map:0"), 
                                     mEngine->getTensorShape("local_descriptor_map:0"));
    }
    else if (mMode == kImageToLocalAndIntermediate || mMode == kIntermediateToGlobal)
    {
        mbVaild = false; // not supported
        return;
    }
    else
    {
        mbVaild = false;
        return;
    }

    mbVaild = true;
}

bool HFNetRTModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, 
                          cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                          int nKeypointsNum, float threshold)
{
    // 如果检测特征点的模式为检测局部与全局 或者 为检测局部与中间变量（中间变量就是将检测后的tensor转化成Mat）
    if (mMode != kImageToLocalAndGlobal && mMode != kImageToLocalAndIntermediate) return false;
    // 将Mat转化为CV_32F tensor类型的数据；
    Mat2Tensor(image, mvInputTensors[0]);
    // 如果模型没有被正常加载，mvInputTensors为空，的话，停止；
    // 然后进行检测
    if (!Run()) return false;
    
    // 根据mode模式的不同选择不同的内容（需要满足阈值）进行复制；
    // 如果运行模式是检测局部与全局描述子的话，执行GetGlobalDescriptorFromTensor；
    if (mMode == kImageToLocalAndGlobal)
        // 将mvOutputTensors中内容复制到globalDescriptors中
        GetGlobalDescriptorFromTensor(mvOutputTensors[2], globalDescriptors);
    // 将mvOutputTensors[2]转换成Mat
    else Tensor2Mat(mvOutputTensors[2], globalDescriptors);
    // 获得局部特征点，筛选分数大于一定阈值的特征点，对描述子进行重采样操作；
    GetLocalFeaturesFromTensor(mvOutputTensors[0], mvOutputTensors[1], vKeyPoints,
                               localDescriptors, nKeypointsNum, threshold);
    return true;
}

bool HFNetRTModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                            int nKeypointsNum, float threshold)
{
    if (mMode != kImageToLocal) return false;

    Mat2Tensor(image, mvInputTensors[0]);

    if (!Run()) return false;
    GetLocalFeaturesFromTensor(mvOutputTensors[0], mvOutputTensors[1], vKeyPoints, localDescriptors, nKeypointsNum, threshold);
    return true;
}

bool HFNetRTModel::Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors)
{
    if (mMode != kIntermediateToGlobal) return false;

    Mat2Tensor(intermediate, mvInputTensors[0]);
    if (!Run()) return false;
    GetGlobalDescriptorFromTensor(mvOutputTensors[0], globalDescriptors);
    return true;
}

bool HFNetRTModel::Run(void)
{
    if (!mbVaild) return false;
    if (mvInputTensors.empty()) return false;

    /*
    接下来，将主机端的输入数据复制到设备端的输入缓冲区中，使用 mpBuffers->copyInputToDevice() 
    函数完成该操作后，调用 mContext->executeV2(mpBuffers->getDeviceBindings().data()) 
    执行模型的推理过程。该函数将设备端的输入缓冲区绑定到执行上下文 (mContext)，并执行推理操作。返回
    值 status 表示执行的状态，如果执行失败，则返回 false。最后，将设备端的输出缓冲区的数据复制到主
    机端的输出缓冲区中，使用 mpBuffers->copyOutputToHost() 函数完成该操作。
    */

    // Memcpy from host input buffers to device input buffers
    mpBuffers->copyInputToDevice();
    // 将设备上的数据传递给模型进行输出；
    bool status = mContext->executeV2(mpBuffers->getDeviceBindings().data());
    if (!status) return false;

    // Memcpy from device output buffers to host output buffers
    // 将模型输出的数据(同步地)拷贝到主机上；
    mpBuffers->copyOutputToHost();

    return true;
}

void HFNetRTModel::GetLocalFeaturesFromTensor(
    const RTTensor &tScoreDense, const RTTensor &tDescriptorsMap,
    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, 
    int nKeypointsNum, float threshold)
{   // output1 这个是检测后的特征点的得分；
    auto vResScoresDense = static_cast<float*>(tScoreDense.data); // shape: [1 image.height image.width]
    // output2
    auto vResLocalDescriptorMap = static_cast<float*>(tDescriptorsMap.data);

    const int width = tScoreDense.shape.d[2], height = tScoreDense.shape.d[1];
    const float scaleWidth = (tDescriptorsMap.shape.d[2] - 1.f) / (float)(tScoreDense.shape.d[2] - 1.f);
    const float scaleHeight = (tDescriptorsMap.shape.d[1] - 1.f) / (float)(tScoreDense.shape.d[1] - 1.f);

    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
    // 筛选符合条件的特征点，分数要大于一定阈值；
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense[row * width + col];
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                // 特征点的得分
                keypoint.response = score;
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

    // 如果特征点个数大于需求的，剔除响应小的
    // std::cout << "vKeyPoints size brfore erase is :" << vKeyPoints.size() << std::endl;
    if (vKeyPoints.size() > nKeypointsNum)
    {
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        // 将响应值大的排到前面，vKeyPoints.begin() + nKeypointsNum,为分割点，
        // nth_element函数按照方函数排序到分割点，分割点之后不做排序；
        std::nth_element(vKeyPoints.begin(), 
                         vKeyPoints.begin() + nKeypointsNum, 
                         vKeyPoints.end(), 
                         [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
                         return p1.response > p2.response;});
        // 删除响应值小的
        vKeyPoints.erase(vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end());
    }
    
    // 创建描述子 行为特征点个数 列为256列，精度是float32
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);

    // 创建特征点个数row * 2 cols 的mat;
    cv::Mat tWarp(vKeyPoints.size(), 2, CV_32FC1);
    auto pWarp = tWarp.ptr<float>();
    // 通过循环将缩放后的特征点坐标赋值给 tWarp
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }
    // 调用 ResamplerRT 函数，将描述子映射 tDescriptorsMap 和缩放后的特
    // 征点坐标 tWarp 作为参数，对数据进行采样，采样结果存储在 localDescriptors 中。
    ResamplerRT(tDescriptorsMap, tWarp, localDescriptors);

    // 对 localDescriptors 中的每一行进行归一化处理，即将其转化为单位向量。
    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
}

void HFNetRTModel::GetGlobalDescriptorFromTensor(const RTTensor &tDescriptors, 
                                                 cv::Mat &globalDescriptors)
{
    // 首先通过以下代码将RTTensor对象的数据指针(tDescriptors.data)转换为float*类型的指针
    auto vResGlobalDescriptor = static_cast<float*>(tDescriptors.data);
    // 创建一个大小为4096行1列的Mat、数据类型为单精度浮点型(CV_32F)的cv::Mat对象globalDescriptors
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    // 通过一个循环，将vResGlobalDescriptor中的数据逐个复制到globalDescriptors中：
    for (int temp = 0; temp < 4096; ++temp)
    {
        // 循环中的代码将vResGlobalDescriptor中的第temp个元素复制到globalDescriptors的
        // 第一行的第temp列上。

        // ？这不是冲突了吗？（）中是行，[]是列，问题是globalDescriptors只有一列啊？
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }
}

bool HFNetRTModel::LoadHFNetTRModel(void)
{
    // 代码开始通过createInferBuilder函数创建一个InferBuilder对象。该对象用于构建推理引擎。
    auto builder = unique_ptr<IBuilder>(createInferBuilder(mLogger));
    if (!builder) return false;
    // 设置explicitBatch标志，这是将值1进行位移操作，然后进行uint32_t类型的转换。该标志用于在创建网络定义时指定显式批处理。
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 使用createNetworkV2函数创建一个INetworkDefinition对象。
    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    // 代码检查network对象是否成功创建，如果未成功创建，则返回false，表示失败。
    if (!network) return false;
    // 使用createParser函数创建一个IParser对象，并将其与network对象和mLogger关联起来。
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if (!parser) return false;
    /// 使用parseFromFile函数从ONNX文件中解析模型，并检查解析是否成功。///
    auto parsed = parser->parseFromFile(mStrONNXFile.c_str(), 2);
    if (!parsed) return false;
    // 将输入张量的维度设置为mInputShape。
    network->getInput(0)->setDimensions(mInputShape);
    // 如果mMode为kImageToLocal，则取消标记第三个输出张量。
    if (mMode == kImageToLocal)
    {
        network->unmarkOutput(*network->getOutput(2));
    }
    // 使用createBuilderConfig函数创建一个IBuilderConfig对象。
    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;
    // 将BuilderFlag标志设置为kFP16，表示使用FP16数据类型。
    config->setFlag(BuilderFlag::kFP16);
    // 将内存池的限制设置为2MB。
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2 << 20);
    std::unique_ptr<ITimingCache> timingCache{nullptr};
    // 使用LoadTimingCacheFile函数加载时序缓存文件，创建一个ITimingCache对象。
    LoadTimingCacheFile(mStrCacheFile, config, timingCache);
    // 使用buildSerializedNetwork函数将网络和配置序列化为引擎，并创建一个IHostMemory对象。
    auto serializedEngine = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    // 如果日志级别大于等于kINFO，则打印输入和输出信息。
    if (mLogger.level >= ILogger::Severity::kINFO) PrintInputAndOutputsInfo(network);

    // Save Engine
    // SaveEngineToFile(DecideEigenFileName(mStrTRModelDir, mMode, mInputShape), serializedEngine);
    
    // 使用createInferRuntime函数创建一个IRuntime对象。
    unique_ptr<IRuntime> runtime{createInferRuntime(mLogger)};
    if (!runtime) return false;
    // 使用deserializeCudaEngine函数将序列化的引擎数据反序列化为ICudaEngine对象。
    mEngine = shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!mEngine) return false;
    // 使用UpdateTimingCacheFile函数更新时序缓存文件。
    UpdateTimingCacheFile(mStrCacheFile, config, timingCache);
    // 代码检查mContext对象是否成功创建，如果未成功创建，则返回false，表示失败。
    mContext = shared_ptr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) return false;
    // 最后，返回true，表示模型加载成功。
    return true;
}

void HFNetRTModel::LoadTimingCacheFile(const std::string& strFileName, std::unique_ptr<IBuilderConfig>& config, std::unique_ptr<ITimingCache>& timingCache)
{
    std::ifstream iFile(strFileName, std::ios::in | std::ios::binary);
    std::vector<char> content;
    if (!iFile)
    {
        cout << "Could not read timing cache from: " << strFileName
                            << ". A new timing cache will be generated and written." << std::endl;
        content = std::vector<char>();
    }
    else
    {
        iFile.seekg(0, std::ifstream::end);
        size_t fsize = iFile.tellg();
        iFile.seekg(0, std::ifstream::beg);
        content.resize(fsize);
        iFile.read(content.data(), fsize);
        iFile.close();
        std::cerr << "Loaded " << fsize << " bytes of timing cache from " << strFileName << std::endl;
    }
    
    timingCache.reset(config->createTimingCache(static_cast<const void*>(content.data()), content.size()));
    config->setTimingCache(*timingCache, false);
}

void HFNetRTModel::UpdateTimingCacheFile(const std::string& strFileName, std::unique_ptr<IBuilderConfig>& config, std::unique_ptr<ITimingCache>& timingCache)
{
    std::unique_ptr<nvinfer1::ITimingCache> fileTimingCache{config->createTimingCache(static_cast<const void*>(nullptr), 0)};

    std::ifstream iFile(strFileName, std::ios::in | std::ios::binary);
    if (iFile)
    {
        iFile.seekg(0, std::ifstream::end);
        size_t fsize = iFile.tellg();
        iFile.seekg(0, std::ifstream::beg);
        std::vector<char> content(fsize);
        iFile.read(content.data(), fsize);
        iFile.close();
        std::cerr << "Loaded " << fsize << " bytes of timing cache from " << strFileName << std::endl;
        fileTimingCache.reset(config->createTimingCache(static_cast<const void*>(content.data()), content.size()));
        if (!fileTimingCache)
        {
            throw std::runtime_error("Failed to create timingCache from " + strFileName + "!");
        }
    }
    fileTimingCache->combine(*timingCache, false);
    std::unique_ptr<nvinfer1::IHostMemory> blob{fileTimingCache->serialize()};
    if (!blob)
    {
        throw std::runtime_error("Failed to serialize ITimingCache!");
    }
    std::ofstream oFile(strFileName, std::ios::out | std::ios::binary);
    if (!oFile)
    {
        std::cerr << "Could not write timing cache to: " << strFileName << std::endl;
        return;
    }
    oFile.write((char*) blob->data(), blob->size());
    oFile.close();
    std::cerr << "Saved " << blob->size() << " bytes of timing cache to " << strFileName << std::endl;
}

std::string HFNetRTModel::DecideEigenFileName(const std::string& strEngineSaveDir, ModelDetectionMode mode, const Dims4 inputShape)
{
    string strFileName;
    strFileName = gStrModelDetectionName[mode] + "_" + 
                  to_string(inputShape.d[0]) + "x" + 
                  to_string(inputShape.d[1]) + "x" + 
                  to_string(inputShape.d[2]) + "x" + 
                  to_string(inputShape.d[3]) + ".engine";
    return strEngineSaveDir + "/" + strFileName;
}

bool HFNetRTModel::SaveEngineToFile(const std::string& strEngineSaveFile, const unique_ptr<IHostMemory>& serializedEngine)
{
    std::ofstream engineFile(strEngineSaveFile, std::ios::binary);
    engineFile.write(reinterpret_cast<char const*>(serializedEngine->data()), serializedEngine->size());
    if (engineFile.fail())
    {
        std::cerr << "Saving engine to file failed." << endl;
        return false;
    }
    return true;
}

bool HFNetRTModel::LoadEngineFromFile(const std::string& strEngineSaveFile)
{
    std::ifstream engineFile(strEngineSaveFile, std::ios::binary);
    if (!engineFile.good())
    {
        std::cerr << "Error opening engine file: " << strEngineSaveFile << endl;
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> vecEngineBlob(fsize);
    engineFile.read(reinterpret_cast<char*>(vecEngineBlob.data()), fsize);
    if (!engineFile.good())
    {
        std::cerr << "Error opening engine file: " << strEngineSaveFile << endl;
        return false;
    }

    unique_ptr<IRuntime> runtime{createInferRuntime(mLogger)};
    if (!runtime) return false;

    mEngine.reset(runtime->deserializeCudaEngine(vecEngineBlob.data(), vecEngineBlob.size()));
    if (!mEngine) return false;

    mContext = shared_ptr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) return false;

    return true;
}

void HFNetRTModel::PrintInputAndOutputsInfo(unique_ptr<INetworkDefinition>& network)
{
    std::cout << "model name: " << network->getName() << std::endl;

    const std::string strTypeName[] = {"FLOAT", "HALF", "INT8", "INT32", "BOOL", "UINT8"};
    for (int index = 0; index < network->getNbInputs(); ++index)
    {
        std::cout << "    inputs" << std::endl;

        auto input = network->getInput(index);

        std::cout << "        input name: " << input->getName() << std::endl;
        std::cout << "        input type: " << strTypeName[(int)input->getType()] << std::endl;
        std::cout << "        input shape: [";
        for (int alpha = 0; alpha < input->getDimensions().nbDims; ++alpha)
        {
            std::cout << input->getDimensions().d[alpha];
            if (alpha != input->getDimensions().nbDims - 1) std::cout << ", ";
            else std::cout << "]" << std::endl;
        }
        std::cout << "        input exec: " << input->isExecutionTensor() << endl;
    }

    for (int index = 0; index < network->getNbOutputs(); ++index)
    {
        std::cout << "    outputs" << std::endl;

        auto output = network->getOutput(index);

        std::cout << "        output name: " << output->getName() << std::endl;
        std::cout << "        output type: " << strTypeName[(int)output->getType()] << std::endl;
        std::cout << "        output shape: [";
        for (int alpha = 0; alpha < output->getDimensions().nbDims; ++alpha)
        {
            std::cout << output->getDimensions().d[alpha];
            if (alpha != output->getDimensions().nbDims - 1) std::cout << ", ";
            else std::cout << "]" << std::endl;
        }
        std::cout << "        output exec: " << output->isExecutionTensor() << endl;
    }
}
/**
 * Mat2Tensor是HFNetRTModel类中的一个函数。该函数接受一个cv::Mat对象(mat)
 * 和一个RTTensor对象(tensor)作为输入。函数的作用是将输入的cv::Mat转换为
 * RTTensor。然后，使用mat.convertTo()函数将输入的mat转换为单精度浮点型
 * （CV_32F）的fromMat。转换后的结果存储在fromMat中。
 * 
 * static_cast<float*>(tensor.data)是将RTTensor对象的数据指针(tensor.data)
 * 进行静态类型转换为float*类型的操作。tensor.data是一个指向RTTensor对象的数据的
 * 指针，它指向存储实际数据的内存位置。在这里，通过将其进行static_cast转换为float*
 * 类型，将其解释为指向float类型数据的指针。这样的转换可能是为了与cv::Mat对象的数据
 * 类型匹配，因为在函数的后续部分，mat.convertTo()函数将mat对象转换为CV_32F类型的
 * 数据。因此，通过将RTTensor的数据指针转换为float*类型，可以确保在转换过程中正确处
 * 理数据类型。
*/
void HFNetRTModel::Mat2Tensor(const cv::Mat &mat, RTTensor &tensor)
{
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), 
                    static_cast<float*>(tensor.data));
    mat.convertTo(fromMat, CV_32F);
}

void HFNetRTModel::Tensor2Mat(const RTTensor &tensor, cv::Mat &mat)
{
    // fromTensor 的大小由 tensor 的形状信息确定，宽度为 tensor.shape.d[1]，高度为 tensor.shape.d[2]，
    // 通道数为 tensor.shape.d[3]。数据类型为单精度浮点型 (CV_32FC)。
    // fromTensor 的数据指针指向 tensor.data 的转换后的 float* 类型。
    // 接下来，使用 fromTensor.convertTo() 函数将 fromTensor 转换为
    // 单精度浮点型 (CV_32F) 的数据，并存储到mat中。
    const cv::Mat fromTensor(cv::Size(tensor.shape.d[1], tensor.shape.d[2]), 
                             CV_32FC(tensor.shape.d[3]), static_cast<float*>(tensor.data));
    fromTensor.convertTo(mat, CV_32F);
}

void HFNetRTModel::ResamplerRT(const RTTensor &data, const cv::Mat &warp, cv::Mat &output)
{
    const Dims data_shape = data.shape;
    const int batch_size = data_shape.d[0];
    const int data_height = data_shape.d[1];
    const int data_width = data_shape.d[2];
    const int data_channels = data_shape.d[3];
    const cv::Size warp_shape = warp.size();

    // output_shape.set_dim(output_shape.dims() - 1, data_channels);
    // output = Tensor(DT_FLOAT, output_shape);
    output = cv::Mat(warp.rows, data_channels, CV_32F);
    
    const int num_sampling_points = warp.size().area() / batch_size / 2;
    if (num_sampling_points > 0)
    {
        Resampler(static_cast<float*>(data.data), warp.ptr<float>(), output.ptr<float>(),
                  batch_size, data_height, data_width, 
                  data_channels, num_sampling_points);
    }
}

#endif // USE_TENSORRT

} // namespace ORB_SLAM3