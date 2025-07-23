
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "Extractors/HFextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

HFextractor::HFextractor(int _nfeatures, float _threshold, BaseModel* _pModels):
    nfeatures(_nfeatures), threshold(_threshold)
{
    mvpModels.resize(1);
    mvpModels[0] = _pModels;
    scaleFactor = 1.0;
    nlevels = 1;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}


HFextractor::HFextractor(int _nfeatures, float _threshold, float _scaleFactor, 
                        int _nlevels, const std::vector<BaseModel*>& _vpModels):
        nfeatures(_nfeatures), threshold(_threshold), mvpModels(_vpModels)
{
    scaleFactor = _scaleFactor;
    nlevels = _nlevels;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);
    // 一共有几层，每一层将会提取几个特征点；
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

// 提取当前帧特征点 
int HFextractor::operator() (const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                             cv::Mat &localDescriptors, cv::Mat &globalDescriptors)
{
    if (image.empty() || image.type() != CV_8UC1) return -1;
    
    int res = -1;
    // 如果金字塔层数等于1，那么在在一层图像上提取特征点；
    if (nlevels == 1) res = ExtractSingleLayer(image, vKeyPoints, localDescriptors, globalDescriptors);
    else // 金字塔层数不为1的话，如果检测模型的种类是RT
    {
        // mvpModels[0]->Type()开启Tensorrt的条件编译后返回的种类是TensorRT种类，到else中执行并行提取特征点；
        if (mvpModels[0]->Type() == kHFNetVINOModel)
            res = ExtractMultiLayers(image, vKeyPoints, localDescriptors, globalDescriptors);
        else
            res = ExtractMultiLayersParallel(image, vKeyPoints, localDescriptors, globalDescriptors);
    }
    return res;
}

void HFextractor::ComputePyramid(const cv::Mat &image)
{
    mvImagePyramid[0] = image;
    for (int level = 1; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));

        // Compute the resized image
        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
        }
    }
}

int HFextractor::ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                                    cv::Mat &localDescriptors, cv::Mat &globalDescriptors)
{
    if (!mvpModels[0]->Detect(image, vKeyPoints, localDescriptors, globalDescriptors, nfeatures, threshold))
        cerr << "Error while detecting keypoints" << endl;

    return vKeyPoints.size();
}

int HFextractor::ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                                    cv::Mat &localDescriptors, cv::Mat &globalDescriptors)
{
    ComputePyramid(image);

    int nKeypoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        if (level == 0)
        {
            if (!mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], globalDescriptors, mnFeaturesPerLevel[level], threshold))
                cerr << "Error while detecting keypoints" << endl;
        }
        else
        {
            if (!mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], mnFeaturesPerLevel[level], threshold))
                cerr << "Error while detecting keypoints" << endl;
        }
        nKeypoints += allKeypoints[level].size();
    }
    vKeyPoints.clear();
    vKeyPoints.reserve(nKeypoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for (auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);

    return vKeyPoints.size();
}

class DetectParallel : public cv::ParallelLoopBody
{
public:

    DetectParallel (vector<cv::KeyPoint> *allKeypoints, cv::Mat *allDescriptors, 
                    cv::Mat *globalDescriptors, HFextractor* pExtractor)
        : mAllKeypoints(allKeypoints), mAllDescriptors(allDescriptors), 
          mGlobalDescriptors(globalDescriptors), mpExtractor(pExtractor) {}

    virtual void operator ()(const cv::Range& range) const override
    {
        for (int level = range.start; level != range.end; ++level)
        {
            if (level == 0)
            {   
                // 初始层：用相应的模型去检测相应的特征点和描述子，最初始层要检测全局描述子
                if (!mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], 
                                                           mAllKeypoints[level], 
                                                           mAllDescriptors[level], 
                                                           *mGlobalDescriptors, 
                                                           mpExtractor->mnFeaturesPerLevel[level], 
                                                           mpExtractor->threshold))
                    cerr << "Error while detecting keypoints" << endl;
            }
            else
            {
                // 非初始层不进行全局描述子检测
                if (!mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], 
                                                           mAllKeypoints[level], 
                                                           mAllDescriptors[level], 
                                                           mpExtractor->mnFeaturesPerLevel[level], 
                                                           mpExtractor->threshold))
                    cerr << "Error while detecting keypoints" << endl;
            }
        }
    }
    /*这是 DetectParallel 类中的赋值运算符 operator= 的实现。在这个实现中，
    赋值运算符的功能是将一个 DetectParallel 对象赋值给另一个 DetectParallel
    对象，并返回被赋值后的对象。*/
    DetectParallel& operator=(const DetectParallel &) {
        return *this;
    };
private:
    vector<cv::KeyPoint> *mAllKeypoints;
    cv::Mat *mAllDescriptors;
    cv::Mat *mGlobalDescriptors;
    HFextractor* mpExtractor;
};

int HFextractor::ExtractMultiLayersParallel(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                                            cv::Mat &localDescriptors, cv::Mat &globalDescriptors)
{
    ComputePyramid(image);
    // 特征点总数
    int nKeypoints = 0;
    // 分层特征点总数
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    // 分层描述子
    vector<cv::Mat> allDescriptors(nlevels);
    // 创建并行检测类 detecter；
    DetectParallel detector(allKeypoints.data(), allDescriptors.data(), &globalDescriptors, this);
    // 函数调用 cv::parallel_for_ 函数，将范围从 0 到 nlevels 分成多个任务，并使用 detector 对象进行并行处理
    cv::parallel_for_(cv::Range(0, nlevels), detector);

    for (int level = 0; level < nlevels; ++level)
        nKeypoints += allKeypoints[level].size();

    vKeyPoints.clear();
    vKeyPoints.reserve(nKeypoints);
    // 将所有特征点都设置金字塔层级，以及大小，都存放入一个容器中；
    for (int level = 0; level < nlevels; ++level)
    {
        for (auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    // vconcat将alldescriptors中的矩阵连接起来，存放到localDescriptors中
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);
    // 返回检测出多少特征点；
    return vKeyPoints.size();
}

} //namespace ORB_SLAM3