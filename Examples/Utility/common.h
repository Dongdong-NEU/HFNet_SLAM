#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "nanoflann.hpp"
#include "Extractors/HFextractor.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

// 函数声明
vector<string> GetPngFiles(string strPngDir);

// 数据结构定义
struct KeyFrameNetVlad
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cv::Mat mGlobalDescriptors;
    int mnFrameId;
    float mPlaceRecognitionScore = 1.0;
    double timeStamp;
    Eigen::Matrix4d curPose;

    KeyFrameNetVlad(int id, const cv::Mat im, BaseModel* pModel, double time_stamp, Eigen::Matrix4d pose);
};

// TUM轨迹数据结构
struct TUMTrajectoryData {
    double timestamp;
    Eigen::Matrix4d pose;
    Eigen::Vector3d translation;
    Eigen::Quaterniond quaternion;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 对齐TUM轨迹数据到图像文件
struct AlignedTUMData {
    vector<double> times;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
};

// KDTree点云适配器
struct KeyFramePointCloud {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx][dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, KeyFramePointCloud>,
    KeyFramePointCloud, 3>;

typedef vector<KeyFrameNetVlad*> KeyFrameDB;

// 函数声明
vector<double> GetTimesFromFileKitti(const string& filePath);
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileKitti(const string& filePath);
std::vector<TUMTrajectoryData, Eigen::aligned_allocator<TUMTrajectoryData>> ReadTUMTrajectory(const string& filePath);
AlignedTUMData AlignTUMTrajectoryToImages(const string& filePath, const vector<string>& imageFiles);
vector<double> GetTimesFromFileTUM(const string& filePath, const vector<string>& imageFiles);
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileTUM(const string& filePath, const vector<string>& imageFiles);

KeyFrameDB GetNCandidateLoopFrameCV(KeyFrameNetVlad* query, const KeyFrameDB &db, int k);
KeyFrameDB GetNCandidateLoopFrameEigen(KeyFrameNetVlad* query, const KeyFrameDB &db, int k);

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str);

#endif // COMMON_H
