#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "../../Thirdparty/nanoflann.hpp"
#include "../../include/Extractors/HFextractor.h"
#include "../../EigenPlaces/tensorrt_engine.h"
#include "../../EigenPlaces/image_processor.h"
#include "../../include/Extractors/EigenPlacesExtractor.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace DeepRoute;

vector<string> GetPngFiles(string strPngDir);

struct KeyFrameNetVlad
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cv::Mat mGlobalDescriptors_front, mGlobalDescriptors_rear;
    int mnFrameId;
    float mPlaceRecognitionScore_front = 1.0, mPlaceRecognitionScore_rear = 1.0; 
    double timeStamp;
    Eigen::Matrix4d curPose;

    KeyFrameNetVlad(int id, const cv::Mat im, const cv::Mat im_rear, EigenPlacesExtractor* pModel, double time_stamp, Eigen::Matrix4d pose);
    KeyFrameNetVlad(int id ,const cv::Mat im, EigenPlacesExtractor* pModel, double time_stamp, Eigen::Matrix4d pose);
    // 从离线描述子文件加载的构造函数（双相机）
    KeyFrameNetVlad(int id, double time_stamp, Eigen::Matrix4d pose, const string& descriptor_path);
    // 从离线描述子文件加载的构造函数（单相机，仅前目）
    KeyFrameNetVlad(int id, double time_stamp, Eigen::Matrix4d pose, const string& descriptor_path, bool front_only);

    // KeyFrameNetVlad(int id, const cv::Mat im, const cv::Mat im_rear, BaseModel* pModel, double time_stamp, Eigen::Matrix4d pose);
    // KeyFrameNetVlad(int id ,const cv::Mat im, BaseModel* pModel, double time_stamp, Eigen::Matrix4d pose);
};

struct TUMTrajectoryData {
    double timestamp;
    Eigen::Matrix4d pose;
    Eigen::Vector3d translation;
    Eigen::Quaterniond quaternion;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AlignedTUMData {
    vector<double> times;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
};

struct AlignedDualCameraData {
    vector<string> files_front;
    vector<string> files_rear;
    vector<double> times;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
};

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

vector<double> GetTimesFromFileKitti(const string& filePath);
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileKitti(const string& filePath);
std::vector<TUMTrajectoryData, Eigen::aligned_allocator<TUMTrajectoryData>> ReadTUMTrajectory(const string& filePath);
AlignedTUMData AlignTUMTrajectoryToImages(const string& filePath, const vector<string>& imageFiles);
AlignedDualCameraData AlignDualCameraDataToTrajectory(const string& strDatasetPath_front, 
                                                      const string& strDatasetPath_rear, 
                                                      const string& strGTPosesPath);
vector<double> GetTimesFromFileTUM(const string& filePath, const vector<string>& imageFiles);
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFileTUM(const string& filePath, const vector<string>& imageFiles);

KeyFrameDB GetNCandidateLoopFrameCV(KeyFrameNetVlad* query, const KeyFrameDB &db, int k);
KeyFrameDB GetNCandidateLoopFrameEigen(KeyFrameNetVlad* query, const KeyFrameDB &db, int k, bool &use_rear);

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str);

void LoadConfigYaml(const string &configPath, cv::Size &ImSize);

// EigenPlaces模型初始化函数
EigenPlacesExtractor* InitEigenPlacesModel(const std::string& onnx_model_path, const std::string& engine_cache_path = "", cv::Size ImSize = cv::Size(512, 512));

// 离线描述子加载函数
bool LoadOfflineDescriptor(const string& bin_file_path, cv::Mat& descriptor_front, cv::Mat& descriptor_rear);


#endif // COMMON_H
