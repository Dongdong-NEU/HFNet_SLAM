/**
 *
GetNCandidateLoopFrameCV():
Query cost time: 1339
Query cost time: 1328
GetNCandidateLoopFrameEigen():
Query cost time: 245
Query cost time: 259
 * Eigen is much faster than OpenCV
 */
#include <iostream>
#include <chrono>
#include <unordered_set>
#include "Eigen/Core"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>

#include "nanoflann.hpp"
#include <memory>

#include "Frame.h"
#include "Extractors/HFextractor.h"
#include "utility_common.h"

#include "utility_common.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;


struct KeyFrameNetVlad
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cv::Mat mGlobalDescriptors;
    int mnFrameId;
    float mPlaceRecognitionScore = 1.0;
    double timeStamp;
    Eigen::Matrix4d curPose;

    KeyFrameNetVlad(int id, const cv::Mat im, BaseModel* pModel, double time_stamp, Eigen::Matrix4d pose) {
        mnFrameId = id;
        timeStamp = time_stamp;
        curPose = pose;
        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, intermediate;
        pModel->Detect(im, vKeyPoints, localDescriptors, mGlobalDescriptors, 1000, 0.01);
    }
};


// 读取时间戳文件，每行一个浮点数，返回double vector
vector<double> GetTimesFromFile(const string& filePath) {
    vector<double> times;
    std::ifstream fin(filePath);
    if (!fin.is_open()) return times;
    string line;
    while (std::getline(fin, line)) {
        if (!line.empty()) times.push_back(std::stod(line));
    }
    return times;
}

// 读取位姿文件，每行12个浮点数，返回每行为一个4x4矩阵（最后一行补[0 0 0 1]）的vector
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> GetGTPosesFromFile(const string& filePath) {
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    std::ifstream fin(filePath);
    if (!fin.is_open()) return poses;
    string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (vals.size() != 12) continue;
        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
        mat(0,0) = vals[0]; mat(0,1) = vals[1]; mat(0,2) = vals[2];  mat(0,3) = vals[3];
        mat(1,0) = vals[4]; mat(1,1) = vals[5]; mat(1,2) = vals[6];  mat(1,3) = vals[7];
        mat(2,0) = vals[8]; mat(2,1) = vals[9]; mat(2,2) = vals[10]; mat(2,3) = vals[11];
        poses.push_back(mat);
    }
    return poses;
}


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

KeyFrameDB GetNCandidateLoopFrameCV(KeyFrameNetVlad* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId >= query->mnFrameId - 100) return KeyFrameDB();
    std::vector<KeyFrameNetVlad*> candidates;
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameNetVlad *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 100) break;
        pKF->mPlaceRecognitionScore = cv::norm(query->mGlobalDescriptors - pKF->mGlobalDescriptors, cv::NORM_L2);
        if (pKF->mPlaceRecognitionScore < 0.7)
            candidates.push_back(pKF);
    }
    int num = std::min(k, (int)candidates.size());
    KeyFrameDB res(num);
    std::partial_sort_copy(candidates.begin(), candidates.end(), res.begin(), res.end(),
        [](KeyFrameNetVlad* const f1, KeyFrameNetVlad* const f2) {
            return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
        });
    return res;
}

KeyFrameDB GetNCandidateLoopFrameEigen(KeyFrameNetVlad* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId >= query->mnFrameId - 100) return KeyFrameDB();
    std::vector<KeyFrameNetVlad*> candidates;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
        queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, 
                        query->mGlobalDescriptors.cols);

    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameNetVlad *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 100) break;
        // 位姿平移距离过滤
        Eigen::Vector3d query_t = query->curPose.block<3,1>(0,3);
        Eigen::Vector3d cand_t = pKF->curPose.block<3,1>(0,3);
        double trans_dist = (query_t - cand_t).norm();
        if (trans_dist >= 10.0) continue;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
            pKFDescriptors(
                pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, 
                pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = (queryDescriptors - pKFDescriptors).norm();
        if (pKF->mPlaceRecognitionScore < 0.8)
            candidates.push_back(pKF);
    }
    int num = std::min(k, (int)candidates.size());
    KeyFrameDB res(num);
    std::partial_sort_copy(candidates.begin(), candidates.end(), res.begin(), res.end(),
        [](KeyFrameNetVlad* const f1, KeyFrameNetVlad* const f2) {
            return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
        });
    if(candidates.size() > 0)
        std::cout << "Candidate size: " << candidates.size() << std::endl;
    return res;
}

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str)
{
    cv::Mat plot;
    if (image.channels() == 1)
        cv::cvtColor(image, plot, cv::COLOR_GRAY2RGB);
    else
        plot = image.clone();
    cv::putText(plot, str, cv::Point2d(0, 30),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::imshow(title, plot);
}

int main(int argc, char** argv)
{
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    if (argc != 5) {
        std::cerr << std::endl << "Usage: test_match_global_feats path_to_dataset path_to_model time_stamp gt_poses" << std::endl;
        return -1;
    }
    
    const string strDatasetPath = string(argv[1]);
    const string strModelPath = string(argv[2]);
    const string strTimeStampPath = string(argv[3]);
    const string strGTPosesPath = string(argv[4]);

    const vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    if (files.empty()) {
        std::cout << "Error, failed to find any valid image in: " << strDatasetPath << std::endl;
        return 1;
    }else{
        std::cout << "Found " << files.size() << " images in: " << strDatasetPath << std::endl;
    }
    const vector<double> times = GetTimesFromFile(strTimeStampPath);
    if (times.empty()) {
        std::cout << "Error, failed to find any valid time in: " << strTimeStampPath << std::endl;
        return 1;
    }else{
        std::cout << "Found " << times.size() << " timestamps in: " << strTimeStampPath << std::endl;
    }
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gtPoses = GetGTPosesFromFile(strGTPosesPath);
    if (gtPoses.empty()) {
        std::cout << "Error, failed to find any valid gt poses in: " << strGTPosesPath << std::endl;
        return 1;
    }else{
        std::cout << "Found " << gtPoses.size() << " ground truth poses in: " << strGTPosesPath << std::endl;
    }

    assert(files.size() == times.size());
    assert(files.size() == gtPoses.size());

    cv::Size ImSize = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE).size();
    if (ImSize.area() == 0) {
        std::cout << "Error, failed to read the image at: " << strDatasetPath + files[0] << std::endl;
        return 1;
    }


    cv::Vec4i inputShape{1, ImSize.height, ImSize.width, 1};
    auto pModel = InitRTModel(strModelPath, kImageToLocalAndGlobal, inputShape);

    int start = 0;
    int end = files.size();

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(30, end);

    const int step = 4;
    int nKeyFrame = (end - start) / step;

    if (nKeyFrame <= 100) exit(-1);
    std::cout << "Dataset range: [" << start << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << std::endl;

    KeyFrameDB vKeyFrameDB;
    vKeyFrameDB.reserve(nKeyFrame);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keyframe_positions;
    float cur = start;
    // 每间隔4步创建1个关键帧
    while (cur < end)
    {
        int select = cur;
        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        KeyFrameNetVlad *pKFHF = new KeyFrameNetVlad(select, image, pModel, times[select], gtPoses[select]);
        vKeyFrameDB.emplace_back(pKFHF);
        // 提取平移部分
        Eigen::Vector3d pos = gtPoses[select].block<3,1>(0,3);
        keyframe_positions.push_back(pos);
        cur += step;
    }

    // 构建KDTree
    KeyFramePointCloud cloud;
    cloud.pts = keyframe_positions;
    auto kdtree = std::make_unique<KDTreeType>(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree->buildIndex();

    char command = ' ';
    int select = 0;
    while (1)
    {
        // if (command == 'w') select += 1;
        // else if (command == 'x') select -= 1;
        // else if (command == ' ') select = distribution(generator);
        select++;

        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        cv::Mat image_show = imread(strDatasetPath + files[select], IMREAD_COLOR);
        double timeStamp = times[select];
        Eigen::Matrix4d pose = gtPoses[select];

        KeyFrameNetVlad *pKFHF = new KeyFrameNetVlad(select, image, pModel, timeStamp, pose);

        Eigen::Vector3d query_pos = pose.block<3,1>(0,3);
        const double search_radius = 5.0; 
        const double time_threshold = 10.0; 
        // 结果容器，包含索引和距离
        std::vector<nanoflann::ResultItem<unsigned int, double>> temp_matches;
        nanoflann::SearchParameters params;
        // 搜索半径为search_radius的邻近点
        // 注意：nanoflann的radiusSearch使用的是平方距离，所以传入的search_radius需要平方
        size_t nMatches = kdtree->radiusSearch(query_pos.data(), search_radius * search_radius, temp_matches, params);
        std::vector<size_t> valid_indices;
        for (const auto& m : temp_matches) {
            size_t idx = m.first;
            // 过滤时间相近的且关键帧ID小于当前关键帧ID - 100（只能从历史帧中选择）
            // query帧的时间减去历史帧的时间大于time_threshold
            if (std::abs(timeStamp - vKeyFrameDB[idx]->timeStamp) > time_threshold && 
                                     vKeyFrameDB[idx]->mnFrameId < pKFHF->mnFrameId - 100)
                valid_indices.push_back(idx);
        }

        auto t1 = chrono::steady_clock::now();
        auto res = GetNCandidateLoopFrameEigen(pKFHF, vKeyFrameDB, 5);
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        std::cout << "Query cost time: " << t << std::endl;

        // 可视化对比：左侧显示KDTree结果，右侧显示Eigen结果
        // 构建7张图像：query, 3 KDTree, 3 Eigen
        std::vector<cv::Mat> imgs(7);
        std::vector<std::string> texts(7);
        // Query
        imgs[0] = image_show.clone();
        texts[0] = "Query: " + std::to_string((int)pKFHF->mnFrameId) + ", t=" + std::to_string(pKFHF->timeStamp);
        // KDTree
        for (size_t i = 0; i < 3; ++i) {
            if (i < valid_indices.size()) {
                int kf_idx = valid_indices[i];
                imgs[1+i] = imread(strDatasetPath + files[vKeyFrameDB[kf_idx]->mnFrameId], IMREAD_COLOR);
                texts[1+i] = "KDTree " + std::to_string(i+1) + ": " + std::to_string((int)vKeyFrameDB[kf_idx]->mnFrameId) + ", t=" + std::to_string(vKeyFrameDB[kf_idx]->timeStamp);
            } else {
                imgs[1+i] = cv::Mat::zeros(ImSize, CV_8UC3);
                texts[1+i] = "KDTree " + std::to_string(i+1) + ": None";
            }
        }
        // Eigen
        for (size_t i = 0; i < 3; ++i) {
            if (i < res.size()) {
                imgs[4+i] = imread(strDatasetPath + files[res[i]->mnFrameId], IMREAD_COLOR);
                texts[4+i] = "Eigen " + std::to_string(i+1) + ": " + std::to_string((int)res[i]->mnFrameId) + ", score=" + std::to_string(res[i]->mPlaceRecognitionScore);
            } else {
                imgs[4+i] = cv::Mat::zeros(ImSize, CV_8UC3);
                texts[4+i] = "Eigen " + std::to_string(i+1) + ": None";
            }
        }
        // 在每张图上加文字
        for (int i = 0; i < 7; ++i) {
            cv::putText(imgs[i], texts[i], cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0,255,0), 2);
        }
        // 拼接成左侧单列Query，右侧2行3列（KDTree+Eigen）
        cv::Mat left = imgs[0];
        cv::Mat right_row1, right_row2, right_all, all;
        cv::hconcat(std::vector<cv::Mat>{imgs[1], imgs[2], imgs[3]}, right_row1);
        cv::hconcat(std::vector<cv::Mat>{imgs[4], imgs[5], imgs[6]}, right_row2);
        cv::vconcat(right_row1, right_row2, right_all);
        // 保证左侧和右侧高度一致，不拉伸，直接黑色补齐
        if (left.rows < right_all.rows) {
            int pad = right_all.rows - left.rows;
            cv::copyMakeBorder(left, left, 0, pad, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        }
        cv::hconcat(std::vector<cv::Mat>{left, right_all}, all);
        cv::namedWindow("Compare Candidates", cv::WINDOW_NORMAL);
        // cv::resizeWindow("Compare Candidates", 1280, 720);
        cv::resizeWindow("Compare Candidates", 2560, 1440);
        if(valid_indices.size() > 0 || res.size() > 0){
            cv::imshow("Compare Candidates", all);
            cv::waitKey(500);
        }
         
        // 输出两组结果的ID便于对比
        std::cout << "KDTree candidates: ";
        for (size_t i = 0; i < std::min(valid_indices.size(), size_t(3)); ++i)
            std::cout << vKeyFrameDB[valid_indices[i]]->mnFrameId << " ";
        std::cout << "\nEigen candidates: ";
        for (size_t i = 0; i < std::min(res.size(), size_t(3)); ++i)
            std::cout << res[i]->mnFrameId << " ";
        std::cout << std::endl;
        if (select >= end - 1) {
            std::cout << "Reached the end of dataset, exiting..." << std::endl;
            break;
        }

        // 检测到回环后10s内不在检测
        if(valid_indices.size() >0 && res.size() >= 3){
            select += 100;
        }
        // command = cv::waitKey();
    }

    system("pause");

    return 0;
}