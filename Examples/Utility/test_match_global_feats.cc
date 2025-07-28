/**
 * HFNet SLAM Loop Detection Test
 * 
 * This application tests loop detection performance comparing:
 * - OpenCV-based global feature matching 
 * - Eigen-based global feature matching (faster)
 * - KDTree spatial search for candidate selection
 * 
 * Includes real-time 3D visualization using Pangolin
 */

#include "common.h"
#include "loop_visual.h"
#include <random>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

/**
 * HFNet SLAM Loop Detection Test
 * 
 * This application tests loop detection performance comparing:
 * - OpenCV-based global feature matching 
 * - Eigen-based global feature matching (faster)
 * - KDTree spatial search for candidate selection
 * 
 * Includes real-time 3D visualization using Pangolin
 */

#include "common.h"
#include "loop_visual.h"
#include <random>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

int main(int argc, char** argv)
{
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    if (argc != 5 && argc != 4) {
        std::cerr << std::endl << "Usage: test_match_global_feats path_to_dataset path_to_model  gt_poses (time_stamp optional)" << std::endl;
        return -1;
    }
    
    const string strDatasetPath = string(argv[1]);
    const string strModelPath = string(argv[2]);
    const string strGTPosesPath = string(argv[3]);
    string strTimeStampPath = "";
    if (argc == 5) {
        strTimeStampPath = string(argv[4]);
    }

    const vector<string> files = GetPngFiles(strDatasetPath);
    if (files.empty()) {
        std::cout << "Error, failed to find any valid image in: " << strDatasetPath << std::endl;
        return 1;
    } else {
        std::cout << "Found " << files.size() << " images in: " << strDatasetPath << std::endl;
    }

    vector<double> times;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gtPoses;
    if(!strTimeStampPath.empty()) {
        times = GetTimesFromFileKitti(strTimeStampPath);  
        gtPoses = GetGTPosesFromFileKitti(strGTPosesPath);
    } else {
        // TUM格式：一次性获取对齐的时间戳和位姿，避免重复计算
        auto alignedData = AlignTUMTrajectoryToImages(strGTPosesPath, files);
        times = alignedData.times;
        gtPoses = alignedData.poses;
    }

    if (gtPoses.empty()) {
        std::cout << "Error, failed to find any valid gt poses in: " << strGTPosesPath << std::endl;
        return 1;
    } else {
        std::cout << "Found " << gtPoses.size() << " ground truth poses in: " << strGTPosesPath << std::endl;
    }

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
    while (cur < end) {
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

    // 启动Pangolin可视化
    StartVisualization();

    int select = 0;
    while (1) {
        // 检查暂停状态
        if (g_viewer && g_viewer->IsPaused()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        select++;

        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        cv::Mat image_show = imread(strDatasetPath + files[select], IMREAD_COLOR);
        double timeStamp = times[select];
        Eigen::Matrix4d pose = gtPoses[select];

        KeyFrameNetVlad *pKFHF = new KeyFrameNetVlad(select, image, pModel, timeStamp, pose);

        Eigen::Vector3d query_pos = pose.block<3,1>(0,3);
        const double search_radius = 5.0; 
        const double time_threshold = 10.0; 
        
        // KDTree搜索
        std::vector<nanoflann::ResultItem<unsigned int, double>> temp_matches;
        nanoflann::SearchParameters params;
        kdtree->radiusSearch(query_pos.data(), search_radius * search_radius, temp_matches, params);
        
        std::vector<size_t> valid_indices;
        for (const auto& m : temp_matches) {
            size_t idx = m.first;
            if (std::abs(timeStamp - vKeyFrameDB[idx]->timeStamp) > time_threshold && 
                vKeyFrameDB[idx]->mnFrameId < pKFHF->mnFrameId - 100)
                valid_indices.push_back(idx);
        }

        // Eigen回环检测
        auto t1 = chrono::steady_clock::now();
        auto res = GetNCandidateLoopFrameEigen(pKFHF, vKeyFrameDB, 5);
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        std::cout << "Query cost time: " << t << std::endl;

        // 更新Pangolin可视化
        UpdateVisualization(gtPoses, select, res, valid_indices, vKeyFrameDB);

        // 可视化对比：构建7张图像：query, 3 KDTree, 3 Eigen  
        std::vector<cv::Mat> imgs(7);
        std::vector<std::string> texts(7);
        
        // Query
        imgs[0] = image_show.clone();
        texts[0] = "Query: " + std::to_string((int)pKFHF->mnFrameId) + ", t=" + std::to_string(pKFHF->timeStamp);
        
        if (res.size() > 0) {
            std::cout << "Found " << res.size() << " Eigen candidates, updating Pangolin..." << std::endl;
        }
        
        // KDTree候选
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
        
        // Eigen候选
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
        
        // 拼接图像
        cv::Mat left = imgs[0];
        cv::Mat right_row1, right_row2, right_all, all;
        cv::hconcat(std::vector<cv::Mat>{imgs[1], imgs[2], imgs[3]}, right_row1);
        cv::hconcat(std::vector<cv::Mat>{imgs[4], imgs[5], imgs[6]}, right_row2);
        cv::vconcat(right_row1, right_row2, right_all);
        
        if (left.rows < right_all.rows) {
            int pad = right_all.rows - left.rows;
            cv::copyMakeBorder(left, left, 0, pad, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        }
        cv::hconcat(std::vector<cv::Mat>{left, right_all}, all);
        cv::namedWindow("Compare Candidates", cv::WINDOW_NORMAL);
        cv::resizeWindow("Compare Candidates", 1280, 720);
        
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
    }

    system("pause");
    return 0;
}