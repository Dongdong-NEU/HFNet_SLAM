#include "common.h"
#include "loop_visual.h"
#include <opencv2/highgui.hpp>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <iostream>
#include <cassert>

#include <opencv2/opencv.hpp>
#include "fisheye_utils.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace DeepRoute;


int main(int argc, char** argv)
{
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    if (argc != 7 && argc != 8) {
        std::cerr << std::endl << "Usage: test_match_global_feats dataset_front dataset_rear path_to_model gt_poses camera_cfg config_yaml [offline_descriptor_path]" << std::endl;
        std::cerr << "If offline_descriptor_path is provided, will load pre-computed descriptors instead of running inference" << std::endl;
        return -1;
    }

    const string strDatasetPath_front = string(argv[1]);
    const string strDatasetPath_rear = string(argv[2]);
    const string strModelPath = string(argv[3]);
    const string strGTPosesPath = string(argv[4]);
    const string strCamerasCfgPath = string(argv[5]);
    const string strConfigYamlpath = string(argv[6]);
    
    // 离线描述子路径（可选）
    bool use_offline_descriptor = (argc == 8);
    string strOfflineDescriptorPath = use_offline_descriptor ? string(argv[7]) : "";

    const bool crop_enabled = true;

    vector<string> files_front, files_rear;
    vector<double> times;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> gtPoses;

    cv::Size ImSizeFinal;
    LoadConfigYaml(strConfigYamlpath, ImSizeFinal);

    FisheyeCameraParams camParams1, camParams2;
    if (!LoadFisheyeCameraParams(strCamerasCfgPath, "camera_1", camParams1)) {
        std::cerr << "Failed to load camera intrinsics for panoramic_1 from: " << strCamerasCfgPath << std::endl;
        return -1;
    }
    if (!LoadFisheyeCameraParams(strCamerasCfgPath, "camera_4", camParams2)) {
        std::cerr << "Failed to load camera intrinsics for panoramic_3 from: " << strCamerasCfgPath << std::endl;
        return -1;
    }
    pair<cv::Mat, cv::Mat> camera1 = UndistortFisheyeParam(camParams1);
    pair<cv::Mat, cv::Mat> camera2 = UndistortFisheyeParam(camParams2);

    auto alignedData = AlignDualCameraDataToTrajectory(strDatasetPath_front, strDatasetPath_rear, strGTPosesPath);
    files_front = alignedData.files_front;
    files_rear = alignedData.files_rear;
    times = alignedData.times;
    gtPoses = alignedData.poses;
    

    if (files_front.empty() || files_rear.empty()) {
        std::cout << "Error, failed to find any valid aligned image pairs in: " << strDatasetPath_front << " or " << strDatasetPath_rear << std::endl;
        return 1;
    } else {
        std::cout << "Found " << files_front.size() << " aligned image pairs" << std::endl;
    }

    if (gtPoses.empty()) {
        std::cout << "Error, failed to find any valid gt poses in: " << strGTPosesPath << std::endl;
        return 1;
    } else {
        std::cout << "Found " << gtPoses.size() << " ground truth poses in: " << strGTPosesPath << std::endl;
    }
    
    assert(files_front.size() == gtPoses.size());

    // 初始化EigenPlaces模型（仅在不使用离线描述子时需要）
    EigenPlacesExtractor* pModel = nullptr;
    if (!use_offline_descriptor) {
        std::cout << "Initializing EigenPlaces model for online inference..." << std::endl;
        std::string onnx_model_path = strModelPath + "/eigenplaces_resnet50_fixedshape_240_320_GPU_simplified.onnx";
        std::string engine_cache_path = strModelPath + "/eigenplaces_resnet50_fixedshape_240_320_GPU_simplified.engine";
        // std::string onnx_model_path = strModelPath + "/eigenplaces_resnet50_fixedshape_360_640_GPU_simplified.onnx";
        // std::string engine_cache_path = strModelPath + "/eigenplaces_resnet50_fixedshape_360_640_GPU_simplified.engine";
        pModel = InitEigenPlacesModel(onnx_model_path, engine_cache_path, ImSizeFinal);
    } else {
        std::cout << "Using offline descriptors from: " << strOfflineDescriptorPath << std::endl;
    }

    int start = 0;
    int end = files_front.size();

    const int step = 1;
    int nKeyFrame = (end - start) / step;

    if (nKeyFrame <= 300) exit(-1);
    std::cout << "Dataset range: [" << start << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << std::endl;

    KeyFrameDB vKeyFrameDB;
    vKeyFrameDB.reserve(nKeyFrame);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keyframe_positions;
    float cur = start;
    
    // 每间隔4步创建1个关键帧
    std::cout << "Building keyframe database..." << std::endl;
    auto db_start = chrono::steady_clock::now();
    
    while (cur < end) {
        int select = cur;
        
        KeyFrameNetVlad *pKFHF = nullptr;
        
        if (use_offline_descriptor) {
            // 使用离线描述子模式
            auto load_start = chrono::steady_clock::now();
            pKFHF = new KeyFrameNetVlad(select, times[select], gtPoses[select], strOfflineDescriptorPath);
            auto load_end = chrono::steady_clock::now();
            auto load_time = chrono::duration_cast<chrono::milliseconds>(load_end - load_start).count();
            
            std::cout << "Frame " << select << " - Offline descriptor loading: " << load_time << "ms" << std::endl;
        } else {
            // 在线提取特征模式
            string front_path = strDatasetPath_front + files_front[select];
            string rear_path = strDatasetPath_rear + files_rear[select];
            std::cout << "Loading: " << front_path << std::endl;
            std::cout << "Loading: " << rear_path << std::endl;

            auto img_start = chrono::steady_clock::now();
            cv::Mat image_front = imread(front_path, IMREAD_GRAYSCALE);
            if (!image_front.empty() && !crop_enabled) {
                image_front = UndistortImage(image_front, camera1.first, camera1.second, camParams1, ImSizeFinal);
            }else if (!image_front.empty() && crop_enabled) {
                std::cout << "Cropping front image" << std::endl;
                image_front = CropImage(image_front, 960,0,1920,1440);
            }

            cv::Mat image_rear = imread(rear_path, IMREAD_GRAYSCALE);
            if (!image_rear.empty() && !crop_enabled) {
                image_rear = UndistortImage(image_rear, camera2.first, camera2.second, camParams2, ImSizeFinal);
            }else if (!image_rear.empty() && crop_enabled) {
                image_rear = CropImage(image_rear, 480,0,960,720);
            }
            auto img_end = chrono::steady_clock::now();
            auto img_time = chrono::duration_cast<chrono::milliseconds>(img_end - img_start).count();
            
            if (image_front.empty() || image_rear.empty()) {
                std::cerr << "Failed to load images at frame " << select << std::endl;
                std::cerr << "Front path: " << front_path << std::endl;
                std::cerr << "Rear path: " << rear_path << std::endl;
                cur += step;
                continue;
            }
            
            auto feat_start = chrono::steady_clock::now();
            pKFHF = new KeyFrameNetVlad(select, image_front, image_rear, pModel, times[select], gtPoses[select]);
            auto feat_end = chrono::steady_clock::now();
            auto feat_time = chrono::duration_cast<chrono::milliseconds>(feat_end - feat_start).count();
            
            std::cout << "Frame " << select << " - Image loading: " << img_time << "ms, Feature extraction: " << feat_time << "ms" << std::endl;
        }
        
        vKeyFrameDB.emplace_back(pKFHF);
        Eigen::Vector3d pos = gtPoses[select].block<3,1>(0,3);
        keyframe_positions.push_back(pos);
        cur += step;
    }
    
    auto db_end = chrono::steady_clock::now();
    auto db_time = chrono::duration_cast<chrono::seconds>(db_end - db_start).count();
    std::cout << "Keyframe database built in " << db_time << " seconds" << std::endl;

    KeyFramePointCloud cloud;
    cloud.pts = keyframe_positions;
    auto kdtree = std::make_unique<KDTreeType>(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree->buildIndex();

    // 启动Pangolin可视化
    StartVisualization();
    
    // 初始化轨迹数据到可视化器
    if (g_viewer) {
        g_viewer->UpdateTrajectory(gtPoses);
        std::cout << "Initialized trajectory with " << gtPoses.size() << " poses" << std::endl;
    }

    int select = 0;
    bool show_undistort = false;
    std::cout << "Show undistort: " << show_undistort << std::endl;
    // 模拟定位
    while (1) {
        // 检查暂停状态
        if (g_viewer && g_viewer->IsPaused()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        select++;

        double timeStamp = times[select];
        Eigen::Matrix4d pose = gtPoses[select];

        KeyFrameNetVlad *pKFHF = nullptr;
        cv::Mat image_show;
        cv::Mat image_show_undistort;
        
        if (use_offline_descriptor) {
            // 使用离线描述子模式（仅前目）
            pKFHF = new KeyFrameNetVlad(select, timeStamp, pose, strOfflineDescriptorPath, true);
            // 仍然需要加载图像用于可视化
            image_show = imread(strDatasetPath_front + files_front[select], IMREAD_COLOR);
            image_show_undistort = image_show.clone();
            if (!image_show.empty()) {
                image_show = UndistortImage(image_show, camera1.first, camera1.second, camParams1, ImSizeFinal);
            }
        } else {
            // 在线提取特征模式 
            cv::Mat image = imread(strDatasetPath_front + files_front[select], IMREAD_GRAYSCALE);
            if (!image.empty() && !crop_enabled) {
                image = UndistortImage(image, camera1.first, camera1.second, camParams1, ImSizeFinal);
            }else if (!image.empty() && crop_enabled) {
                image = CropImage(image, 960,0,1920,1440);
            }

            image_show = imread(strDatasetPath_front + files_front[select], IMREAD_COLOR);
            image_show_undistort = image_show.clone();

            if (!image_show.empty() && !crop_enabled) {
            image_show = UndistortImage(image_show, camera1.first, camera1.second, camParams1, ImSizeFinal);
            }else if (!image_show.empty() && crop_enabled) {
                image_show = CropImage(image_show, 960,0,1920,1440);
            }
            pKFHF = new KeyFrameNetVlad(select, image, pModel, timeStamp, pose);
        }

        // 更新Query位置（每次都调用，记录红色点走过的路径）
        UpdateQueryVisualization(gtPoses, select);

        Eigen::Vector3d query_pos = pose.block<3,1>(0,3);
        const double search_radius = 5.0; 
        const double time_threshold = 30.0; 

        std::vector<nanoflann::ResultItem<unsigned int, double>> temp_matches;
        nanoflann::SearchParameters params;
        kdtree->radiusSearch(query_pos.data(), search_radius * search_radius, temp_matches, params);
        
        std::vector<size_t> valid_indices;
        for (const auto& m : temp_matches) {
            size_t idx = m.first;
            // 真值要求有两个：一个是事件差大于30s, 一个是候选帧的id需要至少与查询帧的id靠后300帧
            if (std::abs(timeStamp - vKeyFrameDB[idx]->timeStamp) > time_threshold && 
                vKeyFrameDB[idx]->mnFrameId < pKFHF->mnFrameId - 300)
                valid_indices.push_back(idx);
        }

        // Eigen回环检测
        auto t1 = chrono::steady_clock::now();
        bool use_rear = false; 
        auto res = GetNCandidateLoopFrameEigen(pKFHF, vKeyFrameDB, 5, use_rear);
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
        
        if (res.size() > 0 && use_rear) {
            std::cout << "Found " << res.size() << " rear candidates, updating Pangolin..." << std::endl;
        }

        string strDatasetPath;
        std::vector<string> files;
        if(use_rear){
            strDatasetPath = strDatasetPath_rear;
            files = files_rear;
        } else {
            strDatasetPath = strDatasetPath_front;
            files = files_front;
        }
        // KDTree候选
        for (size_t i = 0; i < 3; ++i) {
            if (i < valid_indices.size()) {
                int kf_idx = valid_indices[i];
                imgs[1+i] = imread(strDatasetPath + files[vKeyFrameDB[kf_idx]->mnFrameId], IMREAD_COLOR);
                if(show_undistort && !crop_enabled){
                    // 使用去畸变模式且不使用裁剪模式
                    imgs[1+i] = UndistortImage(imgs[1+i], camera1.first, camera1.second, camParams1, ImSizeFinal);
                }else if (!crop_enabled){
                    // 不使用去畸变模式且不使用裁剪模式
                    cv::resize(imgs[1+i], imgs[1+i], ImSizeFinal);
                }else if (crop_enabled) {
                    // 使用裁剪模式
                    imgs[1+i] = CropImage(imgs[1+i], 960,0,1920,1440);
                    cv::resize(imgs[1+i], imgs[1+i], ImSizeFinal);
                }
                texts[1+i] = "KDTree " + std::to_string(i+1) + ": " + std::to_string((int)vKeyFrameDB[kf_idx]->mnFrameId) + ", t=" + std::to_string(vKeyFrameDB[kf_idx]->timeStamp);
            } else {
                imgs[1+i] = cv::Mat::zeros(ImSizeFinal, CV_8UC3);
                texts[1+i] = "KDTree " + std::to_string(i+1) + ": None";
            }
        }
        // Eigen候选
        for (size_t i = 0; i < 3; ++i) {
            if (i < res.size()) {
                float score;
                if(use_rear){
                    score = res[i]->mPlaceRecognitionScore_rear;
                    imgs[4+i] = imread(strDatasetPath + files[res[i]->mnFrameId], IMREAD_COLOR);
                    if(show_undistort && !crop_enabled){
                        // 使用去畸变模式且不使用裁剪模式
                        imgs[4+i] = UndistortImage(imgs[4+i], camera2.first, camera2.second, camParams2, ImSizeFinal);
                    }else if (!crop_enabled){
                        // 不使用去畸变模式且不使用裁剪模式
                        cv::resize(imgs[4+i], imgs[4+i], ImSizeFinal);
                    }else if (crop_enabled) {
                        // 使用裁剪模式
                        imgs[4+i] = CropImage(imgs[4+i], 480,0,960,720);
                        cv::resize(imgs[4+i], imgs[4+i], ImSizeFinal);
                    }
                    texts[4+i] = "Eigen Rear " + std::to_string(i+1) + ": " + std::to_string((int)res[i]->mnFrameId) + ", score=" + std::to_string(score);
                }else{
                    score = res[i]->mPlaceRecognitionScore_front;
                    imgs[4+i] = imread(strDatasetPath + files[res[i]->mnFrameId], IMREAD_COLOR);
                    if(show_undistort && !crop_enabled){
                        // 使用去畸变模式且不使用裁剪模式
                        imgs[4+i] = UndistortImage(imgs[4+i], camera1.first, camera1.second, camParams1, ImSizeFinal);
                    }else if (!crop_enabled){
                        // 不使用去畸变模式且不使用裁剪模式
                        cv::resize(imgs[4+i], imgs[4+i], ImSizeFinal);
                    }else if (crop_enabled) {
                        // 使用裁剪模式
                        imgs[4+i] = CropImage(imgs[4+i], 960,0,1920,1440);
                        cv::resize(imgs[4+i], imgs[4+i], ImSizeFinal);
                    }
                    texts[4+i] = "Eigen Front " + std::to_string(i+1) + ": " + std::to_string((int)res[i]->mnFrameId) + ", score=" + std::to_string(score);
                }
                
            } else {
                imgs[4+i] = cv::Mat::zeros(ImSizeFinal, CV_8UC3);
                texts[4+i] = "Eigen " + std::to_string(i+1) + ": None";
            }
        }
        
        // 在每张图上加文字
        for (int i = 0; i < 7; ++i) {
            cv::putText(imgs[i], texts[i], cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
        }
        
        // 拼接图像
        cv::Mat left_all;
        cv::Mat left_distort = imgs[0];
        cv::Mat left_undistort;
        cv::resize(image_show_undistort, left_undistort, ImSizeFinal);
        cv::vconcat(std::vector<cv::Mat>{left_distort, left_undistort}, left_all);

        cv::Mat right_row1, right_row2, right_all, all;
        cv::hconcat(std::vector<cv::Mat>{imgs[1], imgs[2], imgs[3]}, right_row1);
        cv::hconcat(std::vector<cv::Mat>{imgs[4], imgs[5], imgs[6]}, right_row2);
        cv::vconcat(right_row1, right_row2, right_all);
        
        cv::hconcat(std::vector<cv::Mat>{left_all, right_all}, all);
        cv::namedWindow("Compare Candidates", cv::WINDOW_NORMAL);
        cv::resizeWindow("Compare Candidates", 1600, 720);
        
        if(valid_indices.size() > 0 || res.size() > 0){
            cv::imshow("Compare Candidates", all);
            cv::waitKey(500);
            
            // 显示完成后清空窗口
            // cv::Mat blank = cv::Mat::zeros(all.size(), all.type());
            // cv::imshow("Compare Candidates", blank);
            // cv::waitKey(1); // 短暂等待确保窗口更新
        }

        if (select >= end - 1) {
            std::cout << "Reached the end of dataset, exiting..." << std::endl;
            break;
        }

        // 检测到回环后5s内不在检测
        if(valid_indices.size() >0 && res.size() >= 3){
            select += 50;
        }
    }

    system("pause");
    return 0;
}

// void TestTwoImages()
// {
//             string image_front_path = "/home/xihuidong/codetree/repo/visual_mapping_test/fisheye1/1747822581.570967.png";
//             string image_rear_path = "/home/xihuidong/codetree/repo/visual_mapping_test/fisheye3/1747822598.340147.png";

//             cv::Mat image = imread(image_front_path, IMREAD_GRAYSCALE);
//             image = UndistortImage(image, camera1.first, camera1.second, camParams1, ImSizeFinal);
//             cv::Mat image_show = imread(image_front_path, IMREAD_COLOR);
//             image_show = UndistortImage(image_show, camera1.first, camera1.second, camParams1, ImSizeFinal);

//             cv::Mat image_rear = imread(image_rear_path, IMREAD_GRAYSCALE);
//             image_rear = UndistortImage(image_rear, camera2.first, camera2.second, camParams2, ImSizeFinal);
//             cv::Mat image_show_rear = imread(image_rear_path, IMREAD_COLOR);
//             image_show_rear = UndistortImage(image_show_rear, camera2.first, camera2.second, camParams2, ImSizeFinal);

//             cv::imshow("Image Front", image_show);
//             cv::imshow("Image Rear", image_show_rear);
//             cv::waitKey(0);

//             KeyFrameNetVlad *pKFHF = new KeyFrameNetVlad(select, image, pModel, timeStamp, pose);
//             KeyFrameNetVlad *pKFHF_rear = new KeyFrameNetVlad(select, image_rear, pModel, timeStamp, pose);

//             Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
//             pkf_q_front(
//             pKFHF->mGlobalDescriptors_front.ptr<float>(), pKFHF->mGlobalDescriptors_front.rows, 
//             pKFHF->mGlobalDescriptors_front.cols);

//             Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> 
//             pkf_m_rear(
//             pKFHF_rear->mGlobalDescriptors_front.ptr<float>(), pKFHF_rear->mGlobalDescriptors_front.rows, 
//             pKFHF_rear->mGlobalDescriptors_front.cols);
//             float current_score = (pkf_q_front - pkf_m_rear).norm();
//             std::cout << "Current query score: " << current_score << std::endl;
//             return ;
// }