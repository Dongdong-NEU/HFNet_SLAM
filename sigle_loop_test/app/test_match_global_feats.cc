#include "common.h"
#include <sys/stat.h>  // for mkdir
#include <sys/types.h>
#include <cerrno>  // for errno
#include "loop_visual.h"
#include "fisheye_utils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <iostream>
#include <cassert>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace DeepRoute;


bool CreateDirectoryRecursive(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return (info.st_mode & S_IFDIR) != 0;  // 目录已存在
    }
    
    // 递归创建父目录
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        std::string parent = path.substr(0, pos);
        if (!CreateDirectoryRecursive(parent)) {
            return false;
        }
    }
    
    // 创建当前目录
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// ============================================================================
// 配置结构体
// ============================================================================

struct LoopDetectionConfig {
    // 路径配置
    string dataset_front_path;
    string dataset_rear_path;
    string model_path;
    string gt_poses_path;
    string cameras_cfg_path;
    string config_yaml_path;
    string offline_descriptor_path;
    
    // 模型配置
    string onnx_model_name;
    string engine_cache_name;
    
    // 运行模式
    bool use_offline_descriptor;
    bool enable_crop;
    bool enable_visualization;
    
    // 图像裁剪参数
    struct CropParams {
        int x, y, width, height;
    } front_crop, rear_crop;
    
    // 回环检测参数
    double search_radius;
    double time_threshold;
    int min_frame_distance;
    int num_candidates;
    
    // 初始化默认值
    LoopDetectionConfig() 
        : use_offline_descriptor(false)
        , enable_crop(true)
        , enable_visualization(true)
        , search_radius(5.0)
        , time_threshold(30.0)
        , min_frame_distance(300)
        , num_candidates(5)
    {
        // 前置相机裁剪参数
        front_crop = {960, 0, 1920, 1440};
        // 后置相机裁剪参数
        rear_crop = {480, 0, 960, 720};
    }
};

// ============================================================================
// 函数声明
// ============================================================================

// 图像处理
cv::Mat LoadAndProcessImage(
    const string& image_path,
    bool enable_crop,
    const LoopDetectionConfig::CropParams& crop_params,
    const pair<cv::Mat, cv::Mat>& undistort_maps,
    const FisheyeCameraParams& cam_params,
    const cv::Size& target_size);

cv::Mat LoadImageForVisualization(
    const string& image_path,
    bool enable_crop,
    const LoopDetectionConfig::CropParams& crop_params,
    const cv::Size& target_size);

// 数据库构建
bool BuildKeyFrameDatabase(
    const LoopDetectionConfig& config,
    const vector<string>& files_front,
    const vector<string>& files_rear,
    const vector<double>& times,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses,
    const pair<cv::Mat, cv::Mat>& undistort_maps_front,
    const pair<cv::Mat, cv::Mat>& undistort_maps_rear,
    const FisheyeCameraParams& cam_params_front,
    const FisheyeCameraParams& cam_params_rear,
    const cv::Size& target_size,
    EigenPlacesExtractor* model,
    KeyFrameDB& keyframe_db,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& keyframe_positions);

// 回环检测
void RunLoopDetection(
    const LoopDetectionConfig& config,
    const vector<string>& files_front,
    const vector<string>& files_rear,
    const vector<double>& times,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses,
    const pair<cv::Mat, cv::Mat>& undistort_maps_front,
    const pair<cv::Mat, cv::Mat>& undistort_maps_rear,
    const FisheyeCameraParams& cam_params_front,
    const FisheyeCameraParams& cam_params_rear,
    const cv::Size& target_size,
    EigenPlacesExtractor* model,
    const KeyFrameDB& keyframe_db,
    const std::unique_ptr<KDTreeType>& kdtree);

// 可视化
void DisplayComparisonImages(
    int query_frame_id,
    const KeyFrameNetVlad* query_frame,
    const cv::Mat& query_image,
    const std::vector<size_t>& gt_candidates,
    const KeyFrameDB& detected_candidates,
    const KeyFrameDB& keyframe_db,
    const LoopDetectionConfig& config,
    const vector<string>& files_front,
    const vector<string>& files_rear,
    const cv::Size& target_size,
    bool use_rear);

// ============================================================================
// 图像处理辅助函数
// ============================================================================

/**
 * 加载并处理图像（用于特征提取）
 */
cv::Mat LoadAndProcessImage(
    const string& image_path,
    bool enable_crop,
    const LoopDetectionConfig::CropParams& crop_params,
    const pair<cv::Mat, cv::Mat>& undistort_maps,
    const FisheyeCameraParams& cam_params,
    const cv::Size& target_size)
{
    cv::Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return cv::Mat();
    }
    
    if (enable_crop) {
        image = CropImage(image, crop_params.x, crop_params.y, 
                         crop_params.width, crop_params.height);
    } else {
        image = UndistortImage(image, undistort_maps.first, undistort_maps.second, 
                              cam_params, target_size);
    }
    
    return image;
}

/**
 * 加载图像用于可视化
 */
cv::Mat LoadImageForVisualization(
    const string& image_path,
    bool enable_crop,
    const LoopDetectionConfig::CropParams& crop_params,
    const cv::Size& target_size)
{
    cv::Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        return cv::Mat();
    }
    
    if (enable_crop) {
        image = CropImage(image, crop_params.x, crop_params.y, 
                         crop_params.width, crop_params.height);
    }
    
    cv::resize(image, image, target_size);
    return image;
}

// ============================================================================
// 关键帧数据库构建
// ============================================================================

bool BuildKeyFrameDatabase(
    const LoopDetectionConfig& config,
    const vector<string>& files_front,
    const vector<string>& files_rear,
    const vector<double>& times,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses,
    const pair<cv::Mat, cv::Mat>& undistort_maps_front,
    const pair<cv::Mat, cv::Mat>& undistort_maps_rear,
    const FisheyeCameraParams& cam_params_front,
    const FisheyeCameraParams& cam_params_rear,
    const cv::Size& target_size,
    EigenPlacesExtractor* model,
    KeyFrameDB& keyframe_db,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& keyframe_positions)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Building keyframe database..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto db_start = chrono::steady_clock::now();
    
    const int step = 1;
    int keyframe_count = 0;


    // string t_t =  "/1763019410.065747.png";
    // string n_b =  "/1763019537.065877.png";
    // string t_b =  "/1763019655.065983.png";
    // string n_t =  "/1763019750.066070.png";
    // bool find_t_t = false;
    // bool find_n_b = false;
    // bool find_t_b = false;
    // bool find_n_t = false;
    // auto it = std::find(files_front.begin(), files_front.end(), t_t);
    // if(it != files_front.end()) {
    //     find_t_t = true;
    // }
    // it = std::find(files_front.begin(), files_front.end(), n_b);
    // if(it != files_front.end()) {
    //     find_n_b = true;
    // }
    // it = std::find(files_front.begin(), files_front.end(), t_b);
    // if(it != files_front.end()) {
    //     find_t_b = true;
    // }
    // it = std::find(files_front.begin(), files_front.end(), n_t);
    // if(it != files_front.end()) {
    //     find_n_t = true;
    // }
    // std::cout << "find_t_t: " << find_t_t << std::endl;
    // std::cout << "find_n_b: " << find_n_b << std::endl;
    // std::cout << "find_t_b: " << find_t_b << std::endl;
    // std::cout << "find_n_t: " << find_n_t << std::endl;

    int end_mapping_frame = 3100;
    
    for (size_t i = 0; i < end_mapping_frame; i += step) {

        Eigen::Matrix4d pose_last = Eigen::Matrix4d::Identity();
        KeyFrameNetVlad* keyframe = nullptr;
        if(i > 0) {
            Eigen::Vector3d pose_last_t = pose_last.block<3,1>(0,3);
            Eigen::Vector3d pose_current_t = poses[i].block<3,1>(0,3);
            double trans_dist = (pose_current_t - pose_last_t).norm();
            if (trans_dist < 1.0) {
                continue;
            }
        }
        pose_last = poses[i];
        if (config.use_offline_descriptor) {
            // 离线描述子模式
            auto load_start = chrono::steady_clock::now();
            keyframe = new KeyFrameNetVlad(i, times[i], poses[i], config.offline_descriptor_path);
            auto load_time = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - load_start).count();
            std::cout << "Frame " << i << " - Offline loading: " << load_time << "ms" << std::endl;
        } else {
            // 在线特征提取模式
            string front_path = config.dataset_front_path + files_front[i];
            string rear_path = config.dataset_rear_path + files_rear[i];
            
            auto img_start = chrono::steady_clock::now();
            cv::Mat image_front = LoadAndProcessImage(
                front_path, config.enable_crop, config.front_crop,
                undistort_maps_front, cam_params_front, target_size);
            cv::Mat image_rear = LoadAndProcessImage(
                rear_path, config.enable_crop, config.rear_crop,
                undistort_maps_rear, cam_params_rear, target_size);
            auto img_time = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - img_start).count();
            
            if (image_front.empty() || image_rear.empty()) {
                std::cerr << "Failed to load images at frame " << i << std::endl;
                continue;
            }
            
            auto feat_start = chrono::steady_clock::now();
            keyframe = new KeyFrameNetVlad(i, image_front, image_rear, model, times[i], poses[i]);
            auto feat_time = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - feat_start).count();
        }
        
        keyframe_db.emplace_back(keyframe);
        keyframe_positions.push_back(poses[i].block<3,1>(0,3));
        keyframe_count++;
        pose_last = poses[i];

        // 每 100 帧显示一次进度
        if (keyframe_count % 100 == 0 || keyframe_count == 1) {
            std::cout << "  进度: " << keyframe_count << "/" << end_mapping_frame 
                     << " (" << (keyframe_count * 100 / end_mapping_frame) << "%)" << std::endl;
        }
    }
    
    auto db_time = chrono::duration_cast<chrono::seconds>(
        chrono::steady_clock::now() - db_start).count();
    
    std::cout << "\nDatabase built: " << keyframe_count << " keyframes in " 
             << db_time << " seconds" << std::endl;
    
    return keyframe_count > 0;
}

// ============================================================================
// 回环检测主循环
// ============================================================================

void RunLoopDetection(
    const LoopDetectionConfig& config,
    const vector<string>& files_front,
    const vector<string>& files_rear,
    const vector<double>& times,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& poses,
    const pair<cv::Mat, cv::Mat>& undistort_maps_front,
    const pair<cv::Mat, cv::Mat>& undistort_maps_rear,
    const FisheyeCameraParams& cam_params_front,
    const FisheyeCameraParams& cam_params_rear,
    const cv::Size& target_size,
    EigenPlacesExtractor* model,
    const KeyFrameDB& keyframe_db,
    const std::unique_ptr<KDTreeType>& kdtree)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting loop detection..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    int frame_idx = 4820;
    bool use_rear = false;
    int end_frame = 0;

    while (frame_idx < static_cast<int>(files_front.size()) - 1) {
        // 检查暂停状态
        if (g_viewer && g_viewer->IsPaused()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        frame_idx++;
        
        double timestamp = times[frame_idx];
        Eigen::Matrix4d pose = poses[frame_idx];
        
        // ========== 加载查询帧 ==========
        KeyFrameNetVlad* query_frame = nullptr;
        KeyFrameNetVlad* query_frame_rear = nullptr;
        cv::Mat image_for_display;
        cv::Mat image_for_display_rear;
        
        if (config.use_offline_descriptor) {
            query_frame = new KeyFrameNetVlad(frame_idx, timestamp, pose, 
                                             config.offline_descriptor_path, true);
            image_for_display = imread(config.dataset_front_path + files_front[frame_idx], 
                                      IMREAD_COLOR);
        } else {
            string front_path = config.dataset_front_path + files_front[frame_idx];
            string rear_path = config.dataset_rear_path + files_rear[frame_idx];
            cv::Mat image = LoadAndProcessImage(
                front_path, config.enable_crop, config.front_crop,
                undistort_maps_front, cam_params_front, target_size);
            cv::Mat image_rear = LoadAndProcessImage(
                rear_path, config.enable_crop, config.rear_crop,
                undistort_maps_rear, cam_params_rear, target_size);
            query_frame = new KeyFrameNetVlad(frame_idx, image, model, timestamp, pose);
            query_frame_rear = new KeyFrameNetVlad(frame_idx, image_rear, model, timestamp, pose);
            image_for_display = imread(front_path, IMREAD_COLOR);
            image_for_display_rear = imread(rear_path, IMREAD_COLOR);
        }
        
        if (!image_for_display.empty() && !image_for_display_rear.empty()) {
            if (config.enable_crop) {
                image_for_display = CropImage(image_for_display, 
                    config.front_crop.x, config.front_crop.y,
                    config.front_crop.width, config.front_crop.height);
                image_for_display_rear = CropImage(image_for_display_rear, 
                    config.rear_crop.x, config.rear_crop.y,
                    config.rear_crop.width, config.rear_crop.height);
            }
            cv::resize(image_for_display, image_for_display, target_size);
            cv::resize(image_for_display_rear, image_for_display_rear, target_size);
        }
        
        // 更新查询位置可视化
        UpdateQueryVisualization(poses, frame_idx);
        
        // ========== KDTree空间搜索 ==========
        Eigen::Vector3d query_pos = pose.block<3,1>(0,3);
        std::vector<nanoflann::ResultItem<unsigned int, double>> spatial_matches;
        nanoflann::SearchParameters params;
        kdtree->radiusSearch(query_pos.data(), 
                            config.search_radius * config.search_radius, 
                            spatial_matches, params);
        
        // 过滤候选帧
        std::vector<size_t> ground_truth_candidates;
        for (const auto& match : spatial_matches) {
            size_t idx = match.first;
            if (std::abs(timestamp - keyframe_db[idx]->timeStamp) > config.time_threshold && 
                keyframe_db[idx]->mnFrameId < query_frame->mnFrameId - config.min_frame_distance) {
                ground_truth_candidates.push_back(idx);
            }
        }
        
        // ========== EigenPlaces回环检测 ==========
        auto t_start = chrono::steady_clock::now();
        auto detected_candidates = GetNCandidateLoopFrameEigen(
            query_frame, keyframe_db, config.num_candidates, use_rear);
        auto query_time = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - t_start).count();
        
        std::cout << "\nFrame " << frame_idx << " - Query time: " << query_time << "ms" << std::endl;
        std::cout << "  Ground truth candidates: " << ground_truth_candidates.size() 
                 << ", Detected: " << detected_candidates.size() << std::endl;
        bool debug_save_image = true;
        if(debug_save_image && ground_truth_candidates.size() > 0 && detected_candidates.size() == 0) {
            string debug_dir = "/home/xihuidong/Documents/workspace/HFNet_SLAM-main/sigle_loop_test/debug_image/" + std::to_string(frame_idx);
            CreateDirectoryRecursive(debug_dir);
            string debug_save_query_path = debug_dir + "/" + "front_query_" + files_front[frame_idx].substr(1);
            string debug_save_query_rear_path = debug_dir + "/" + "rear_query_" + files_rear[frame_idx].substr(1);
            string debug_save_gt_path = debug_dir + "/";
            string debug_save_gt_rear_path = debug_dir + "/";

            string image_path = config.dataset_front_path + files_front[frame_idx];
            string image_path_rear = config.dataset_rear_path + files_rear[frame_idx];
            cv::Mat image_front_query = cv::imread(image_path, IMREAD_COLOR);
            cv::Mat image_rear_query = cv::imread(image_path_rear, IMREAD_COLOR);
            cv::imwrite(debug_save_query_path , image_front_query);
            cv::imwrite(debug_save_query_rear_path , image_rear_query);
            for(size_t i = 0; i < ground_truth_candidates.size(); i++) {
                size_t gt_idx = ground_truth_candidates[i];
                
                // 计算候选帧与查询帧的距离
                Eigen::Vector3d gt_pos = keyframe_db[gt_idx]->curPose.block<3,1>(0,3);
                double distance = (gt_pos - query_pos).norm();
                
                string image_path_gt = config.dataset_front_path + files_front[gt_idx];
                string image_path_gt_rear = config.dataset_rear_path + files_rear[gt_idx];
                cv::Mat image_gt = cv::imread(image_path_gt, IMREAD_COLOR);
                cv::Mat image_gt_rear = cv::imread(image_path_gt_rear, IMREAD_COLOR);
                
                // 格式化距离（保留2位小数）
                char dist_str[32];
                snprintf(dist_str, sizeof(dist_str), "_dist_%.2fm", distance);
                
                cv::imwrite(debug_save_gt_path + std::to_string(i) + "_front_gt" + dist_str + "_" + files_front[gt_idx].substr(1), image_gt);
                cv::imwrite(debug_save_gt_rear_path + std::to_string(i) + "_rear_gt" + dist_str + "_" + files_rear[gt_idx].substr(1), image_gt_rear);
                
                if(i == 2){
                    break;
                }
            }
        }
        // ========== 更新可视化 ==========
        if (config.enable_visualization) {
            UpdateVisualization(poses, frame_idx, detected_candidates, 
                              ground_truth_candidates, keyframe_db);
            
            // 显示对比图像（如果检测到回环）
            if (!ground_truth_candidates.empty() || !detected_candidates.empty()) {
                DisplayComparisonImages(frame_idx, query_frame, image_for_display,
                                       ground_truth_candidates, detected_candidates,
                                       keyframe_db, config, files_front, files_rear, target_size,
                                       use_rear);
            }
        }
        
        // 检测到回环后跳过一段时间
        if (!ground_truth_candidates.empty() && detected_candidates.size() >= 3) {
            frame_idx += 10;  // 跳过5秒（假设10fps）
        }
        
        delete query_frame;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Loop detection completed!" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// 对比图像显示
// ============================================================================

void DisplayComparisonImages(
    int query_frame_id,
    const KeyFrameNetVlad* query_frame,
    const cv::Mat& query_image,
    const std::vector<size_t>& gt_candidates,
    const KeyFrameDB& detected_candidates,
    const KeyFrameDB& keyframe_db,
    const LoopDetectionConfig& config,
    const vector<string>& files_front,
    const vector<string>& files_rear,
    const cv::Size& target_size,
    bool use_rear)
{
    // TODO:注意：这里是为了测试，所以强制使用后置相机
    bool tmp_use_rear = use_rear;
    use_rear = true;
    
    std::vector<cv::Mat> images(7);
    std::vector<std::string> labels(7);
    
    // 查询帧
    images[0] = query_image.clone();
    labels[0] = "Query: " + std::to_string(query_frame_id) + 
                ", t=" + std::to_string(query_frame->timeStamp);
    
    // KDTree候选（ground truth）
    for (size_t i = 0; i < 3; ++i) {
        if (i < gt_candidates.size()) {
            int kf_idx = gt_candidates[i];
            string image_path = (use_rear ? config.dataset_rear_path : config.dataset_front_path) 
                              + (use_rear ? files_rear[keyframe_db[kf_idx]->mnFrameId] : files_front[keyframe_db[kf_idx]->mnFrameId]);
            images[1+i] = LoadImageForVisualization(
                image_path, config.enable_crop, 
                use_rear ? config.rear_crop : config.front_crop, target_size);
            labels[1+i] = "GT " + std::to_string(i+1) + ": " + 
                         std::to_string((int)keyframe_db[kf_idx]->mnFrameId) +
                         ", t=" + std::to_string(keyframe_db[kf_idx]->timeStamp);
        } else {
            images[1+i] = cv::Mat::zeros(target_size, CV_8UC3);
            labels[1+i] = "GT " + std::to_string(i+1) + ": None";
        }
    }
    use_rear = tmp_use_rear;
    // EigenPlaces检测候选
    for (size_t i = 0; i < 3; ++i) {
        if (i < detected_candidates.size()) {
            float score = use_rear ? detected_candidates[i]->mPlaceRecognitionScore_rear 
                                   : detected_candidates[i]->mPlaceRecognitionScore_front;
            string image_path = (use_rear ? config.dataset_rear_path : config.dataset_front_path) 
                              + (use_rear ? files_rear[detected_candidates[i]->mnFrameId] : files_front[detected_candidates[i]->mnFrameId]);
            images[4+i] = LoadImageForVisualization(
                image_path, config.enable_crop,
                use_rear ? config.rear_crop : config.front_crop, target_size);
            labels[4+i] = "Detected " + std::string(use_rear ? "rear" : "front") + " " + std::to_string(i+1) + ": " + 
                         std::to_string((int)detected_candidates[i]->mnFrameId) +
                         ", score=" + std::to_string(score);
        } else {
            images[4+i] = cv::Mat::zeros(target_size, CV_8UC3);
            labels[4+i] = "Detected " + std::string(use_rear ? "rear" : "front") + " " + std::to_string(i+1) + ": None";
        }
    }
    
    // 添加文字标签
    for (int i = 0; i < 7; ++i) {
        cv::putText(images[i], labels[i], cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
    }
    
    // 拼接并显示
    cv::Mat row1, row2, display;
    cv::hconcat(std::vector<cv::Mat>{images[0], images[1], images[2], images[3]}, row1);
    cv::hconcat(std::vector<cv::Mat>{cv::Mat::zeros(target_size, CV_8UC3), 
                                     images[4], images[5], images[6]}, row2);
    cv::vconcat(row1, row2, display);
    
    cv::namedWindow("Loop Detection Results", cv::WINDOW_NORMAL);
    cv::resizeWindow("Loop Detection Results", 1600, 800);
    cv::imshow("Loop Detection Results", display);
    if(detected_candidates.size() > 0) {
        cv::waitKey(0);
    }else{
    cv::waitKey(500);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv)
{
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));
    
    // ========== 参数解析 ==========
    if (argc != 7 && argc != 8 && argc != 9) {
        std::cerr << "\n用法: " << argv[0] << " <dataset_front> <dataset_rear> "
                  << "<model_path> <gt_poses> <camera_cfg> <config_yaml> "
                  << "[offline_descriptor_path] [map_path]" << std::endl;
        std::cerr << "\n可选参数说明:" << std::endl;
        std::cerr << "  offline_descriptor_path - 预计算描述子路径（用于离线模式）" << std::endl;
        std::cerr << "  map_path - 地图文件路径" << std::endl;
        std::cerr << "    * 如果文件已存在：直接加载地图，跳过构建" << std::endl;
        std::cerr << "    * 如果文件不存在：构建地图后保存到该路径" << std::endl;
        std::cerr << "\n示例：" << std::endl;
        std::cerr << "  # 构建地图并保存" << std::endl;
        std::cerr << "  " << argv[0] << " <参数...> /path/to/map.bin" << std::endl;
        std::cerr << "  # 加载已有地图" << std::endl;
        std::cerr << "  " << argv[0] << " <参数...> /path/to/existing_map.bin" << std::endl;
        return -1;
    }
    
    LoopDetectionConfig config;
    config.dataset_front_path = argv[1];
    config.dataset_rear_path = argv[2];
    config.model_path = argv[3];
    config.gt_poses_path = argv[4];
    config.cameras_cfg_path = argv[5];
    config.config_yaml_path = argv[6];
    
    // 解析可选参数
    string map_file_path = "";
    if (argc == 8) {
        // 可能是 offline_descriptor_path 或 map_path
        string arg7 = argv[7];
        // 判断是否为地图文件（以.bin结尾）
        if (arg7.size() > 4 && arg7.substr(arg7.size() - 4) == ".bin") {
            map_file_path = arg7;
            config.use_offline_descriptor = false;
        } else {
            config.use_offline_descriptor = true;
            config.offline_descriptor_path = arg7;
        }
    } else if (argc == 9) {
        config.use_offline_descriptor = true;
        config.offline_descriptor_path = argv[7];
        map_file_path = argv[8];
    } else {
        config.use_offline_descriptor = false;
    }
    
    // 设置模型文件名
    config.onnx_model_name = "/eigenplaces_resnet50_fixedshape_300_400_GPU_simplified.onnx";
    config.engine_cache_name = "/eigenplaces_resnet50_fixedshape_300_400_GPU_simplified.engine";
    
    // ========== 加载配置和数据 ==========
    cv::Size target_size;
    LoadConfigYaml(config.config_yaml_path, target_size);
    std::cout << "Target image size: " << target_size.width << "x" << target_size.height << std::endl;
    
    // 加载相机参数
    FisheyeCameraParams cam_params_front, cam_params_rear;
    if (!LoadFisheyeCameraParams(config.cameras_cfg_path, "camera_1", cam_params_front)) {
        std::cerr << "Failed to load camera_1 parameters" << std::endl;
        return -1;
    }
    if (!LoadFisheyeCameraParams(config.cameras_cfg_path, "camera_4", cam_params_rear)) {
        std::cerr << "Failed to load camera_4 parameters" << std::endl;
        return -1;
    }
    
    pair<cv::Mat, cv::Mat> undistort_maps_front = UndistortFisheyeParam(cam_params_front);
    pair<cv::Mat, cv::Mat> undistort_maps_rear = UndistortFisheyeParam(cam_params_rear);
    
    // 加载数据集
    auto aligned_data = AlignDualCameraDataToTrajectory(
        config.dataset_front_path, config.dataset_rear_path, config.gt_poses_path);
    
    if (aligned_data.files_front.empty() || aligned_data.poses.empty()) {
        std::cerr << "Failed to load dataset or poses" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << aligned_data.files_front.size() << " aligned image pairs" << std::endl;
    std::cout << "Loaded " << aligned_data.poses.size() << " ground truth poses" << std::endl;
    assert(aligned_data.files_front.size() == aligned_data.poses.size());
    
    // ========== 检查是否加载已有地图 ==========
    KeyFrameDB keyframe_db;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keyframe_positions;
    bool map_loaded = false;
    
    if (!map_file_path.empty()) {
        // 检查地图文件是否存在
        std::ifstream map_check(map_file_path);
        if (map_check.good()) {
            map_check.close();
            std::cout << "\n检测到已有地图文件，正在加载..." << std::endl;
            if (LoadKeyFrameDatabase(map_file_path, keyframe_db, keyframe_positions)) {
                map_loaded = true;
                std::cout << "地图加载成功，跳过构建阶段！" << std::endl;
            } else {
                std::cerr << "地图加载失败，将重新构建" << std::endl;
            }
        } else {
            std::cout << "\n地图文件不存在，将构建新地图并保存到: " << map_file_path << std::endl;
        }
    }
    
    // ========== 初始化模型 ==========
    // 注意：即使加载了地图，也需要初始化模型用于查询帧的特征提取
    EigenPlacesExtractor* model = nullptr;
    
    if (!map_loaded) {
        // 需要构建地图，必须初始化模型（除非使用离线描述子）
        if (!config.use_offline_descriptor) {
            std::cout << "\nInitializing EigenPlaces model..." << std::endl;
            string onnx_path = config.model_path + config.onnx_model_name;
            string engine_path = config.model_path + config.engine_cache_name;
            model = InitEigenPlacesModel(onnx_path, engine_path, target_size);
            if (!model || !model->IsValid()) {
                std::cerr << "Failed to initialize model" << std::endl;
                return -1;
            }
        }
        
        // ========== 构建关键帧数据库 ==========
        if (!BuildKeyFrameDatabase(
            config, aligned_data.files_front, aligned_data.files_rear, 
            aligned_data.times, aligned_data.poses,
            undistort_maps_front, undistort_maps_rear,
            cam_params_front, cam_params_rear, target_size,
            model, keyframe_db, keyframe_positions))
        {
            std::cerr << "Failed to build keyframe database" << std::endl;
            return -1;
        }
        
        // ========== 保存地图（如果指定了地图路径） ==========
        if (!map_file_path.empty()) {
            if (!SaveKeyFrameDatabase(map_file_path, keyframe_db)) {
                std::cerr << "警告：地图保存失败，但继续运行" << std::endl;
            }
        }
    } else {
        // 地图已加载，但仍需初始化模型用于查询帧特征提取（除非使用离线描述子）
        if (!config.use_offline_descriptor) {
            std::cout << "\n地图已加载，正在初始化模型用于查询帧..." << std::endl;
            string onnx_path = config.model_path + config.onnx_model_name;
            string engine_path = config.model_path + config.engine_cache_name;
            model = InitEigenPlacesModel(onnx_path, engine_path, target_size);
            if (!model || !model->IsValid()) {
                std::cerr << "Failed to initialize model for query frames" << std::endl;
                return -1;
            }
        }
    }
    
    // ========== 验证关键帧数量 ==========
    if (keyframe_db.size() <= 300) {
        std::cerr << "Too few keyframes: " << keyframe_db.size() << " (need > 300)" << std::endl;
        return -1;
    }
    
    // ========== 构建KD树 ==========
    KeyFramePointCloud cloud;
    cloud.pts = keyframe_positions;
    auto kdtree = std::make_unique<KDTreeType>(3, cloud, 
                                               nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree->buildIndex();
    std::cout << "KD-tree built with " << keyframe_positions.size() << " points" << std::endl;
    
    // ========== 启动可视化 ==========
    if (config.enable_visualization) {
        StartVisualization();
        if (g_viewer) {
            g_viewer->UpdateTrajectory(aligned_data.poses);
            std::cout << "Visualization initialized with trajectory" << std::endl;
        }
    }
    
    // ========== 运行回环检测 ==========
    RunLoopDetection(
        config, aligned_data.files_front, aligned_data.files_rear,
        aligned_data.times, aligned_data.poses,
        undistort_maps_front, undistort_maps_rear, cam_params_front, cam_params_rear, target_size,
        model, keyframe_db, kdtree);
    
    // ========== 清理 ==========
    // 先停止可视化线程，避免访问即将被删除的数据
    if (config.enable_visualization) {
        StopVisualization();
    }
    
    // 清理模型
    if (model) {
        delete model;
    }
    
    // 清理关键帧数据库
    for (auto* kf : keyframe_db) {
        delete kf;
    }
    
    std::cout << "\nProgram finished successfully!" << std::endl;
    return 0;
}

